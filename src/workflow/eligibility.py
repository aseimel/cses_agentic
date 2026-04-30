"""
Initial CSES eligibility review.

This is used during Step 1 so processors can decide early whether the study is
eligible for CSES processing before investing time in matching and syntax work.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from src.config import LLM_TEMPERATURE
from src.ingest.data_loader import DataLoader
from src.ingest.doc_parser import DocumentParser
from src.matching.llm_matcher import CSES_TARGET_VARIABLES
from src.settings import apply_settings_to_environment
from src.study_kb import StudyKnowledgeBase
from src.model_runtime import ModelRole, ModelTaskRunner


MAX_DOC_CHARS = 8000
MAX_VARIABLES_FOR_PROMPT = 250
EVIDENCE_WINDOW_CHARS = 450

POSITIVE_SAMPLING_TERMS = [
    "probability sample",
    "probability sampling",
    "random sample",
    "random sampling",
    "stratified random",
    "multistage",
    "multi-stage",
    "cluster sample",
    "cluster sampling",
    "sampling frame",
    "random digit dialing",
    "random-digit dialing",
    "pps",
    "probability proportional to size",
    "randomly selected",
    "selected at random",
]

NEGATIVE_SAMPLING_TERMS = [
    "non-probability",
    "nonprobability",
    "not a probability sample",
    "quota sample",
    "quota sampling",
    "convenience sample",
    "convenience sampling",
    "opt-in",
    "opt in",
    "volunteer sample",
    "voluntary sample",
    "purposive sample",
    "purposive sampling",
    "snowball sample",
    "snowball sampling",
    "river sample",
    "access panel",
]


@dataclass
class EligibilityReview:
    data_file: str = ""
    sample_size_rows: int | None = None
    variable_count: int | None = None
    metadata_quality: str = ""
    direct_cses_variable_count: int = 0
    direct_cses_variables: list[str] = field(default_factory=list)
    questionnaire_item_count: int = 0
    cses_items_included_count: int | None = None
    cses_items_evidence: str = "Not yet assessed."
    missing_cses_items: list[str] = field(default_factory=list)
    sample_design: str = ""
    target_population: str = ""
    response_rate: str = ""
    fieldwork_dates: str = ""
    mode: str = ""
    weights: str = ""
    probability_sample_assessment: str = "Unclear from available materials."
    probability_sample_status: str = "unclear"
    sampling_evidence: list[str] = field(default_factory=list)
    documented_sample_size: str = ""
    eligibility_assessment: str = "Processor review required."
    processor_eligibility_decision: str = "pending"
    collaborator_questions: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    def summary_lines(self) -> list[str]:
        lines = ["Initial CSES eligibility review:"]
        if self.data_file:
            lines.append(f"- Data file: {Path(self.data_file).name}")
        if self.sample_size_rows is not None:
            lines.append(f"- Data rows / preliminary sample size: {self.sample_size_rows}")
        if self.variable_count is not None:
            lines.append(f"- Variables in data file: {self.variable_count}")
        lines.append(f"- Direct CSES-coded variables in data: {self.direct_cses_variable_count}")
        lines.append(f"- Questionnaire items detected by parser: {self.questionnaire_item_count}")
        if self.cses_items_included_count is not None:
            lines.append(f"- CSES items included according to documentation review: {self.cses_items_included_count}")
        lines.append(f"- CSES item coverage: {self.cses_items_evidence}")
        if self.missing_cses_items:
            lines.append(f"- Missing CSES items: {', '.join(self.missing_cses_items[:50])}")
        if self.sample_design:
            lines.append(f"- Sample design: {self.sample_design}")
        if self.target_population:
            lines.append(f"- Target population: {self.target_population}")
        if self.response_rate:
            lines.append(f"- Response rate: {self.response_rate}")
        if self.fieldwork_dates:
            lines.append(f"- Fieldwork dates: {self.fieldwork_dates}")
        if self.mode:
            lines.append(f"- Mode: {self.mode}")
        if self.weights:
            lines.append(f"- Weights: {self.weights}")
        lines.append(f"- Probability sample status: {self.probability_sample_status}")
        lines.append(f"- Probability sample evidence: {self.probability_sample_assessment}")
        for evidence in self.sampling_evidence[:5]:
            lines.append(f"  Evidence: {evidence}")
        if self.documented_sample_size:
            lines.append(f"- Documented sample size: {self.documented_sample_size}")
        lines.append(f"- Eligibility assessment: {self.eligibility_assessment}")
        lines.append(f"- Processor eligibility decision: {self.processor_eligibility_decision}")
        return lines

    def to_log_message(self) -> str:
        return "\n".join(self.summary_lines())


def review_initial_eligibility(
    working_dir: Path,
    data_files: list[Path],
    questionnaire_files: list[Path],
    codebook_files: list[Path],
    design_report_files: list[Path],
) -> EligibilityReview:
    review = EligibilityReview()
    dataset_variables: list[str] = []

    if data_files:
        data_file = data_files[0]
        review.data_file = str(data_file)
        dataset = DataLoader().load(data_file)
        if dataset:
            review.sample_size_rows = dataset.n_rows
            review.variable_count = dataset.n_variables
            review.metadata_quality = dataset.metadata_quality
            dataset_variables = list(dataset.variables.keys())
            target_codes = {code.upper() for code in CSES_TARGET_VARIABLES}
            direct_hits = sorted(
                variable for variable in dataset_variables
                if variable.upper() in target_codes
            )
            review.direct_cses_variables = direct_hits
            review.direct_cses_variable_count = len(direct_hits)
        else:
            review.issues.append(f"Could not read data file to estimate sample size: {data_file.name}")
    else:
        review.issues.append("No data file available to estimate sample size.")

    parser = DocumentParser()
    document_texts: list[tuple[str, str]] = []
    for path in questionnaire_files + codebook_files + design_report_files:
        parsed = parser.parse(path)
        if not parsed:
            review.issues.append(f"Could not read documentation file: {path.name}")
            continue
        review.questionnaire_item_count += len(parsed.questions)
        if parsed.full_text.strip():
            document_texts.append((path.name, parsed.full_text[:MAX_DOC_CHARS]))

    _apply_sampling_rules(review, document_texts)
    _apply_study_kb(review, working_dir)
    _add_llm_assessment(review, dataset_variables, document_texts)
    _enforce_conservative_sampling_decision(review)
    _add_default_questions(review)
    return review


def _apply_study_kb(review: EligibilityReview, working_dir: Path) -> None:
    """Use the study KB as the first source-backed eligibility layer when present."""
    kb = StudyKnowledgeBase(working_dir)
    if not kb.exists():
        return

    sample_size = kb.get_field("sample_size")
    if sample_size and not review.documented_sample_size:
        review.documented_sample_size = sample_size[0].get("value", "")

    for attr, field_names in {
        "sample_design": ["sample_design"],
        "target_population": ["target_population", "eligible_population"],
        "response_rate": ["response_rate"],
        "fieldwork_dates": ["fieldwork_dates", "collection_period"],
        "mode": ["mode"],
        "weights": ["weights", "weighting"],
    }.items():
        if getattr(review, attr):
            continue
        for field_name in field_names:
            facts = kb.get_field(field_name)
            if facts:
                value = str(facts[0].get("value", "")).strip()
                if value:
                    setattr(review, attr, _windows_safe_text(value))
                    break

    coverage = kb.get_field("cses_item_coverage")
    missing_items = _extract_missing_cses_items(kb)
    if coverage:
        review.cses_items_evidence = coverage[0].get("value", "")
        if review.cses_items_included_count is None:
            review.cses_items_included_count = _extract_first_integer(review.cses_items_evidence)
    if missing_items:
        review.missing_cses_items = missing_items
        review.issues.append("Missing CSES questionnaire items after full search: " + ", ".join(missing_items[:60]))

    probability_status = kb.get_field("probability_sample_status")
    probability_evidence = kb.get_field("probability_sample_evidence") or kb.get_field("sample_design")
    if probability_status:
        status_text = probability_status[0].get("value", "")
        review.probability_sample_status = _normalize_probability_status(status_text)
    if probability_evidence:
        evidence_lines = []
        for fact in probability_evidence[:6]:
            value = fact.get("value", "")
            for citation in fact.get("citations", [])[:2]:
                source_id = citation.get("source_id", "KB")
                evidence = citation.get("evidence", "")
                if evidence:
                    evidence_lines.append(f"Study information {source_id}: {evidence}")
            if value and not evidence_lines:
                evidence_lines.append(f"Study information: {value}")
        if evidence_lines:
            review.sampling_evidence = _dedupe_preserve_order([*evidence_lines, *review.sampling_evidence])
        review.probability_sample_assessment = "Study information evidence: " + " | ".join(evidence_lines[:3])


def _add_llm_assessment(
    review: EligibilityReview,
    dataset_variables: list[str],
    document_texts: list[tuple[str, str]],
) -> None:
    apply_settings_to_environment()
    model = os.getenv("CSES_AGENTIC_MODEL") or os.getenv("CSES_CHAT_MODEL")
    if not model or not document_texts or not _has_credentials_for_model(model):
        return

    target_rows = "\n".join(
        f"- {code}: {description}"
        for code, description in CSES_TARGET_VARIABLES.items()
    )
    variable_text = "\n".join(dataset_variables[:MAX_VARIABLES_FOR_PROMPT])
    docs = "\n\n".join(
        f"### {name}\n{text}"
        for name, text in document_texts
    )
    sampling_evidence = "\n".join(f"- {item}" for item in review.sampling_evidence) or "(no explicit sampling evidence found)"
    prompt = f"""Review this deposited election study for initial CSES eligibility.

Assess only from the provided data/documentation excerpts. If evidence is
missing, say it is unclear and list the exact missing information for processor
review. Do not create collaborator questions.
For probability-sample status, use only these values:
confirmed_probability_sample, confirmed_non_probability_sample, unclear.
Only use confirmed_non_probability_sample if the excerpts explicitly say the
study used non-probability, quota, opt-in, convenience, volunteer, purposive, or
similar non-probability sampling. Absence of probability-sampling details means
unclear, not no.
The collaborator is not expected to provide a mapping from local variables to
CSES F-codes; that mapping is created by the processor/CSES workflow and must
not be treated as missing deposit information.
Mode is documentation information, not the core CSES eligibility criterion.

CSES Module target items:
{target_rows}

Dataset variables:
{variable_text or "(no variables loaded)"}

Sampling evidence windows extracted before model review:
{sampling_evidence}

Documentation excerpts:
{docs}

Return JSON only with these keys:
- cses_items_included_count: integer or null
- cses_items_evidence: concise explanation of which CSES items appear included
- probability_sample_status: confirmed_probability_sample, confirmed_non_probability_sample, or unclear
- probability_sample_assessment: concise assessment grounded in evidence snippets
- probability_sample_evidence_snippets: array of exact short snippets used
- probability_sample_missing_info: missing details, or empty string
- documented_sample_size: sample size stated in documentation, or empty string
- eligibility_assessment: concise processor-facing eligibility assessment
- processor_review_items: array of exact unresolved items for processor review
"""

    try:
        response = ModelTaskRunner().response(
            ModelRole.AGENTIC,
            model_override=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=LLM_TEMPERATURE,
            max_tokens=1200,
            drop_params=True,
            timeout=120,
            purpose="Initial eligibility review",
        )
        content = response.choices[0].message.content or "{}"
        data = _parse_json_object(content)
    except Exception as exc:
        # The deterministic evidence and Study KB layers still provide the
        # processor-facing answer. Model endpoint failures are recorded in
        # model diagnostics by ModelTaskRunner and should not become CSES
        # missing-information claims.
        return

    if isinstance(data.get("cses_items_included_count"), int):
        review.cses_items_included_count = data["cses_items_included_count"]
    for key in [
        "cses_items_evidence",
        "documented_sample_size",
        "eligibility_assessment",
    ]:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            setattr(review, key, _windows_safe_text(value.strip()))
    status = data.get("probability_sample_status")
    assessment = data.get("probability_sample_assessment")
    snippets = data.get("probability_sample_evidence_snippets")
    missing_info = data.get("probability_sample_missing_info")
    if isinstance(status, str) and status.strip():
        review.probability_sample_status = _normalize_probability_status(status)
    if isinstance(assessment, str) and assessment.strip():
        review.probability_sample_assessment = _windows_safe_text(assessment.strip())
    if isinstance(snippets, list):
        review.sampling_evidence.extend(_windows_safe_text(str(item).strip()) for item in snippets if str(item).strip())
    if isinstance(missing_info, str) and missing_info.strip():
        review.issues.append(f"Missing sampling information: {_windows_safe_text(missing_info.strip())}")
    review_items = data.get("processor_review_items")
    if isinstance(review_items, list):
        review.issues.extend(
            _windows_safe_text(str(item).strip())
            for item in review_items
            if str(item).strip()
        )
    questions = data.get("collaborator_questions")
    if isinstance(questions, list):
        # Backward compatible guard for older prompts/model behavior. These are
        # processor-review items, not ready-to-send collaborator questions.
        review.issues.extend(
            _windows_safe_text(str(q).strip())
            for q in questions
            if str(q).strip()
        )
    review.issues = [
        issue for issue in review.issues
        if not _is_invalid_deposit_requirement(issue)
    ]


def _windows_safe_text(text: str) -> str:
    replacements = {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2190": "<-",
        "\u2192": "->",
        "\u2713": "OK",
        "\u2714": "OK",
        "\u2705": "OK",
        "\ufe0f": "",
        "\u202f": " ",
        "\u00a0": " ",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text.encode("cp1252", errors="replace").decode("cp1252")


def _parse_json_object(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:].strip()
    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end >= start:
        content = content[start:end + 1]
    return json.loads(content)


def _apply_sampling_rules(review: EligibilityReview, document_texts: list[tuple[str, str]]) -> None:
    positive = _find_sampling_evidence(document_texts, POSITIVE_SAMPLING_TERMS)
    negative = [
        item for item in _find_sampling_evidence(document_texts, NEGATIVE_SAMPLING_TERMS)
        if not _is_unselected_or_definition_only_sampling_snippet(item)
    ]
    review.sampling_evidence = _dedupe_preserve_order([*positive, *negative])

    if positive and not negative:
        review.probability_sample_status = "confirmed_probability_sample"
        review.probability_sample_assessment = (
            "Evidence suggests a probability sample. Processor should confirm against the extracted sampling evidence."
        )
        return
    if negative and not positive:
        review.probability_sample_status = "confirmed_non_probability_sample"
        review.probability_sample_assessment = (
            "Documentation contains explicit non-probability sampling evidence. Processor review required before marking the study ineligible."
        )
        return
    if positive and negative:
        review.probability_sample_status = "unclear"
        review.probability_sample_assessment = (
            "Sampling evidence is contradictory: both probability-sampling and non-probability terms appear in the documentation."
        )
        review.issues.append("Sampling evidence is contradictory and requires processor review.")
        return
    review.probability_sample_status = "unclear"
    review.probability_sample_assessment = "No explicit sampling-method evidence was found in the available excerpts."


def _is_unselected_or_definition_only_sampling_snippet(snippet: str) -> bool:
    text = snippet.casefold()
    if "[ ]" in text and any(term in text for term in ["quota", "non-sample", "substitution", "replacement"]):
        return True
    if "definitions:" in text and any(term in text for term in ["quota sampling", "cluster sampling", "stratification"]):
        return True
    if "if no, please describe" in text and "non-probability" in text:
        return True
    if "in quota sampling" in text:
        return True
    return False


def _find_sampling_evidence(document_texts: list[tuple[str, str]], terms: list[str]) -> list[str]:
    evidence = []
    seen_snippets = set()
    for filename, text in document_texts:
        lower_text = text.casefold()
        for term in terms:
            start = 0
            term_lower = term.casefold()
            while True:
                index = lower_text.find(term_lower, start)
                if index < 0:
                    break
                snippet = _clean_evidence_snippet(text, index, len(term))
                normalized = snippet.casefold()
                if normalized not in seen_snippets:
                    evidence.append(f"{filename} [{term}]: {snippet}")
                    seen_snippets.add(normalized)
                start = index + len(term)
                if len(evidence) >= 12:
                    return evidence
    return evidence


def _clean_evidence_snippet(text: str, index: int, term_length: int) -> str:
    left = max(0, index - EVIDENCE_WINDOW_CHARS // 2)
    right = min(len(text), index + term_length + EVIDENCE_WINDOW_CHARS // 2)

    sentence_left = max(text.rfind(".", 0, index), text.rfind("\n", 0, index))
    if sentence_left >= 0 and index - sentence_left < EVIDENCE_WINDOW_CHARS // 2:
        left = sentence_left + 1

    sentence_right_candidates = [
        pos for pos in [text.find(".", index + term_length), text.find("\n", index + term_length)]
        if pos >= 0
    ]
    if sentence_right_candidates:
        sentence_right = min(sentence_right_candidates)
        if sentence_right - index < EVIDENCE_WINDOW_CHARS // 2:
            right = sentence_right + 1

    return " ".join(text[left:right].split())


def _normalize_probability_status(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"yes", "probability", "probability_sample", "confirmed_probability_sample"}:
        return "confirmed_probability_sample"
    if normalized in {"no", "non_probability", "nonprobability", "confirmed_non_probability_sample"}:
        return "confirmed_non_probability_sample"
    return "unclear"


def _extract_missing_cses_items(kb: StudyKnowledgeBase) -> list[str]:
    items: list[str] = []
    for fact in kb.get_field("missing_cses_items"):
        value = fact.get("value", "")
        if isinstance(value, list):
            items.extend(str(item) for item in value)
        else:
            items.extend(_split_missing_items(str(value)))
    for item in kb.missing_fields():
        text = str(item).strip()
        upper = text.upper()
        if upper.startswith(("A", "B", "C", "D", "E", "F")) and any(char.isdigit() for char in upper[:8]):
            items.append(text)
        elif "CSES ITEM" in upper or "QUESTIONNAIRE ITEM" in upper:
            items.append(text)
    return _dedupe_preserve_order([
        _windows_safe_text(item.strip())
        for item in items
        if item.strip()
        and not _is_invalid_deposit_requirement(item)
        and not _means_no_missing_items(item)
    ])


def _split_missing_items(value: str) -> list[str]:
    cleaned = value.strip()
    if not cleaned:
        return []
    for sep in [";", "\n", "|"]:
        cleaned = cleaned.replace(sep, ",")
    return [part.strip(" -") for part in cleaned.split(",") if part.strip(" -")]


def _extract_first_integer(value: str) -> int | None:
    import re

    text = value or ""
    match = re.search(r"\b(\d{1,3})\s+(?:cses\s+)?(?:items|questions)\b", text, re.IGNORECASE)
    if not match:
        match = re.search(r"\b(?:items|questions)\s+(?:included|covered|administered)\s*[:=]\s*(\d{1,3})\b", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _is_invalid_deposit_requirement(text: str) -> bool:
    lowered = text.casefold()
    invalid_fragments = [
        "mapping of local variable names",
        "mapping from local variables",
        "mapping to cses",
        "cses standard f-codes",
        "cses f-codes",
        "f-code mapping",
        "explicit mapping",
    ]
    return any(fragment in lowered for fragment in invalid_fragments)


def _means_no_missing_items(text: str) -> bool:
    lowered = text.strip().casefold().strip(".")
    return lowered in {
        "none",
        "none missing",
        "no missing",
        "no missing items",
        "not applicable",
        "n/a",
    }


def _enforce_conservative_sampling_decision(review: EligibilityReview) -> None:
    evidence_text = "\n".join(review.sampling_evidence).casefold()
    has_positive = any(term.casefold() in evidence_text for term in POSITIVE_SAMPLING_TERMS)
    substantive_negative_evidence = [
        item for item in review.sampling_evidence
        if any(term.casefold() in item.casefold() for term in NEGATIVE_SAMPLING_TERMS)
        and not _is_unselected_or_definition_only_sampling_snippet(item)
    ]
    has_negative = bool(substantive_negative_evidence)

    if review.probability_sample_status == "confirmed_non_probability_sample" and not has_negative:
        review.probability_sample_status = "unclear"
        review.probability_sample_assessment = (
            "Unclear from available materials. The model suggested non-probability sampling, but no explicit non-probability evidence was extracted."
        )
        review.issues.append(
            "Model non-probability classification downgraded to unclear because explicit negative sampling evidence was not found."
        )
    if has_positive and not has_negative and review.probability_sample_status != "confirmed_probability_sample":
        review.probability_sample_status = "confirmed_probability_sample"
        review.probability_sample_assessment = (
            "Evidence suggests a probability sample. Processor should confirm against the extracted sampling evidence."
        )
    if has_positive and not has_negative:
        review.issues = [
            issue for issue in review.issues
            if "sampling evidence is contradictory" not in issue.casefold()
            and "probability sample eligibility is unclear" not in issue.casefold()
        ]
    if has_positive and has_negative:
        review.probability_sample_status = "unclear"


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    result = []
    seen = set()
    for item in items:
        key = item.casefold()
        if key not in seen:
            result.append(item)
            seen.add(key)
    return result


def _add_default_questions(review: EligibilityReview) -> None:
    if review.cses_items_included_count is None and review.direct_cses_variable_count == 0 and not review.missing_cses_items:
        review.issues.append("CSES questionnaire item coverage needs processor review after source inspection.")
    if review.probability_sample_status == "unclear":
        review.issues.append("Probability-sample eligibility needs processor review.")
    if review.sample_size_rows is None and not review.documented_sample_size:
        review.issues.append("Final respondent sample size is not available from the deposited data/documentation.")

    deduped = []
    seen = set()
    for question in review.collaborator_questions:
        normalized = question.lower()
        if normalized not in seen:
            deduped.append(question)
            seen.add(normalized)
    review.collaborator_questions = deduped


def _short_error(exc: Exception) -> str:
    text = str(exc)
    lowered = text.lower()
    if "<!doctype html" in lowered or "<html" in lowered:
        if "cloudflare" in lowered or "just a moment" in lowered:
            return "model endpoint returned a Cloudflare/browser challenge instead of an API response"
        return "model endpoint returned HTML instead of an API response"
    text = " ".join(text.split())
    if len(text) > 240:
        text = text[:237] + "..."
    return text


def _has_credentials_for_model(model: str) -> bool:
    lowered = model.lower()
    if lowered.startswith("xai/"):
        return bool(os.getenv("XAI_API_KEY"))
    if lowered.startswith("anthropic/"):
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    if lowered.startswith("gemini/") or lowered.startswith("google/"):
        return bool(os.getenv("GEMINI_API_KEY"))
    if lowered.startswith("openai/") or "/" not in lowered:
        return bool(os.getenv("OPENAI_API_KEY"))
    return bool(
        os.getenv("OPENAI_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("XAI_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    )

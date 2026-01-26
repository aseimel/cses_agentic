"""
Document aggregator for CSES variable mapping.

Three-stage parallel pipeline:
- Stage 1: Extract variable info from EACH document in parallel (llama4)
- Stage 2: Aggregate extractions into single condensed TOON (llama4)
- Stage 3: Semantic matching (gpt-oss:120b) - handled by llm_matcher.py

This approach is faster because Stage 1 runs in parallel across all documents.
"""

import os
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Callable
from litellm import completion

from src.utils.toon_encoder import encode_table

logger = logging.getLogger(__name__)


class DocumentAggregator:
    """
    Aggregates information from multiple documents per variable.

    Uses parallel processing to extract from each document separately,
    then aggregates into a single condensed summary.
    """

    def __init__(self, model: Optional[str] = None):
        """Initialize the aggregator."""
        self.model = model or os.getenv("LLM_MODEL_PREPROCESS", "openai/llama4:latest")
        self.temperature = 0
        logger.info(f"DocumentAggregator initialized with model: {self.model}")

    def aggregate_variable_info(
        self,
        source_variables: list[dict],
        codebook_text: str = "",
        questionnaire_text: str = "",
        design_report_text: str = "",
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Main entry point - orchestrates the 3-stage pipeline.

        Args:
            source_variables: List of dicts with 'name', 'description', 'value_labels'
            codebook_text: Full text of codebook document
            questionnaire_text: Full text of questionnaire document
            design_report_text: Full text of design report document
            progress_callback: Optional callback for progress updates

        Returns:
            TOON-formatted summary file content with per-variable information
        """
        def update_progress(msg: str):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        if not source_variables:
            logger.warning("No source variables provided")
            return "# No variables to aggregate\nvars[0]{name|desc|labels}:"

        var_names = [v.get('name', '?') for v in source_variables]
        update_progress(f"Starting pipeline for {len(var_names)} variables")

        # Build list of documents to process
        documents = []
        if codebook_text:
            documents.append(("codebook", codebook_text))
        if questionnaire_text:
            documents.append(("questionnaire", questionnaire_text))
        if design_report_text:
            documents.append(("design_report", design_report_text))

        update_progress(f"Documents to process: {[d[0] for d in documents]}")

        if not documents:
            logger.warning("No documents provided - using only source metadata")
            return self._format_as_toon({}, source_variables)

        # STAGE 1: Parallel extraction from each document
        update_progress("=== STAGE 1: Parallel per-document extraction ===")
        start_time = time.time()
        extractions = self._extract_parallel(var_names, documents, update_progress)
        stage1_time = time.time() - start_time
        update_progress(f"Stage 1 complete in {stage1_time:.1f}s")

        # STAGE 2: Aggregate extractions into single condensed TOON
        update_progress("=== STAGE 2: Aggregating extractions ===")
        start_time = time.time()
        aggregated = self._aggregate_extractions(var_names, extractions, source_variables, update_progress)
        stage2_time = time.time() - start_time
        update_progress(f"Stage 2 complete in {stage2_time:.1f}s")

        # Format as TOON
        toon = self._format_as_toon(aggregated, source_variables)
        update_progress(f"Final TOON size: {len(toon)} chars")
        return toon

    def _extract_parallel(
        self,
        var_names: list[str],
        documents: list[tuple[str, str]],
        update_progress: Callable[[str], None]
    ) -> dict[str, dict[str, dict]]:
        """Extract variable info from each document in parallel."""
        extractions = {}
        max_workers = min(len(documents), 4)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_doc = {
                executor.submit(
                    self._extract_from_document,
                    var_names,
                    doc_type,
                    doc_text
                ): doc_type
                for doc_type, doc_text in documents
            }

            for future in as_completed(future_to_doc):
                doc_type = future_to_doc[future]
                try:
                    result = future.result()
                    extractions[doc_type] = result
                    update_progress(f"  {doc_type}: extracted {len(result)} variables")
                except Exception as e:
                    logger.error(f"  {doc_type}: extraction failed - {e}")
                    extractions[doc_type] = {}

        return extractions

    def _extract_from_document(
        self,
        var_names: list[str],
        doc_type: str,
        doc_text: str
    ) -> dict[str, dict]:
        """
        Extract variable information from a single document in ONE LLM call.

        Focuses on extracting QUESTION TEXT for each variable, which is critical
        for matching CSES survey data.
        """
        logger.info(f"Extracting from {doc_type} ({len(doc_text)} chars) - single prompt")

        # Customize prompt based on document type
        if doc_type == "codebook":
            focus = """For each variable, extract:
- The EXACT question text (the survey question asked to respondents)
- Variable description/label
- Value labels (response options with codes)"""
        elif doc_type == "questionnaire":
            focus = """For each variable, extract:
- The EXACT question text as it appears in the survey
- Response options/answer categories
- Any skip patterns or conditions"""
        else:
            focus = """For each variable, extract any available information about:
- Question text
- Variable meaning
- Response categories"""

        var_list = "\n".join([f"  - {v}" for v in var_names])

        prompt = f"""You are extracting survey variable information from a {doc_type} document.

VARIABLES TO FIND (these are variable names from the dataset):
{var_list}

DOCUMENT CONTENT:
{doc_text}

TASK: {focus}

IMPORTANT: This is CSES (Comparative Study of Electoral Systems) data. Variables often have:
- Question codes like Q1, Q2, D1, D2, etc.
- Standardized question texts about political attitudes, voting, demographics
- Response scales (1-5, 1-7, etc.) with labeled categories

Return JSON with info for variables found:
{{
  "variables": {{
    "VAR_NAME": {{
      "question_text": "The exact survey question text",
      "desc": "Brief description if different from question",
      "labels": "1=Strongly agree;2=Agree;3=Neither;4=Disagree;5=Strongly disagree"
    }},
    ...
  }}
}}

RULES:
- PRIORITIZE extracting the actual question text - this is most important for matching
- Use EXACT variable names (case-sensitive)
- Include full value labels with codes
- Return ONLY valid JSON"""

        return self._call_llm_for_extraction(prompt, var_names)

    def _call_llm_for_extraction(
        self,
        prompt: str,
        var_names: list[str]
    ) -> dict[str, dict]:
        """Call LLM and parse extraction response."""
        max_retries = 2
        retry_delay = 3

        # Calculate max_tokens - need enough for all variables
        # ~150 tokens per variable (name + question_text + desc + labels)
        max_tokens = max(8192, min(len(var_names) * 150, 16384))

        for attempt in range(max_retries):
            try:
                response = completion(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )

                text = response.choices[0].message.content or ""

                # Extract JSON
                if '{' in text:
                    start = text.find('{')
                    depth = 0
                    end = start
                    for j, c in enumerate(text[start:], start):
                        if c == '{':
                            depth += 1
                        elif c == '}':
                            depth -= 1
                            if depth == 0:
                                end = j + 1
                                break

                    if end > start:
                        data = json.loads(text[start:end])
                        extracted = data.get('variables', {})

                        # Normalize labels format
                        for var_name, var_data in extracted.items():
                            if isinstance(var_data.get('labels'), dict):
                                labels_dict = var_data['labels']
                                var_data['labels'] = ';'.join(
                                    f"{k}={v}" for k, v in labels_dict.items()
                                )
                        return extracted

            except Exception as e:
                logger.warning(f"Extraction attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2

        return {}

    def _aggregate_extractions(
        self,
        var_names: list[str],
        extractions: dict[str, dict[str, dict]],
        source_variables: list[dict],
        update_progress: Callable[[str], None]
    ) -> dict[str, dict]:
        """
        Aggregate extractions from multiple documents into one.

        For single document, just uses simple merge.
        For multiple documents, combines information intelligently.
        """
        # Simple merge - prioritize question_text field
        merged = self._simple_merge(var_names, extractions, source_variables)

        # Count stats
        has_question = sum(1 for v in merged.values() if v.get('question_text'))
        has_desc = sum(1 for v in merged.values() if v.get('desc'))
        has_labels = sum(1 for v in merged.values() if v.get('labels'))

        update_progress(f"  Merged: {has_question} with question_text, {has_desc} with desc, {has_labels} with labels")

        return merged

    def _simple_merge(
        self,
        var_names: list[str],
        extractions: dict[str, dict[str, dict]],
        source_variables: list[dict]
    ) -> dict[str, dict]:
        """Simple merge: prioritize codebook > questionnaire > design_report > source."""
        merged = {}
        source_lookup = {v.get('name', ''): v for v in source_variables}
        priority = ['codebook', 'questionnaire', 'design_report']

        for var_name in var_names:
            question_text = ""
            desc = ""
            labels = ""

            # Try each source in priority order
            for doc_type in priority:
                if doc_type in extractions:
                    var_data = extractions[doc_type].get(var_name, {})
                    if not question_text and var_data.get('question_text'):
                        question_text = var_data['question_text']
                    if not desc and var_data.get('desc'):
                        desc = var_data['desc']
                    if not labels and var_data.get('labels'):
                        labels = var_data['labels']

            # Fallback to source metadata
            if var_name in source_lookup:
                src = source_lookup[var_name]
                if not desc:
                    desc = src.get('description', '')
                if not labels and src.get('value_labels'):
                    labels = self._format_labels(src['value_labels'])

            merged[var_name] = {
                'question_text': question_text,
                'desc': desc,
                'labels': labels
            }

        return merged

    def _format_as_toon(
        self,
        aggregated: dict[str, dict],
        source_variables: list[dict]
    ) -> str:
        """
        Format aggregated info as TOON table.

        Includes question_text as primary field for matching.
        """
        rows = []
        source_lookup = {v.get('name', ''): v for v in source_variables}

        for var in source_variables:
            name = var.get('name', '?')
            agg = aggregated.get(name, {})

            # Get aggregated data
            question_text = agg.get('question_text', '')
            desc = agg.get('desc', '')
            if not desc:
                desc = var.get('description', '')

            # Combine question_text and desc for maximum info
            full_desc = question_text if question_text else desc
            if question_text and desc and desc not in question_text:
                full_desc = f"{question_text} [{desc}]"

            labels = agg.get('labels', '')
            if not labels:
                labels = self._format_labels(var.get('value_labels', {}))

            rows.append({
                'name': name,
                'desc': (full_desc or '')[:300],  # Allow longer for question text
                'labels': (labels or '')[:150]
            })

        header = f"""# Variable Summary (3-stage parallel pipeline)
# Variables: {len(rows)}
# Model: {self.model}
# Format: TOON table - name|description(includes question text)|labels

"""
        return header + 'vars' + encode_table(rows, ['name', 'desc', 'labels'], '|')

    def _format_labels(self, labels: dict) -> str:
        """Format value labels dictionary as compact string."""
        if not labels or not isinstance(labels, dict):
            return ''
        items = list(labels.items())[:10]
        return ';'.join(f"{k}={v}" for k, v in items)


def create_aggregator(model: Optional[str] = None) -> DocumentAggregator:
    """Factory function to create a DocumentAggregator instance."""
    return DocumentAggregator(model=model)

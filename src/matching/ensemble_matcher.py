"""
Multi-model ensemble matcher for CSES variable matching with quorum voting.

Uses 3 diverse models in parallel to match variables, requiring 2/3 agreement
(quorum threshold) to accept a match. This approach:
- Reduces hallucinations by having models check each other
- Provides higher confidence matches through multi-model agreement
- Better NOT_FOUND detection when models disagree
- Combines diverse reasoning from different model perspectives

Research shows majority voting captures most benefits of ensemble approaches,
with 33% improvement using 2-model ensemble, up to F1=0.92 with more models.
"""

import json
import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional, Callable

from src.model_runtime import ModelRole, ModelTaskRunner

from src.config import ENSEMBLE_MODELS, QUORUM_THRESHOLD, LLM_TEMPERATURE
from src.matching.llm_matcher import CSES_TARGET_VARIABLES, MatchProposal, MatchingResult
from src.settings import apply_settings_to_environment
from src.utils.toon_encoder import encode_table

logger = logging.getLogger(__name__)


@dataclass
class ModelVote:
    """A single model's vote for a variable mapping."""
    model: str
    source: str
    confidence: float
    reasoning: str


@dataclass
class EnsembleMapping:
    """Result of ensemble voting for a single CSES target."""
    target: str
    source: str
    vote_ratio: float
    model_votes: dict[str, ModelVote]
    combined_reasoning: str
    needs_review: bool


class EnsembleMatcher:
    """
    Multi-model ensemble for variable matching with quorum voting.

    Uses 3 diverse models (different architectures) to independently match
    all variables, then votes on results. A match is accepted if at least
    2/3 of models agree (quorum threshold).
    """

    def __init__(self, models: Optional[list[str]] = None, quorum_threshold: Optional[float] = None, working_dir=None):
        """
        Initialize the ensemble matcher.

        Args:
            models: List of model identifiers to use (defaults to ENSEMBLE_MODELS from config)
            quorum_threshold: Minimum vote ratio to accept match (defaults to QUORUM_THRESHOLD)
        """
        self.models = models or ENSEMBLE_MODELS
        if models is None:
            self.models = ModelTaskRunner().models_for_ensemble()
        self.quorum_threshold = quorum_threshold or QUORUM_THRESHOLD
        self.temperature = LLM_TEMPERATURE
        from pathlib import Path
        self.runner = ModelTaskRunner(Path(working_dir) if working_dir else Path.cwd())
        apply_settings_to_environment()

        logger.info(f"EnsembleMatcher initialized with {len(self.models)} models: {self.models}")
        logger.info(f"Quorum threshold: {self.quorum_threshold}")

    def match_with_quorum(
        self,
        source_contexts: list[dict],
        pre_aggregated_summary: str,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> MatchingResult:
        """
        Run matching on all models in parallel, vote on results.

        Args:
            source_contexts: List of dicts with variable metadata from data file
            pre_aggregated_summary: TOON summary from DocumentAggregator
            progress_callback: Optional callback for progress updates

        Returns:
            MatchingResult with proposals, confidence scores, and voting details
        """
        def update_progress(msg: str):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        cses_targets = list(CSES_TARGET_VARIABLES.keys())
        update_progress(f"Starting ensemble matching with {len(self.models)} models for {len(cses_targets)} targets")

        # 1. Launch all models in parallel
        update_progress("Phase 1: Running parallel model matching...")
        model_results = {}

        with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
            future_to_model = {
                executor.submit(
                    self._single_model_match,
                    model,
                    cses_targets,
                    pre_aggregated_summary
                ): model
                for model in self.models
            }

            for future in as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    result = future.result()
                    model_results[model] = result
                    matched_count = len([m for m in result if m.get("source") not in ["NOT_FOUND", "ERROR"]])
                    update_progress(f"  {model}: matched {matched_count}/{len(cses_targets)} targets")
                except Exception as e:
                    logger.error(f"  {model}: matching failed - {e}")
                    model_results[model] = []

        # 2. Vote on each CSES target
        update_progress("Phase 2: Voting on results...")
        result = MatchingResult()

        for target in cses_targets:
            votes = self._collect_votes(model_results, target)
            consensus = self._determine_consensus(votes)

            # Determine confidence level based on vote ratio
            if consensus["vote_ratio"] >= 1.0:
                confidence_level = "high"
                confidence = 0.95
            elif consensus["vote_ratio"] >= self.quorum_threshold:
                confidence_level = "high" if consensus["avg_confidence"] >= 0.85 else "medium"
                confidence = consensus["avg_confidence"]
            else:
                confidence_level = "low"
                confidence = consensus["avg_confidence"] * consensus["vote_ratio"]

            needs_review = consensus["vote_ratio"] < self.quorum_threshold or confidence_level != "high"

            proposal = MatchProposal(
                source_variable=consensus["source"],
                target_variable=target,
                confidence=confidence,
                confidence_level=confidence_level,
                reasoning=consensus["combined_reasoning"],
                matched_by="ensemble_quorum",
                needs_review=needs_review
            )

            result.proposals.append(proposal)

            if confidence_level == "high":
                result.high_confidence_count += 1
            elif confidence_level == "medium":
                result.medium_confidence_count += 1
            else:
                result.low_confidence_count += 1

            if consensus["source"] == "NOT_FOUND":
                result.unmatched.append(target)

        # Summary
        matched_count = len([p for p in result.proposals if p.source_variable not in ["NOT_FOUND", "ERROR"]])
        unanimous = len([p for p in result.proposals if "unanimous" in p.reasoning.lower()])
        update_progress(f"Ensemble complete: {matched_count}/{len(cses_targets)} matched, {unanimous} unanimous")

        return result

    def _single_model_match(
        self,
        model: str,
        cses_targets: list[str],
        source_index: str
    ) -> list[dict]:
        """
        Match ALL CSES targets using a single model.

        Args:
            model: Model identifier to use
            cses_targets: List of CSES target variable codes
            source_index: TOON-formatted source variable summary

        Returns:
            List of mapping dicts with target, source, confidence, reasoning
        """
        # Build TOON table for all targets
        target_rows = [{'code': c, 'desc': CSES_TARGET_VARIABLES[c]} for c in cses_targets]
        targets_toon = 'targets' + encode_table(target_rows, ['code', 'desc'], '|')

        prompt = f"""You are matching source survey variables to CSES (Comparative Study of Electoral Systems) Module 6 target variables.

=== CSES TARGET VARIABLES (what we need to create) ===
{targets_toon}

=== AVAILABLE SOURCE VARIABLES IN DEPOSITED DATA ===
{source_index}

=== MATCHING INSTRUCTIONS ===

For EACH CSES target, find the source variable that measures the SAME concept:

1. DEMOGRAPHICS (F2001-F2021):
   - F2001_Y (Year of birth): Look for birth year, age, DOB variables
   - F2001_A (Age): Look for age in years or calculated from birth year
   - F2002 (Gender): Look for sex, gender with Male/Female coding
   - F2003 (Education): Look for education level, degree variables
   - F2004 (Marital status): Look for married, single, partnership
   - F2005 (Union membership): Look for trade union membership
   - F2006 (Employment): Look for job, work status variables

2. SURVEY QUESTIONS (F3001-F3024):
   - F3001 (Political interest): "How interested are you in politics?"
   - F3002_* (Media usage): TV, radio, newspaper, social media
   - F3006 (How democratic): "How democratic is [country]?" (0-10)
   - F3007_* (Trust): Trust in parliament, government, courts, parties
   - F3010 (Turnout): "Did you vote in the election?"
   - F3018_* (Party likes): Like/dislike scale for parties (0-10)
   - F3020 (Self left-right): Respondent's left-right position (0-10)
   - F3024 (Satisfaction): Satisfaction with democracy

3. MATCHING RULES:
   - Match by CONCEPT, not just name similarity
   - Value labels are strong indicators (1=Male,2=Female -> F2002)
   - If confident, propose with 0.90+ confidence
   - If uncertain but plausible, propose with 0.60-0.89 confidence
   - Use "NOT_FOUND" ONLY if no variable measures this concept

Return JSON with ALL {len(cses_targets)} mappings:
{{"mappings":[
  {{"target":"F2001_Y","source":"D01b","confidence":0.95,"reasoning":"D01b asks year of birth"}},
  {{"target":"F2002","source":"TSEX","confidence":0.98,"reasoning":"TSEX measures gender with 1=Male,2=Female"}},
  ...
]}}

Return ONLY valid JSON, no other text."""

        try:
            runner = self.runner
            response = runner.response(
                ModelRole.MATCH_ENSEMBLE,
                max_tokens=8192,
                temperature=self.temperature,
                timeout=120,
                purpose=f"Ensemble variable matching with {model}",
                include_shared_context=True,
                model_override=model,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.choices[0].message.content.strip() if response.choices[0].message.content else ""

            # Parse JSON
            json_data = self._extract_json(response_text)

            if json_data is None:
                logger.error(f"JSON parse failed for {model}")
                return [{"target": t, "source": "ERROR", "confidence": 0.0, "reasoning": "JSON parse error"}
                        for t in cses_targets]

            return json_data.get("mappings", [])

        except Exception as e:
            logger.error(f"Model {model} matching failed: {e}")
            return [{"target": t, "source": "ERROR", "confidence": 0.0, "reasoning": str(e)[:100]}
                    for t in cses_targets]

    def _connection_kwargs(self) -> dict:
        import os

        kwargs = {}
        api_base = os.environ.get("OPENAI_API_BASE", "").strip()
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if api_base:
            kwargs["api_base"] = api_base
        if api_key:
            kwargs["api_key"] = api_key
        return kwargs

    def _collect_votes(
        self,
        model_results: dict[str, list[dict]],
        target: str
    ) -> dict[str, ModelVote]:
        """Collect votes from all models for a specific target."""
        votes = {}

        for model, mappings in model_results.items():
            # Find this target in the model's results
            for mapping in mappings:
                if mapping.get("target") == target:
                    votes[model] = ModelVote(
                        model=model,
                        source=mapping.get("source", "NOT_FOUND"),
                        confidence=float(mapping.get("confidence", 0.0)),
                        reasoning=mapping.get("reasoning", "")
                    )
                    break
            else:
                # Target not in results
                votes[model] = ModelVote(
                    model=model,
                    source="NOT_FOUND",
                    confidence=0.0,
                    reasoning="Not returned by model"
                )

        return votes

    def _determine_consensus(self, votes: dict[str, ModelVote]) -> dict:
        """
        Determine consensus from model votes.

        Uses a match-preferring strategy:
        - If multiple models agree on an actual variable, use it
        - If only one model proposes a match, use it but flag for review
        - Only return NOT_FOUND if ALL models agree it's not found

        This prevents conservative models from drowning out valid matches.

        Args:
            votes: Dict mapping model name to ModelVote

        Returns:
            Dict with source, vote_ratio, avg_confidence, combined_reasoning
        """
        # Filter out ERROR votes - failed models shouldn't affect quorum
        valid_votes = {m: v for m, v in votes.items() if v.source != "ERROR"}

        if not valid_votes:
            # All models failed
            return {
                "source": "ERROR",
                "vote_ratio": 0.0,
                "avg_confidence": 0.0,
                "combined_reasoning": "All models failed to return valid results"
            }

        # Separate actual matches from NOT_FOUND votes
        match_votes = {m: v for m, v in valid_votes.items() if v.source != "NOT_FOUND"}
        not_found_votes = {m: v for m, v in valid_votes.items() if v.source == "NOT_FOUND"}

        # Count votes for actual matches only
        if match_votes:
            match_counts = Counter(v.source for v in match_votes.values())
            best_match, match_count = match_counts.most_common(1)[0]

            # If 2+ models agree on the same actual variable, use it
            if match_count >= 2:
                vote_ratio = match_count / len(valid_votes)
                winning_votes = [v for v in match_votes.values() if v.source == best_match]
                avg_confidence = sum(v.confidence for v in winning_votes) / len(winning_votes)
                combined_reasoning = self._merge_reasoning(votes, best_match, vote_ratio, len(valid_votes))
                return {
                    "source": best_match,
                    "vote_ratio": vote_ratio,
                    "avg_confidence": avg_confidence,
                    "combined_reasoning": combined_reasoning
                }

            # If only 1 model proposes a match, use it but with lower confidence
            # (Better to have a match that needs review than to lose data)
            if match_count == 1 and len(not_found_votes) < len(valid_votes):
                vote_ratio = 1 / len(valid_votes)
                winning_vote = list(match_votes.values())[0]
                # Lower confidence since only 1 model agrees
                avg_confidence = winning_vote.confidence * 0.6
                combined_reasoning = self._merge_reasoning(votes, best_match, vote_ratio, len(valid_votes))
                combined_reasoning = "[SINGLE MODEL MATCH - NEEDS REVIEW]\n" + combined_reasoning
                return {
                    "source": best_match,
                    "vote_ratio": vote_ratio,
                    "avg_confidence": avg_confidence,
                    "combined_reasoning": combined_reasoning
                }

        # All models agree on NOT_FOUND
        if len(not_found_votes) == len(valid_votes):
            combined_reasoning = self._merge_reasoning(votes, "NOT_FOUND", 1.0, len(valid_votes))
            return {
                "source": "NOT_FOUND",
                "vote_ratio": 1.0,
                "avg_confidence": 0.95,
                "combined_reasoning": combined_reasoning
            }

        # Fallback: use original voting logic
        vote_counts = Counter(v.source for v in valid_votes.values())
        winner, count = vote_counts.most_common(1)[0]
        vote_ratio = count / len(valid_votes)

        # Calculate average confidence for winning source
        winning_votes = [v for v in valid_votes.values() if v.source == winner]
        avg_confidence = sum(v.confidence for v in winning_votes) / len(winning_votes) if winning_votes else 0.0

        # Combine reasoning from agreeing models
        combined_reasoning = self._merge_reasoning(votes, winner, vote_ratio, len(valid_votes))

        return {
            "source": winner if vote_ratio >= self.quorum_threshold else "NO_CONSENSUS",
            "vote_ratio": vote_ratio,
            "avg_confidence": avg_confidence,
            "combined_reasoning": combined_reasoning
        }

    def _merge_reasoning(
        self,
        votes: dict[str, ModelVote],
        winner: str,
        vote_ratio: float,
        valid_count: int = None
    ) -> str:
        """Merge reasoning from multiple models into a coherent explanation."""
        parts = []

        # Use valid_count if provided (excludes ERROR votes)
        total_valid = valid_count if valid_count is not None else len(votes)
        agreeing = [m for m, v in votes.items() if v.source == winner and v.source != "ERROR"]

        # Indicate consensus level
        if vote_ratio >= 1.0:
            parts.append(f"[UNANIMOUS] All {total_valid} valid models agree: {winner}")
        elif vote_ratio >= self.quorum_threshold:
            parts.append(f"[QUORUM {int(vote_ratio*100)}%] {len(agreeing)}/{total_valid} models agree: {winner}")
        else:
            parts.append(f"[NO CONSENSUS] Only {int(vote_ratio*100)}% agreement among {total_valid} valid models")

        # Note any failed models
        failed_models = [m for m, v in votes.items() if v.source == "ERROR"]
        if failed_models:
            short_names = [m.split("/")[-1].split(":")[0] for m in failed_models]
            parts.append(f"  (Models failed: {', '.join(short_names)})")

        # Add reasoning from agreeing models
        for model, vote in votes.items():
            if vote.source == winner and vote.reasoning and vote.source != "ERROR":
                # Extract model name for brevity
                short_name = model.split("/")[-1].split(":")[0]
                parts.append(f"  - {short_name}: {vote.reasoning[:100]}")

        return "\n".join(parts)

    def _extract_json(self, text: str) -> Optional[dict]:
        """Extract JSON from LLM response."""
        # Try code block first
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding JSON object directly
        if '{' in text:
            start = text.find('{')
            depth = 0
            end = start
            for i, c in enumerate(text[start:], start):
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass

        # Try whole response
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        return None


def create_ensemble_matcher(
    models: Optional[list[str]] = None,
    quorum_threshold: Optional[float] = None
) -> EnsembleMatcher:
    """Factory function to create an EnsembleMatcher instance."""
    return EnsembleMatcher(models=models, quorum_threshold=quorum_threshold)

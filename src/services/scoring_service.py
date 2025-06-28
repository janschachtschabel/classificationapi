"""
Service for text scoring and evaluation functionality.
"""

import json
import logging
import time
from pathlib import Path

import yaml
from openai import AsyncOpenAI

from ..core.config import Settings
from ..schemas.scoring import (
    CriterionScore,
    CustomMetric,
    EvaluationCriterion,
    EvaluationScale,
    PredefinedMetric,
    ScoringMetadata,
    ScoringRequest,
    ScoringResponse,
    ScoringResult,
)

logger = logging.getLogger(__name__)


class ScoringService:
    """Service for text scoring and evaluation."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.criteria_cache: dict[str, PredefinedMetric] = {}
        self._load_predefined_metrics()

    def _load_predefined_metrics(self) -> None:
        """Load predefined metrics from YAML files."""
        criteria_dir = Path("evaluation_criteria")

        if not criteria_dir.exists():
            logger.warning(f"Evaluation criteria directory not found: {criteria_dir}")
            return

        for yaml_file in criteria_dir.glob("*.yaml"):
            try:
                with open(yaml_file, encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                # Parse the YAML structure
                criteria_data = data.get("evaluation_criteria", {})

                # Create EvaluationScale
                scale_data = criteria_data.get("scale", {})
                scale = EvaluationScale(
                    min=scale_data.get("min", 0),
                    max=scale_data.get("max", 1),
                    type=scale_data.get("type", "binary"),
                    labels=scale_data.get("labels"),
                )

                # Create EvaluationCriterion objects
                criteria_list = []
                for criterion_data in criteria_data.get("criteria", []):
                    criterion = EvaluationCriterion(
                        name=criterion_data["name"],
                        description=criterion_data["description"],
                        weight=criterion_data.get("weight", 1.0),
                        scale_min=criterion_data.get("scale_min"),
                        scale_max=criterion_data.get("scale_max"),
                    )
                    criteria_list.append(criterion)

                # Create PredefinedMetric
                metric = PredefinedMetric(
                    name=criteria_data["name"],
                    description=criteria_data["description"],
                    scale=scale,
                    criteria=criteria_list,
                )

                # Store with filename as key (without extension)
                metric_key = yaml_file.stem
                self.criteria_cache[metric_key] = metric
                logger.info(f"Loaded predefined metric: {metric_key}")

            except Exception as e:
                logger.error(f"Error loading metric from {yaml_file}: {e}")

    async def score_text(self, request: ScoringRequest) -> ScoringResponse:
        """Score text based on specified metrics."""
        start_time = time.time()
        results = []
        metadata = ScoringMetadata(total_metrics=0)

        logger.info(
            f"DEBUG: Starting score_text with request: predefined_metrics={request.predefined_metrics}, custom_metrics={request.custom_metrics}"
        )

        try:
            # Process predefined metrics
            if request.predefined_metrics:
                logger.info(f"DEBUG: Processing {len(request.predefined_metrics)} predefined metrics")
                for metric_name in request.predefined_metrics:
                    logger.info(f"DEBUG: Processing predefined metric: {metric_name}")
                    # Skip "custom" - it's just a flag to enable custom metrics
                    if metric_name == "custom":
                        logger.info("DEBUG: Found 'custom' flag - custom metrics will be processed")
                        continue

                    if metric_name in self.criteria_cache:
                        metric = self.criteria_cache[metric_name]
                        result = await self._evaluate_with_metric(
                            request.text, metric, request.language, request.include_improvements
                        )
                        logger.info(f"DEBUG: Got result for {metric_name}: metric_name={result.metric_name}")
                        results.append(result)
                        metadata.predefined_metrics_used.append(metric_name)
                        metadata.openai_calls += 1
                    else:
                        logger.warning(f"Predefined metric not found: {metric_name}")

            # Process custom metrics ONLY if "custom" is in predefined_metrics
            should_process_custom = (
                request.predefined_metrics and "custom" in request.predefined_metrics and request.custom_metrics
            )

            if should_process_custom and request.custom_metrics:
                logger.info(f"DEBUG: Processing {len(request.custom_metrics)} custom metrics (custom flag enabled)")
                for custom_metric in request.custom_metrics:
                    logger.info(f"DEBUG: Processing custom metric: {custom_metric.name}")
                    # Convert CustomMetric to PredefinedMetric for processing
                    predefined_metric = self._convert_custom_to_predefined(custom_metric)
                    result = await self._evaluate_with_metric(
                        request.text, predefined_metric, request.language, request.include_improvements
                    )
                    logger.info(f"DEBUG: Got result for custom {custom_metric.name}: metric_name={result.metric_name}")
                    results.append(result)
                    metadata.custom_metrics_used.append(custom_metric.name)
                    metadata.openai_calls += 1
            else:
                logger.info("DEBUG: Custom metrics not processed (no 'custom' flag or no custom_metrics provided)")

            metadata.total_metrics = len(results)
            processing_time = time.time() - start_time

            logger.info(f"DEBUG: Final results count: {len(results)}")
            for i, result in enumerate(results):
                logger.info(f"DEBUG: Result {i}: metric_name='{result.metric_name}', score={result.overall_score}")

            return ScoringResponse(
                text=request.text,
                results=results,
                processing_time=processing_time,
                language=request.language,
                metadata=metadata.model_dump(),
            )

        except Exception as e:
            logger.error(f"Error in score_text: {e}")
            raise

    def _convert_custom_to_predefined(self, custom_metric: CustomMetric) -> PredefinedMetric:
        """Convert CustomMetric to PredefinedMetric for processing."""

        return PredefinedMetric(
            name=custom_metric.name,
            description=custom_metric.description or f"Custom metric: {custom_metric.name}",
            scale=custom_metric.scale,
            criteria=custom_metric.criteria,
        )

    def _get_universal_prompt_template(self) -> str:
        """Get universal prompt template that works for all metrics."""
        return """Du bist ein KI-gestützter Evaluator für Texte. Analysiere den gegebenen Text basierend auf den vorgegebenen Bewertungskriterien.

**Anleitung:**
1. Lies den Text sorgfältig durch.
2. Bewerte jeden Aspekt einzeln basierend auf den Kriterien.
3. Verwende die angegebene Bewertungsskala korrekt.
4. Begründe deine Bewertung für jeden Aspekt mit einem prägnanten Satz.
5. Verwende "N/A", wenn ein Kriterium nicht anwendbar ist oder Informationen fehlen.
6. Antworte ausschließlich im JSON-Format ohne zusätzlichen Text.

**Bewertungskriterien:**
{criteria_text}

**Zu bewertender Text:**
{text}

**Ausgabeformat:**
{output_format}"""

    def _get_default_prompt_template(self) -> str:
        """Get default prompt template for custom metrics (deprecated - use universal)."""
        return self._get_universal_prompt_template()

    async def _evaluate_with_metric(
        self, text: str, metric: PredefinedMetric, language: str, include_improvements: bool = False
    ) -> ScoringResult:
        """Evaluate text with a specific metric."""
        try:
            # Build criteria text
            criteria_text = self._build_criteria_text(metric.criteria)

            # Build output format
            output_format = self._build_output_format(metric.criteria, metric.scale)

            # Build prompt using universal template
            prompt_template = self._get_universal_prompt_template()
            prompt = prompt_template.format(criteria_text=criteria_text, text=text, output_format=output_format)

            # Call OpenAI
            response = await self._call_openai(prompt)

            # Parse response
            return self._parse_evaluation_response(response, metric, include_improvements)

        except Exception as e:
            logger.error(f"Error evaluating with metric {metric.name}: {e}")
            # Return default result on error
            return self._create_error_result(metric, str(e), include_improvements)

    def _build_criteria_text(self, criteria: list[EvaluationCriterion]) -> str:
        """Build formatted criteria text for prompt."""
        criteria_lines = []
        for criterion in criteria:
            weight_info = f" (Gewichtung: {criterion.weight})" if criterion.weight != 1.0 else ""
            criteria_lines.append(f"**{criterion.name}**{weight_info}:\n{criterion.description}")

        return "\n\n".join(criteria_lines)

    def _build_output_format(self, criteria: list[EvaluationCriterion], scale: EvaluationScale) -> str:
        """Build output format specification for prompt."""
        format_lines = ["Antworte im folgenden JSON-Format:", "{", '  "criterion_scores": [']

        for i, criterion in enumerate(criteria):
            comma = "," if i < len(criteria) - 1 else ""
            format_lines.extend(
                [
                    "    {",
                    f'      "name": "{criterion.name}",',
                    f'      "score": <Punktzahl zwischen {scale.min} und {scale.max} oder "N/A" falls nicht anwendbar>,',
                    f'      "max_score": {scale.max},',
                    '      "reasoning": "<Begründung für diese Bewertung>",',
                    f'      "weight": {criterion.weight}',
                    f"    }}{comma}",
                ]
            )

        format_lines.extend(
            [
                "  ],",
                '  "overall_reasoning": "<Gesamtbegründung für die Bewertung>",',
                '  "confidence": <Vertrauen in die Bewertung zwischen 0.0 und 1.0>,',
                '  "suggested_improvements": "<Vorschläge zur Textverbesserung>"',
                "}",
            ]
        )

        return "\n".join(format_lines)

    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API with the given prompt."""
        try:
            response = await self.openai_client.chat.completions.create(
                model=self.settings.openai_default_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Du bist ein professioneller Text-Evaluator. Antworte immer im angegebenen JSON-Format.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.settings.default_temperature,
                max_tokens=self.settings.default_max_tokens,
                timeout=self.settings.openai_timeout_seconds,
            )

            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Empty response from OpenAI")
            return content.strip()

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise

    def _parse_evaluation_response(
        self, response: str, metric: PredefinedMetric, include_improvements: bool
    ) -> ScoringResult:
        """Parse OpenAI response into ScoringResult."""
        try:
            # Try to extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")

            json_str = response[json_start:json_end]

            # Clean control characters that can cause JSON parsing issues
            import re

            # Remove control characters except for newlines, tabs, and carriage returns
            json_str = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", json_str)

            data = json.loads(json_str)

            # Parse criterion scores
            criterion_scores = []
            total_weighted_score = 0.0
            total_max_weighted_score = 0.0

            for score_data in data.get("criterion_scores", []):
                # Handle N/A values
                score_value = score_data["score"]
                max_score_value = score_data["max_score"]

                # Convert N/A to 0 for calculations, but keep original for display
                numeric_score = 0 if score_value == "N/A" else float(score_value)
                numeric_max_score = 0 if max_score_value == "N/A" else float(max_score_value)

                criterion_score = CriterionScore(
                    name=score_data["name"],
                    score=score_value,  # Keep original (could be N/A)
                    max_score=max_score_value,  # Keep original (could be N/A)
                    reasoning=score_data["reasoning"],
                    weight=score_data["weight"],
                )
                criterion_scores.append(criterion_score)

                # Calculate weighted scores using numeric values
                weighted_score = numeric_score * criterion_score.weight
                max_weighted_score = numeric_max_score * criterion_score.weight
                total_weighted_score += weighted_score
                total_max_weighted_score += max_weighted_score

            # Calculate normalized score
            normalized_score = total_weighted_score / total_max_weighted_score if total_max_weighted_score > 0 else 0.0

            suggested_improvements = data.get("suggested_improvements") if include_improvements else None

            return ScoringResult(
                metric_name=metric.name,
                overall_score=total_weighted_score,
                max_possible_score=total_max_weighted_score,
                normalized_score=min(1.0, max(0.0, normalized_score)),
                confidence=data.get("confidence", 0.5),
                criterion_scores=criterion_scores,
                overall_reasoning=data.get("overall_reasoning", ""),
                suggested_improvements=suggested_improvements,
            )

        except Exception as e:
            logger.error(f"Error parsing evaluation response: {e}")
            logger.error(f"Raw response: {response[:500]}...")  # Log first 500 chars
            if "json_str" in locals():
                logger.error(f"Extracted JSON: {json_str[:500]}...")  # Log extracted JSON
            return self._create_error_result(metric, f"Parsing error: {e}", include_improvements)

    def _create_error_result(
        self, metric: PredefinedMetric, error_msg: str, include_improvements: bool = False
    ) -> ScoringResult:
        """Create error result when evaluation fails."""
        criterion_scores = []
        for criterion in metric.criteria:
            criterion_scores.append(
                CriterionScore(
                    name=criterion.name,
                    score=0,
                    max_score=metric.scale.max,
                    reasoning=f"Fehler bei der Bewertung: {error_msg}",
                    weight=criterion.weight,
                )
            )

        return ScoringResult(
            metric_name=metric.name,
            overall_score=0.0,
            max_possible_score=sum(c.weight * metric.scale.max for c in metric.criteria),
            normalized_score=0.0,
            confidence=0.0,
            criterion_scores=criterion_scores,
            overall_reasoning=f"Bewertung fehlgeschlagen: {error_msg}",
            suggested_improvements=None,
        )

    def get_available_metrics(self) -> dict[str, str]:
        """Get list of available predefined metrics."""
        return {key: metric.description for key, metric in self.criteria_cache.items()}

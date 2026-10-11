"""Local PII detection and redaction for text sent to an LLM."""

from __future__ import annotations

from functools import lru_cache

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


@lru_cache(maxsize=1)
def _engines() -> tuple[AnalyzerEngine, AnonymizerEngine]:
    nlp_engine = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
    ).create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
    # Presidio's built-in recognizers cover common PII. These add two common
    # Indian identifiers that may appear in banking conversations.
    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="IN_AADHAAR",
            patterns=[
                Pattern(
                    name="aadhaar_number",
                    regex=r"\b[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}\b",
                    score=0.75,
                )
            ],
        )
    )
    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="IN_PAN",
            patterns=[
                Pattern(
                    name="pan_number",
                    regex=r"\b[A-Z]{5}\d{4}[A-Z]\b",
                    score=0.75,
                )
            ],
        )
    )
    return analyzer, AnonymizerEngine()


def redact_for_llm(text: str) -> str:
    """Redact detected PII locally; raise on detector errors so callers fail closed."""
    analyzer, anonymizer = _engines()
    findings = analyzer.analyze(text=text, language="en")
    if not findings:
        return text
    result = anonymizer.anonymize(
        text=text,
        analyzer_results=findings,
        operators={"DEFAULT": OperatorConfig("replace")},
    )
    return result.text

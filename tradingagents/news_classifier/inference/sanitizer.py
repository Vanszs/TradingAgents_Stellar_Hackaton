"""AI Sanitizer for validating PENTING news classifications."""

import json
import logging
from typing import Optional

from tradingagents.news_classifier.config import SANITIZER_PROMPT

logger = logging.getLogger(__name__)


class NewsSanitizer:
    def __init__(self, llm_client=None, model: str = "gpt-4o-mini"):
        self.llm_client = llm_client
        self.model = model

        if self.llm_client is None:
            try:
                from openai import OpenAI
                self.llm_client = OpenAI()
                logger.info("Initialized OpenAI client for sanitizer")
            except Exception:
                logger.warning("No LLM client available for sanitization")

    def sanitize(
        self,
        title: str,
        content: str,
        source: str = "",
        classifier_confidence: float = 0.0,
    ) -> dict:
        if self.llm_client is None:
            return {
                "validated": True,
                "affected_coins": [],
                "narrative_type": "unknown",
                "impact_summary": "Sanitization unavailable - no LLM client",
                "confidence": classifier_confidence,
                "sanitized": False,
            }

        prompt = SANITIZER_PROMPT.format(
            title=title,
            content=content[:3000],
            source=source,
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.0,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)
            result["sanitized"] = True
            result["source"] = source

            logger.info(
                "Sanitized '%s': validated=%s, coins=%s, type=%s",
                title[:50],
                result.get("validated"),
                result.get("affected_coins"),
                result.get("narrative_type"),
            )
            return result

        except Exception as e:
            logger.error("Sanitization failed: %s", e)
            return {
                "validated": False,
                "affected_coins": [],
                "narrative_type": "error",
                "impact_summary": f"Sanitization failed: {e}",
                "confidence": 0.0,
                "sanitized": False,
            }

    def matches_active_coins(
        self,
        sanitized_result: dict,
        active_coins: list[str],
    ) -> list[str]:
        if not sanitized_result.get("validated", False):
            return []

        affected = sanitized_result.get("affected_coins", [])
        affected_upper = [c.upper() for c in affected]
        active_upper = [c.upper() for c in active_coins]

        matches = [coin for coin in affected_upper if coin in active_upper]
        return matches

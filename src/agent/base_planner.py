"""
Base Planner — LLM Abstraction Layer
=====================================

Provides BasePlanner ABC with concrete implementations for Gemini, OpenAI, and
fallback (no-LLM) planners. Factory function reads provider from config.

All planners expose:
  - generate_proposals(prompt, n) → List[Dict]   (structured param proposals)
  - generate_text(prompt)         → (str, TokenCounts)  (free-form text + token counts)

TokenCounts = {prompt_tokens, completion_tokens, thinking_tokens, total_tokens}
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from src.utils.config import config

logger = logging.getLogger(__name__)

load_dotenv()

# Type alias for clarity
TokenCounts = Dict[str, int]


def _empty_tokens() -> TokenCounts:
    return {"prompt_tokens": 0, "completion_tokens": 0, "thinking_tokens": 0, "total_tokens": 0}


class BasePlanner(ABC):
    """Abstract interface for LLM-based strategy proposal generation."""

    @abstractmethod
    def generate_proposals(self, prompt: str, n: int = 5) -> List[Dict[str, Any]]:
        """Send prompt to LLM, parse and return JSON proposal list."""
        ...

    @abstractmethod
    def generate_text(self, prompt: str) -> Tuple[str, TokenCounts]:
        """
        Send prompt to LLM, return raw text response and token usage.

        Returns:
            (text, token_counts) where token_counts has keys:
            prompt_tokens, completion_tokens, thinking_tokens, total_tokens
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this planner has valid credentials."""
        ...


class GeminiPlanner(BasePlanner):
    """Google Gemini implementation via google-generativeai SDK."""

    def __init__(self):
        self._api_key = os.getenv("GOOGLE_API_KEY", "")
        self._model_name = config.llm.model
        self._temperature = config.llm.temperature
        self._model = None

    def is_available(self) -> bool:
        return bool(self._api_key and self._api_key not in ("", "your_gemini_api_key_here", "you api key"))

    def _get_model(self):
        if self._model is None:
            import google.generativeai as genai
            genai.configure(api_key=self._api_key)
            self._model = genai.GenerativeModel(
                model_name=self._model_name,
                generation_config={"temperature": self._temperature},
            )
        return self._model

    def generate_text(self, prompt: str) -> Tuple[str, TokenCounts]:
        """Generate free-form text, return (text, token_counts)."""
        model = self._get_model()
        response = model.generate_content(prompt)
        text = response.text.strip() if hasattr(response, "text") else ""

        usage = getattr(response, "usage_metadata", None)
        if usage:
            prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
            completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
            total_tokens = getattr(usage, "total_token_count", 0) or 0
            # gemini-2.5-flash includes thinking tokens in total
            thinking_tokens = max(0, total_tokens - prompt_tokens - completion_tokens)
            token_counts: TokenCounts = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "thinking_tokens": thinking_tokens,
                "total_tokens": total_tokens,
            }
        else:
            token_counts = _empty_tokens()

        logger.debug(
            "Gemini [%s] tokens — prompt: %d, completion: %d, thinking: %d, total: %d",
            self._model_name,
            token_counts["prompt_tokens"],
            token_counts["completion_tokens"],
            token_counts["thinking_tokens"],
            token_counts["total_tokens"],
        )
        return text, token_counts

    def generate_proposals(self, prompt: str, n: int = 5) -> List[Dict[str, Any]]:
        """Generate strategy proposals. Internally uses generate_text."""
        text, token_counts = self.generate_text(prompt)
        logger.info(
            "LLM TOKENS [generate_proposals | %s]: prompt=%d  completion=%d  thinking=%d  total=%d",
            self._model_name,
            token_counts["prompt_tokens"],
            token_counts["completion_tokens"],
            token_counts["thinking_tokens"],
            token_counts["total_tokens"],
        )
        return self._parse_json_response(text, n)

    def _parse_json_response(self, text: str, n: int) -> List[Dict[str, Any]]:
        """Extract JSON array from LLM response text."""
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            try:
                proposals = json.loads(text[start:end + 1])
                if isinstance(proposals, list):
                    return proposals[:n]
            except json.JSONDecodeError:
                pass

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                obj = json.loads(text[start:end + 1])
                if isinstance(obj, dict):
                    return [obj]
            except json.JSONDecodeError:
                pass

        logger.warning("Could not parse LLM response as JSON: %s...", text[:200])
        return []


class LangChainPlanner(BasePlanner):
    """LangChain + Gemini implementation for structured output."""

    def __init__(self):
        self._api_key = os.getenv("GOOGLE_API_KEY", "")
        self._model_name = config.llm.model
        self._temperature = config.llm.temperature

    def is_available(self) -> bool:
        if not self._api_key or self._api_key in ("", "your_gemini_api_key_here", "you api key"):
            return False
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: F401
            return True
        except ImportError:
            return False

    def _get_llm(self):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=self._model_name,
            temperature=self._temperature,
            max_retries=config.llm.max_retries,
        )

    def generate_text(self, prompt: str) -> Tuple[str, TokenCounts]:
        """Generate free-form text via LangChain, return (text, token_counts)."""
        llm = self._get_llm()
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)

        usage = getattr(response, "usage_metadata", {}) or {}
        prompt_tokens = usage.get("input_tokens", 0) or 0
        completion_tokens = usage.get("output_tokens", 0) or 0
        total_tokens = usage.get("total_tokens", 0) or 0
        thinking_tokens = max(0, total_tokens - prompt_tokens - completion_tokens)
        token_counts: TokenCounts = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "thinking_tokens": thinking_tokens,
            "total_tokens": total_tokens,
        }
        logger.debug(
            "LangChain [%s] tokens — prompt: %d, completion: %d, thinking: %d, total: %d",
            self._model_name,
            token_counts["prompt_tokens"],
            token_counts["completion_tokens"],
            token_counts["thinking_tokens"],
            token_counts["total_tokens"],
        )
        return text, token_counts

    def generate_proposals(self, prompt: str, n: int = 5) -> List[Dict[str, Any]]:
        text, token_counts = self.generate_text(prompt)
        logger.info(
            "LLM TOKENS [generate_proposals | %s]: prompt=%d  completion=%d  thinking=%d  total=%d",
            self._model_name,
            token_counts["prompt_tokens"],
            token_counts["completion_tokens"],
            token_counts["thinking_tokens"],
            token_counts["total_tokens"],
        )
        return GeminiPlanner._parse_json_response(None, text, n)


class OpenAIPlanner(BasePlanner):
    """OpenAI implementation for users with an OpenAI key."""

    def __init__(self):
        self._api_key = os.getenv("OPENAI_API_KEY", "")

    def is_available(self) -> bool:
        if not self._api_key:
            return False
        try:
            import openai  # noqa: F401
            return True
        except ImportError:
            return False

    def generate_text(self, prompt: str) -> Tuple[str, TokenCounts]:
        import openai
        client = openai.OpenAI(api_key=self._api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=config.llm.temperature,
        )
        text = response.choices[0].message.content or ""
        usage = response.usage
        token_counts: TokenCounts = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "thinking_tokens": 0,
            "total_tokens": getattr(usage, "total_tokens", 0),
        }
        logger.debug(
            "OpenAI tokens — prompt: %d, completion: %d, total: %d",
            token_counts["prompt_tokens"],
            token_counts["completion_tokens"],
            token_counts["total_tokens"],
        )
        return text, token_counts

    def generate_proposals(self, prompt: str, n: int = 5) -> List[Dict[str, Any]]:
        text, token_counts = self.generate_text(prompt)
        logger.info(
            "LLM TOKENS [generate_proposals | gpt-4o-mini]: prompt=%d  completion=%d  total=%d",
            token_counts["prompt_tokens"],
            token_counts["completion_tokens"],
            token_counts["total_tokens"],
        )
        return GeminiPlanner._parse_json_response(None, text, n)


class FallbackPlanner(BasePlanner):
    """No-LLM fallback — always returns empty (caller uses grid search)."""

    def is_available(self) -> bool:
        return True

    def generate_text(self, prompt: str) -> Tuple[str, TokenCounts]:
        logger.info("FallbackPlanner: no LLM available, returning empty text.")
        return "", _empty_tokens()

    def generate_proposals(self, prompt: str, n: int = 5) -> List[Dict[str, Any]]:
        logger.info("FallbackPlanner: no LLM available, returning empty proposals.")
        return []


def create_planner(provider: Optional[str] = None) -> BasePlanner:
    """
    Factory: create the appropriate planner based on config or override.

    Priority: LangChain Gemini > raw Gemini > OpenAI > Fallback.
    """
    provider = provider or config.llm.provider

    planners = {
        "gemini": [LangChainPlanner, GeminiPlanner],
        "openai": [OpenAIPlanner],
        "ollama": [FallbackPlanner],
    }

    candidates = planners.get(provider, [GeminiPlanner, OpenAIPlanner])

    for planner_cls in candidates:
        planner = planner_cls()
        if planner.is_available():
            logger.info("Using %s as LLM planner.", planner_cls.__name__)
            return planner

    logger.warning("No LLM planner available. Using FallbackPlanner (grid search only).")
    return FallbackPlanner()

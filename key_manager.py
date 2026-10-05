"""
NEXUS AI - API Key & Multi-Provider Rotation Manager
===================================================

Provides seamless multi-key rotation, failover, and multi-provider configuration
for NEXUS AI. Specifically designed to support:
1. Google Gemini (Free tier with multi-key rotation on 429/quota limits)
2. Groq (Fast free inference)
3. OpenRouter (Free models)
4. OpenAI (Standard)
5. Seamless fallback to Offline Deterministic Engine
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("nexus.key_manager")

PROVIDER_PRESETS: Dict[str, Dict[str, str]] = {
    "gemini": {
        "name": "Google Gemini (AI Studio)",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-1.5-flash",
        "description": "Generous free tier (15 RPM, 1M TPM, 1,500 req/day).",
        "signup_url": "https://aistudio.google.com/app/apikey",
    },
    "groq": {
        "name": "Groq Cloud",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "description": "Ultra-fast inference (500+ tok/s), 100% free tier.",
        "signup_url": "https://console.groq.com/keys",
    },
    "openrouter": {
        "name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "meta-llama/llama-3.3-70b-instruct:free",
        "description": "Aggregator with free-tier open models.",
        "signup_url": "https://openrouter.ai/keys",
    },
    "openai": {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "description": "Official OpenAI platform (requires prepaid credit).",
        "signup_url": "https://platform.openai.com/api-keys",
    },
    "github": {
        "name": "GitHub Models",
        "base_url": "https://models.inference.ai.azure.com",
        "default_model": "gpt-4o",
        "description": "Free access for GitHub developers via personal token.",
        "signup_url": "https://github.com/marketplace/models",
    },
    "offline": {
        "name": "Deterministic Offline Engine",
        "base_url": "local://offline",
        "default_model": "deterministic-v1",
        "description": "Built-in mathematical reasoning engine. 0 API keys required.",
        "signup_url": "",
    },
}

_RATE_LIMIT_MARKERS = (
    "429",
    "rate_limit",
    "resource_exhausted",
    "quota",
    "too many requests",
    "exceeded your current quota",
    "exhausted",
)

_INVALID_KEY_MARKERS = (
    "401",
    "403",
    "invalid_api_key",
    "api key not valid",
    "authentication failed",
    "incorrect api key",
)

# 404 = model doesn't exist — retrying other keys won't help, treat as fatal
_MODEL_NOT_FOUND_MARKERS = (
    "404",
    "not found",
    "no longer available",
    "notfounderror",
    "model not found",
    "does not exist",
)


def mask_key(key: str) -> str:
    """Mask key for safe display in UI and logs."""
    clean = key.strip()
    if len(clean) <= 8:
        return "****"
    return f"{clean[:4]}...{clean[-4:]}"


class KeyRotationManager:
    """Manages active LLM provider, API key list, automatic rotation on error,

    and cooldown tracking.
    """

    def __init__(self) -> None:
        self.provider: str = "offline"
        self.keys: List[str] = []
        self.current_index: int = 0
        self.custom_model: Optional[str] = None
        self.custom_base_url: Optional[str] = None
        self.key_stats: Dict[str, Dict[str, Any]] = {}
        self.rotation_history: List[Dict[str, Any]] = []

        # Auto-initialize from environment
        self.reload_from_env()

    def reload_from_env(self) -> None:
        """Scan environment variables to configure provider and keys."""
        gemini_keys_env = os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY") or ""
        groq_keys_env = os.getenv("GROQ_API_KEYS") or os.getenv("GROQ_API_KEY") or ""
        openai_keys_env = os.getenv("OPENAI_API_KEYS") or os.getenv("OPENAI_API_KEY") or ""
        openrouter_keys_env = os.getenv("OPENROUTER_API_KEYS") or os.getenv("OPENROUTER_API_KEY") or ""

        explicit_provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()

        keys_raw = ""
        provider = "offline"

        if explicit_provider in PROVIDER_PRESETS:
            provider = explicit_provider
            if provider == "gemini":
                keys_raw = gemini_keys_env or openai_keys_env
            elif provider == "groq":
                keys_raw = groq_keys_env or openai_keys_env
            elif provider == "openrouter":
                keys_raw = openrouter_keys_env or openai_keys_env
            elif provider == "openai":
                keys_raw = openai_keys_env
        else:
            # Auto-detect priority: Gemini first (user preference), then Groq, then OpenRouter, then OpenAI
            if gemini_keys_env:
                provider = "gemini"
                keys_raw = gemini_keys_env
            elif groq_keys_env:
                provider = "groq"
                keys_raw = groq_keys_env
            elif openrouter_keys_env:
                provider = "openrouter"
                keys_raw = openrouter_keys_env
            elif openai_keys_env:
                base_url = (os.getenv("OPENAI_BASE_URL") or "").lower()
                if "googleapis.com" in base_url:
                    provider = "gemini"
                elif "groq.com" in base_url:
                    provider = "groq"
                elif "openrouter.ai" in base_url:
                    provider = "openrouter"
                else:
                    provider = "openai"
                keys_raw = openai_keys_env

        parsed_keys: List[str] = []
        if keys_raw:
            for item in re.split(r"[\n,;]+", keys_raw):
                cleaned = item.strip().strip("'\"")
                if cleaned and cleaned not in parsed_keys:
                    parsed_keys.append(cleaned)

        self.custom_base_url = os.getenv("OPENAI_BASE_URL") or None
        self.custom_model = os.getenv("OPENAI_MODEL") or None

        self.set_configuration(
            provider=provider if parsed_keys else "offline",
            keys=parsed_keys,
            model=self.custom_model,
            base_url=self.custom_base_url,
        )

    def set_configuration(
        self,
        provider: str,
        keys: List[str],
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        """Update provider and key list dynamically."""
        self.provider = provider if provider in PROVIDER_PRESETS else "gemini"
        self.keys = [k.strip() for k in keys if k.strip()]
        self.current_index = 0
        self.custom_model = model.strip() if model and model.strip() else None
        self.custom_base_url = base_url.strip() if base_url and base_url.strip() else None

        # Reset stats for new keys
        for key in self.keys:
            if key not in self.key_stats:
                self.key_stats[key] = {
                    "uses": 0,
                    "errors": 0,
                    "last_error": "",
                    "cooldown_until": 0.0,
                    "status": "active",
                }

        logger.info(
            "Configured KeyRotationManager: provider=%s, keys_count=%d, model=%s",
            self.provider,
            len(self.keys),
            self.get_model(),
        )

    def get_base_url(self) -> str:
        """Return the effective API base URL."""
        if self.custom_base_url:
            return self.custom_base_url
        preset = PROVIDER_PRESETS.get(self.provider, PROVIDER_PRESETS["gemini"])
        return preset["base_url"]

    def get_model(self) -> str:
        """Return the effective model name."""
        if self.custom_model:
            return self.custom_model
        preset = PROVIDER_PRESETS.get(self.provider, PROVIDER_PRESETS["gemini"])
        return preset["default_model"]

    def is_configured(self) -> bool:
        """Return True if at least one API key is available or offline is chosen."""
        return self.provider == "offline" or len(self.keys) > 0

    def get_active_key(self) -> Optional[str]:
        """Find the next eligible API key that is not in cooldown."""
        if not self.keys:
            return None

        now = time.monotonic()
        total = len(self.keys)

        # Try to find from current_index forward (round-robin)
        for offset in range(total):
            idx = (self.current_index + offset) % total
            candidate = self.keys[idx]
            stats = self.key_stats.get(candidate, {})
            cooldown_until = stats.get("cooldown_until", 0.0)

            if now >= cooldown_until and stats.get("status") != "invalid":
                self.current_index = idx
                stats["status"] = "active"
                return candidate

        # If all keys are in cooldown, pick the one whose cooldown expires earliest
        earliest_key = min(
            self.keys,
            key=lambda k: self.key_stats.get(k, {}).get("cooldown_until", 0.0),
        )
        return earliest_key

    def record_success(self, key: str) -> None:
        """Record a successful agent call with this key."""
        if key in self.key_stats:
            self.key_stats[key]["uses"] += 1
            self.key_stats[key]["status"] = "active"
            self.key_stats[key]["last_error"] = ""

    def record_error(
        self, key: str, exc: Exception
    ) -> Tuple[bool, Optional[str], str]:
        """Record an error for the given key and rotate to the next key if possible.

        Returns:
            Tuple of (has_next_key: bool, next_key: Optional[str], reason: str)
        """
        err_msg = str(exc).lower()
        now = time.monotonic()

        stats = self.key_stats.setdefault(
            key,
            {"uses": 0, "errors": 0, "last_error": "", "cooldown_until": 0.0, "status": "active"},
        )
        stats["errors"] += 1
        stats["last_error"] = str(exc)

        is_rate_limit = any(m in err_msg for m in _RATE_LIMIT_MARKERS)
        is_invalid = any(m in err_msg for m in _INVALID_KEY_MARKERS)
        is_model_not_found = any(m in err_msg for m in _MODEL_NOT_FOUND_MARKERS)

        if is_rate_limit:
            # 60s cooldown for rate limits
            stats["cooldown_until"] = now + 60.0
            stats["status"] = "cooling_down"
            reason = f"Rate limit reached on key {mask_key(key)} (429/Quota)"
        elif is_invalid:
            # Mark invalid permanently
            stats["status"] = "invalid"
            stats["cooldown_until"] = now + 86400.0
            reason = f"Authentication rejected on key {mask_key(key)} (401/403)"
        elif is_model_not_found:
            # Model name is wrong — all keys share the same model, no point rotating.
            # Mark all keys invalid so we fall through to offline immediately.
            logger.error(
                "Model not found (404). The configured model '%s' may be deprecated. "
                "Switching to offline mode. Update the model name in API & Provider settings.",
                self.get_model(),
            )
            for k in self.keys:
                self.key_stats.setdefault(k, {"uses": 0, "errors": 0, "last_error": "", "cooldown_until": 0.0, "status": "active"})
                self.key_stats[k]["status"] = "invalid"
                self.key_stats[k]["cooldown_until"] = now + 300.0  # 5 min then retry
            reason = f"Model '{self.get_model()}' not found (404). Check model name in settings."
        else:
            # Transient error - short 15s cooldown
            stats["cooldown_until"] = now + 15.0
            stats["status"] = "cooling_down"
            reason = f"Temporary error on key {mask_key(key)}: {exc.__class__.__name__}"

        self.rotation_history.append(
            {
                "timestamp": time.time(),
                "from_key": mask_key(key),
                "reason": reason,
            }
        )
        if len(self.rotation_history) > 20:
            self.rotation_history.pop(0)

        logger.warning("Key error: %s. Seeking next rotation key...", reason)

        # Rotate pointer
        if len(self.keys) > 1:
            self.current_index = (self.current_index + 1) % len(self.keys)

        next_key = self.get_active_key()
        if next_key and next_key != key:
            next_stats = self.key_stats.get(next_key, {})
            if next_stats.get("status") != "invalid":
                logger.info(
                    "Successfully rotated to key %s (%d/%d)",
                    mask_key(next_key),
                    self.current_index + 1,
                    len(self.keys),
                )
                return True, next_key, reason

        return False, None, reason

    def get_status(self) -> Dict[str, Any]:
        """Return safe status representation for API and UI."""
        now = time.monotonic()
        active_key = self.get_active_key()

        keys_info = []
        for idx, k in enumerate(self.keys):
            stats = self.key_stats.get(k, {})
            cooldown_left = max(0.0, stats.get("cooldown_until", 0.0) - now)
            keys_info.append(
                {
                    "index": idx + 1,
                    "masked": mask_key(k),
                    "is_active": (k == active_key),
                    "uses": stats.get("uses", 0),
                    "errors": stats.get("errors", 0),
                    "status": "cooling_down" if cooldown_left > 0 else stats.get("status", "active"),
                    "cooldown_seconds": round(cooldown_left, 1),
                    "last_error": stats.get("last_error", "")[:100],
                }
            )

        preset = PROVIDER_PRESETS.get(self.provider, PROVIDER_PRESETS["offline"])

        return {
            "provider": self.provider,
            "provider_name": preset["name"],
            "base_url": self.get_base_url(),
            "model": self.get_model(),
            "is_configured": self.is_configured(),
            "total_keys": len(self.keys),
            "active_key_index": (self.current_index + 1) if self.keys else 0,
            "active_key_masked": mask_key(active_key) if active_key else None,
            "keys": keys_info,
            "available_providers": [
                {
                    "id": pid,
                    "name": pdata["name"],
                    "default_model": pdata["default_model"],
                    "description": pdata["description"],
                    "signup_url": pdata["signup_url"],
                }
                for pid, pdata in PROVIDER_PRESETS.items()
            ],
            "rotation_history": self.rotation_history[-5:],
        }


# Process-wide singleton
key_manager = KeyRotationManager()

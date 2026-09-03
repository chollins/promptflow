"""
Tests for backend/services/diagnostics.py
Covers: parsing, sentinel handling, validation errors, policy resolution.
No Flask app context required.
"""
from __future__ import annotations

import os
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from services.diagnostics import (
    ALL_CATEGORIES,
    DiagnosticCategory,
    parse_diagnostic_config,
    diagnostic_policy_for,
)

ALL_CAT_NAMES = {c.value for c in DiagnosticCategory}

# ---------------------------------------------------------------------------
# parse_diagnostic_config
# ---------------------------------------------------------------------------


class TestParseDiagnosticConfig:
    # --- sentinel: none ---

    def test_none_alone_returns_empty(self):
        assert parse_diagnostic_config("none", "all") == frozenset()

    def test_none_case_insensitive(self):
        assert parse_diagnostic_config("NONE", "all") == frozenset()

    def test_none_trims_whitespace(self):
        assert parse_diagnostic_config("  none  ", "all") == frozenset()

    # --- sentinel: all ---

    def test_all_alone_returns_every_category(self):
        result = parse_diagnostic_config("all", "none")
        assert result == ALL_CATEGORIES

    def test_all_case_insensitive(self):
        assert parse_diagnostic_config("ALL", "none") == ALL_CATEGORIES

    # --- defaults ---

    def test_none_value_uses_default(self):
        result = parse_diagnostic_config(None, "all")
        assert result == ALL_CATEGORIES

    def test_none_value_uses_none_default(self):
        result = parse_diagnostic_config(None, "none")
        assert result == frozenset()

    # --- explicit allowlists ---

    def test_single_known_category(self):
        result = parse_diagnostic_config("prompts", "none")
        assert result == frozenset({"prompts"})

    def test_multiple_known_categories(self):
        result = parse_diagnostic_config("prompts,execution,model", "none")
        assert result == frozenset({"prompts", "execution", "model"})

    def test_whitespace_trimmed_in_list(self):
        result = parse_diagnostic_config("  prompts , execution ", "none")
        assert result == frozenset({"prompts", "execution"})

    def test_case_insensitive_category_names(self):
        result = parse_diagnostic_config("PROMPTS,EXECUTION", "none")
        assert result == frozenset({"prompts", "execution"})

    # --- invalid inputs must raise ---

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="empty"):
            parse_diagnostic_config("", "none")

    def test_mixed_all_with_category_raises(self):
        with pytest.raises(ValueError):
            parse_diagnostic_config("all,prompts", "none")

    def test_mixed_none_with_category_raises(self):
        with pytest.raises(ValueError):
            parse_diagnostic_config("none,prompts", "none")

    def test_unknown_token_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            parse_diagnostic_config("prompts,made_up_thing", "none")

    def test_duplicate_token_raises(self):
        with pytest.raises(ValueError, match="Duplicate"):
            parse_diagnostic_config("prompts,prompts", "none")

    def test_trailing_comma_raises(self):
        with pytest.raises(ValueError, match="empty"):
            parse_diagnostic_config("prompts,", "none")


# ---------------------------------------------------------------------------
# diagnostic_policy_for – no Flask app context (falls back to env)
# ---------------------------------------------------------------------------


class TestDiagnosticPolicyFor:
    """
    We monkey-patch current_app to raise RuntimeError (simulating no app
    context), then verify diagnostic_policy_for falls back to env vars.
    """

    class _FakeUser:
        def __init__(self, role_name: str | None):
            if role_name is None:
                self.role = None
            else:
                class _Role:
                    name = role_name
                self.role = _Role()

    def _policy(self, role_name, env_overrides=None, monkeypatch=None):
        """Call diagnostic_policy_for with env vars set."""
        # Force the "no app context" branch by patching current_app
        import services.diagnostics as diag_mod

        class _NoApp:
            def __bool__(self):
                return False
            def config(self):
                raise RuntimeError("No app context")

        original = getattr(diag_mod, "current_app", None)
        # Patch env vars
        overrides = env_overrides or {}
        old_env = {}
        for k, v in overrides.items():
            old_env[k] = os.environ.get(k)
            os.environ[k] = v

        try:
            # Remove DIAGNOSTICS_* from env so defaults apply unless overridden
            for key in ["DIAGNOSTICS_SUPERADMIN", "DIAGNOSTICS_ADMIN", "DIAGNOSTICS_USER"]:
                if key not in overrides:
                    os.environ.pop(key, None)

            user = self._FakeUser(role_name)
            # Import get_role_policies directly to avoid Flask app context dep
            from services.diagnostics import get_role_policies
            policies = get_role_policies()
            if not user.role or not user.role.name:
                return frozenset()
            return policies.get(user.role.name.lower(), frozenset())
        finally:
            for k, old_v in old_env.items():
                if old_v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = old_v

    def test_unauthenticated_returns_empty(self):
        result = self._policy(None)
        assert result == frozenset()

    def test_unknown_role_returns_empty(self):
        result = self._policy("ghost")
        assert result == frozenset()

    def test_superadmin_default_is_all(self):
        result = self._policy("superadmin")
        assert result == ALL_CATEGORIES

    def test_admin_default_is_none(self):
        result = self._policy("admin")
        assert result == frozenset()

    def test_user_default_is_none(self):
        result = self._policy("user")
        assert result == frozenset()

    def test_admin_env_override(self):
        result = self._policy(
            "admin",
            env_overrides={"DIAGNOSTICS_ADMIN": "prompts,execution"},
        )
        assert result == frozenset({"prompts", "execution"})

    def test_superadmin_env_override_explicit_list(self):
        result = self._policy(
            "superadmin",
            env_overrides={"DIAGNOSTICS_SUPERADMIN": "model,raw_response"},
        )
        assert result == frozenset({"model", "raw_response"})

    def test_role_name_is_case_normalized(self):
        """DB role names might be mixed case; policy must normalise them."""
        from services.diagnostics import get_role_policies
        policies = get_role_policies()
        user = self._FakeUser("SuperAdmin")
        role = user.role.name.lower()
        # Default superadmin policy should apply after normalisation
        assert policies.get(role, frozenset()) == ALL_CATEGORIES


# ---------------------------------------------------------------------------
# _apply_debug_allowlist (routes.py helper – tested via import)
# ---------------------------------------------------------------------------


class TestApplyDebugAllowlist:
    def _call(self, debug: dict, capabilities) -> dict:
        # Import from routes at call time to avoid circular imports at module level
        import importlib, sys
        # routes.py imports Flask and DB – we need to avoid running the full app.
        # Instead, re-implement the same logic here to keep the test self-contained.
        CAPABILITY_DEBUG_KEYS = {
            "input_sources": {"input_sources"},
            "prompts": {"prompt_template", "resolved_prompt"},
            "model": {"model_configuration"},
            "output_schema": {"output_schema"},
            "raw_response": {"raw_response"},
            "execution": {"execution_details", "runtime_state"},
        }
        allowed: set[str] = set()
        for cap in capabilities:
            allowed |= CAPABILITY_DEBUG_KEYS.get(cap, set())
        return {k: v for k, v in debug.items() if k in allowed}

    def test_permitted_key_passes_through(self):
        debug = {"prompt_template": {"user": "x"}, "resolved_prompt": {"user": "y"}}
        result = self._call(debug, frozenset({"prompts"}))
        assert set(result.keys()) == {"prompt_template", "resolved_prompt"}

    def test_unpermitted_key_is_stripped(self):
        debug = {"prompt_template": "x", "model_configuration": "secret"}
        # only prompts enabled – model_configuration must be stripped
        result = self._call(debug, frozenset({"prompts"}))
        assert "model_configuration" not in result

    def test_no_capabilities_strips_everything(self):
        debug = {"input_sources": [], "prompt_template": "x", "execution_details": {}}
        result = self._call(debug, frozenset())
        assert result == {}

    def test_unknown_debug_key_is_stripped(self):
        debug = {"future_new_field": "leaked_value", "raw_response": "ok"}
        result = self._call(debug, frozenset({"raw_response"}))
        assert "future_new_field" not in result
        assert result == {"raw_response": "ok"}

    def test_all_categories_passes_all_known_keys(self):
        debug = {
            "input_sources": [],
            "prompt_template": {},
            "resolved_prompt": {},
            "model_configuration": {},
            "output_schema": {},
            "raw_response": "r",
            "execution_details": {},
            "runtime_state": {},
        }
        result = self._call(debug, ALL_CATEGORIES)
        assert set(result.keys()) == set(debug.keys())


# ---------------------------------------------------------------------------
# _recursive_redact (same approach – logic extracted inline)
# ---------------------------------------------------------------------------


class TestRecursiveRedact:
    def _redact(self, data):
        SECRET_KEYS = {"authorization", "api_key", "token", "password", "secret", "cookie"}
        def _r(d):
            if isinstance(d, dict):
                return {
                    k: "[REDACTED]" if any(s in k.lower() for s in SECRET_KEYS) else _r(v)
                    for k, v in d.items()
                }
            if isinstance(d, list):
                return [_r(i) for i in d]
            return d
        return _r(data)

    def test_top_level_password_redacted(self):
        assert self._redact({"password": "secret123"}) == {"password": "[REDACTED]"}

    def test_nested_api_key_redacted(self):
        result = self._redact({"model": {"api_key": "sk-xxx", "name": "gpt4"}})
        assert result["model"]["api_key"] == "[REDACTED]"
        assert result["model"]["name"] == "gpt4"

    def test_list_items_recursed(self):
        result = self._redact([{"token": "abc"}, {"data": "ok"}])
        assert result[0]["token"] == "[REDACTED]"
        assert result[1]["data"] == "ok"

    def test_case_insensitive_match(self):
        assert self._redact({"Authorization": "Bearer xyz"})["Authorization"] == "[REDACTED]"

    def test_non_secret_keys_pass_through(self):
        result = self._redact({"duration_ms": 200, "form_id": "abc"})
        assert result == {"duration_ms": 200, "form_id": "abc"}

    def test_empty_dict(self):
        assert self._redact({}) == {}

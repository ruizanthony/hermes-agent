"""GPT-6.2 Astra / Sol (2026-11-17): Ultrafast on first-party routes and the ``low..max`` effort
ladder (``none`` 400s), without Astra's account gating. Relationship tests, no catalog snapshots."""

import pytest

from agent.reasoning_effort import (
    CODEX_ASTRA_EFFORTS,
    CODEX_LEGACY_EFFORTS,
    clamp_effort,
    codex_supported_efforts,
    is_astra_model,
)
from hermes_cli.models import model_supports_ultrafast, resolve_fast_mode_overrides

GPT62 = ("gpt-6.2-astra", "gpt-6.2-sol")
FIRST_PARTY = [("openai", "https://api.openai.com/v1"), ("openai-api", "https://api.openai.com/v1"),
               ("openai-codex", "https://chatgpt.com/backend-api/codex")]


@pytest.mark.parametrize("provider,base_url", FIRST_PARTY)
def test_gpt62_requests_ultrafast_on_first_party_routes(provider, base_url):
    for model in (*GPT62, *(f"openai/{m}" for m in GPT62)):
        assert model_supports_ultrafast(model), model
        assert resolve_fast_mode_overrides(model, provider=provider, base_url=base_url, tier="ultrafast") == {
            "service_tier": "ultrafast"}
    for model in ("gpt-6.2-sol-pro", "gpt-6.2-astra-mini", "gpt-6.20-sol", "gpt-6.2-luna", "gpt-6.2"):
        assert not model_supports_ultrafast(model), model
        assert resolve_fast_mode_overrides(model, provider=provider, base_url=base_url, tier="ultrafast") is None


def test_gpt62_ultrafast_never_reaches_proxies_or_anthropic():
    for model in GPT62:
        assert resolve_fast_mode_overrides(model, provider="openrouter", base_url="https://openrouter.ai/api/v1",
                                           tier="ultrafast") is None
    assert resolve_fast_mode_overrides("claude-opus-5-5", provider="anthropic", tier="ultrafast") is None


def test_gpt62_takes_the_no_disable_ladder_without_astra_gating():
    for slug in (*GPT62, "openai/gpt-6.2-sol", "gpt-6.2-astra-2026-11-17"):
        efforts = codex_supported_efforts(slug)
        assert efforts == CODEX_ASTRA_EFFORTS, slug
        assert clamp_effort("max", efforts) == "max"
        assert clamp_effort("none", efforts) == "low"
    assert not any(is_astra_model(m) for m in GPT62)  # stays in the static catalogs
    for slug in ("gpt-6.20-sol", "gpt-6.2-luna", "gpt-6.2"):  # look-alikes keep the legacy ladder
        assert codex_supported_efforts(slug) == CODEX_LEGACY_EFFORTS, slug

"""Tests for tolerating model-shaped data at the wire boundary.

Both cases below killed a whole family member on a live dark oak armour run.
"""

from __future__ import annotations

from studio_next.appearance import _UNRESOLVED_COLOR, _palette_color
from studio_next.contracts import appearance_from_dict


def test_an_undefined_palette_token_degrades_instead_of_aborting() -> None:
    """A helmet died on ValueError: invalid color: 'oak_deep'."""
    assert _palette_color({"wood": "#8A6A3F"}, "wood") == (0x8A, 0x6A, 0x3F)
    assert _palette_color({}, "#123456") == (0x12, 0x34, 0x56)
    assert _palette_color({}, "oak_deep") == _UNRESOLVED_COLOR
    assert _palette_color({"wood": "#FFF"}, None) == _UNRESOLVED_COLOR
    assert _palette_color({"bad": "not-a-colour"}, "bad") == _UNRESOLVED_COLOR


def test_an_unrecognised_composite_mode_is_dropped_not_fatal() -> None:
    """Trousers died on 'part_reference_composite values must be overlay or replace'."""
    spec = appearance_from_dict({
        "palette": {"wood": "#8A6A3F"},
        "parts": {"plate": {"colors": ["wood"]}},
        "part_reference_composite": {"plate": "composite", "trim": "overlay"},
    })
    assert spec.part_reference_composite == {"trim": "overlay"}


def test_an_unrecognised_sampling_mode_is_dropped_not_fatal() -> None:
    spec = appearance_from_dict({
        "palette": {"wood": "#8A6A3F"},
        "parts": {"plate": {"colors": ["wood"]}},
        "part_reference_sampling": {"plate": "exact", "trim": "pattern"},
    })
    assert spec.part_reference_sampling == {"trim": "pattern"}


# -- a retry must change the request -----------------------------------

def test_a_length_limited_empty_answer_is_retried_without_thinking(monkeypatch) -> None:
    """Two live clock members died on finish_reason=length with no content.

    The model spent 200k+ reasoning characters and never reached the JSON
    contract. The old retry resent the byte-identical body, so it failed the
    same way; the retry now turns thinking off and caps the completion.
    """
    import json as _json
    import urllib.request

    from studio_next.llm import OpenAICompatibleClient

    sent: list[dict] = []

    class _Response:
        def __init__(self, payload):
            self._data = _json.dumps(payload).encode("utf-8")

        def read(self):
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout=None):
        sent.append(_json.loads(request.data.decode("utf-8")))
        if len(sent) == 1:
            # The failure that killed the clock members.
            return _Response({"choices": [{"finish_reason": "length", "message": {
                "content": "", "reasoning_content": "x" * 200000}}]})
        return _Response({"choices": [{"finish_reason": "stop", "message": {
            "content": "{\"ok\": true}", "reasoning_content": ""}}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = OpenAICompatibleClient(
        api_key="test", base_url="https://api.deepseek.com/v1", model="deepseek-flash",
        reasoning_effort="high",
    )
    assert client.complete("plan something", json_mode=True) == "{\"ok\": true}"
    assert len(sent) == 2, "an empty answer is worth exactly one retry"
    assert sent[0]["thinking"] == {"type": "enabled"}
    assert "max_tokens" not in sent[0], "the first attempt stays uncapped"
    assert sent[1]["thinking"] == {"type": "disabled"}, "the retry must differ"
    assert sent[1]["max_tokens"] >= 2048, "the retry must leave room for the contract"


def test_a_truncated_json_body_is_retried_instead_of_raising(monkeypatch) -> None:
    """A live bow member died on JSONDecodeError: line 1 column 846.

    The provider cut the object off mid-flight. json_mode promises parseable
    JSON, so an unparseable body belongs in the same retry budget as an empty
    one rather than surfacing from deep inside a planner.
    """
    import json as _json
    import urllib.request

    from studio_next.llm import OpenAICompatibleClient

    sent: list[dict] = []

    class _Response:
        def __init__(self, payload):
            self._data = _json.dumps(payload).encode("utf-8")

        def read(self):
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout=None):
        sent.append(_json.loads(request.data.decode("utf-8")))
        if len(sent) == 1:
            return _Response({"choices": [{"finish_reason": "stop", "message": {
                "content": "{\"parts\": [{\"id\": \"bow_body\", \"mean", "reasoning_content": ""}}]})
        return _Response({"choices": [{"finish_reason": "stop", "message": {
            "content": "{\"parts\": []}", "reasoning_content": ""}}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = OpenAICompatibleClient(api_key="test", base_url="https://api.deepseek.com/v1", model="deepseek-flash")
    assert client.complete("describe it", json_mode=True) == "{\"parts\": []}"
    assert len(sent) == 2
    assert sent[1]["thinking"] == {"type": "disabled"}


def test_a_non_json_caller_still_gets_whatever_came_back(monkeypatch) -> None:
    """Only json_mode is a parse promise; prose callers take the text as-is."""
    import json as _json
    import urllib.request

    from studio_next.llm import OpenAICompatibleClient

    class _Response:
        def __init__(self, payload):
            self._data = _json.dumps(payload).encode("utf-8")

        def read(self):
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout=None: _Response(
        {"choices": [{"finish_reason": "stop", "message": {"content": "just prose", "reasoning_content": ""}}]}
    ))
    client = OpenAICompatibleClient(api_key="test", base_url="https://api.deepseek.com/v1", model="deepseek-flash")
    assert client.complete("say something") == "just prose"

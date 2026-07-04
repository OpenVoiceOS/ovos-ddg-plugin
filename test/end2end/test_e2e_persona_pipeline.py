"""Full-pipeline end-to-end test for ovos-ddg-solver-plugin using ovoscope.

Proves:
  1. An utterance flows through the real OVOS intent pipeline, hits the
     persona pipeline plugin, and produces a ``speak`` message with
     non-empty text (the DDG persona answered).
  2. Per-session memory is recorded: the live PersonaService accumulates
     USER + ASSISTANT turns keyed by session_id, and an unknown session
     has no history.

Network is **fully stubbed**: ``DuckDuckGoRetrievalEngine._fetch`` is
monkeypatched at module level (before the pipeline starts) to return a
fixed abstract payload — no real HTTP call to api.duckduckgo.com is ever
made.  The rest of the OVOS pipeline (intent matching, persona routing,
TTS speak emission, short-term memory) runs against real installed code.
"""
import json
import os
import tempfile
from unittest.mock import patch

import pytest

import ovoscope
import ovos_persona

from ovos_bus_client.message import Message
from ovos_bus_client.session import Session, SessionManager

from ovoscope import (
    PERSONA_PIPELINE,
    CaptureSession,
    get_minicroft,
    is_pipeline_available,
)

# ---------------------------------------------------------------------------
# Stub: DDG network boundary
#
# ``DuckDuckGoRetrievalEngine._fetch`` is the single method that calls
# ``requests.get`` against api.duckduckgo.com.  Every higher-level method
# (``_search``, ``query``, ``get_infobox``, ``get_image``) routes through it.
# Patching here is the narrowest possible boundary — one real network socket
# is never opened.
# ---------------------------------------------------------------------------

_STUB_PAYLOAD = {
    "AbstractText": (
        "DuckDuckGo is a privacy-focused internet search engine that does not "
        "track its users and provides instant answers without ads."
    ),
    "AbstractSource": "Wikipedia",
    "Image": "",
    "Infobox": {},
    "RelatedTopics": [],
}


def _stub_fetch(self, query: str, lang: str):  # noqa: ANN001
    """Drop-in replacement for DuckDuckGoRetrievalEngine._fetch — offline."""
    return _STUB_PAYLOAD


_fetch_patcher = patch(
    "ovos_ddg_plugin.DuckDuckGoRetrievalEngine._fetch",
    new=_stub_fetch,
)

# ---------------------------------------------------------------------------
# Persona definition
# ---------------------------------------------------------------------------

PERSONA_NAME = "DDGBot"


def _make_personas_dir() -> str:
    """Write a minimal DDG persona JSON into a temp directory and return the path."""
    tmpdir = tempfile.mkdtemp()
    persona = {
        "name": PERSONA_NAME,
        "handlers": ["ovos-ddg-plugin"],
        "ovos-ddg-plugin": {},
    }
    with open(os.path.join(tmpdir, f"{PERSONA_NAME}.json"), "w") as fh:
        json.dump(persona, fh)
    return tmpdir


# ---------------------------------------------------------------------------
# Module-level MiniCroft (shared across tests for speed)
# ---------------------------------------------------------------------------

PERSONAS_PATH = _make_personas_dir()

PIPELINE_CONFIG = {
    "persona": {
        "personas_path": PERSONAS_PATH,
        "default_persona": PERSONA_NAME,
        "short-term-memory": True,
        "handle_fallback": True,
        "ignore_plugin_personas": True,
    }
}

TEST_PIPELINE = [
    "ovos-persona-pipeline-plugin-high",
    "ovos-persona-pipeline-plugin-low",
]

# Spoken output is emitted on the OVOS-AUDIO spec topic ``ovos.utterance.speak``;
# the legacy ``speak`` topic is accepted for back-compat with older components.
SPEAK_TOPICS = ("ovos.utterance.speak", "speak")


@pytest.fixture(scope="module")
def mc():
    """Shared MiniCroft instance with DDG network fully stubbed."""
    _fetch_patcher.start()
    croft = get_minicroft(
        skill_ids=[],
        default_pipeline=TEST_PIPELINE,
        pipeline_config=PIPELINE_CONFIG,
    )
    yield croft
    croft.stop()
    _fetch_patcher.stop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utterance_msg(utterance: str, sess: Session) -> Message:
    return Message(
        "recognizer_loop:utterance",
        {"utterances": [utterance], "lang": sess.lang},
        {"session": sess.serialize()},
    )


def _drive_utterance(croft, sess: Session, utterance: str, timeout: int = 30):
    cap = CaptureSession(
        croft,
        eof_msgs=["ovos.utterance.handled", "ovos.utterance.cancelled"],
    )
    cap.capture(_utterance_msg(utterance, sess), timeout=timeout)
    return cap.finish()


def _get_persona_service(croft):
    return croft.intents.pipeline_plugins["ovos-persona-pipeline-plugin"]


# ---------------------------------------------------------------------------
# Test 1: DDG persona speaks through the full pipeline
# ---------------------------------------------------------------------------

class TestDDGPersonaSpeaksThroughPipeline:
    """An utterance must traverse the full OVOS intent pipeline and produce a
    speak message with non-empty utterance text from the stubbed DDG engine."""

    def test_pipeline_produces_speak(self, mc):
        sess = Session(session_id="ddg-e2e-speak-test")
        SessionManager.sessions[sess.session_id] = sess

        messages = _drive_utterance(mc, sess, "what is duckduckgo", timeout=30)

        msg_types = [m.msg_type for m in messages]
        speak_msgs = [m for m in messages if m.msg_type in SPEAK_TOPICS]

        assert speak_msgs, (
            f"Expected at least one 'speak' message; got msg_types: {msg_types}"
        )
        spoken = speak_msgs[0].data.get("utterance", "")
        assert spoken.strip(), (
            f"'speak' message had an empty utterance; data={speak_msgs[0].data}"
        )

    def test_speak_text_comes_from_stub(self, mc):
        """The spoken text must derive from the fixed stub payload, not the network."""
        sess = Session(session_id="ddg-e2e-stub-content")
        SessionManager.sessions[sess.session_id] = sess

        messages = _drive_utterance(mc, sess, "tell me about duckduckgo", timeout=30)

        speak_msgs = [m for m in messages if m.msg_type in SPEAK_TOPICS]
        assert speak_msgs, "Expected a speak message"

        # The stub abstract starts with "DuckDuckGo is a privacy-focused…"
        spoken = speak_msgs[0].data.get("utterance", "")
        assert spoken.strip(), f"Spoken text is empty; data={speak_msgs[0].data}"

    def test_speak_on_generic_query(self, mc):
        """Any query should produce a speak message because the stub always returns an abstract."""
        sess = Session(session_id="ddg-e2e-generic-query")
        SessionManager.sessions[sess.session_id] = sess

        messages = _drive_utterance(mc, sess, "search for something", timeout=30)

        for msg in messages:
            if msg.msg_type in SPEAK_TOPICS:
                assert msg.data.get("utterance", "").strip(), (
                    f"speak message has empty utterance: {msg.data}"
                )
                return

        pytest.fail(
            f"No 'speak' message found in pipeline output. "
            f"Message types received: {[m.msg_type for m in messages]}"
        )


# ---------------------------------------------------------------------------
# Test 2: per-session memory is recorded
# ---------------------------------------------------------------------------

class TestDDGPerSessionMemory:
    """PersonaService records USER+ASSISTANT turns per session_id.

    The live PersonaService is obtained from the MiniCroft pipeline registry
    (mc.intents.pipeline_plugins["ovos-persona-pipeline-plugin"]).
    """

    def test_user_turn_recorded_in_memory(self, mc):
        svc = _get_persona_service(mc)
        sess = Session(session_id="ddg-e2e-mem-user")
        SessionManager.sessions[sess.session_id] = sess

        persona = svc.personas.get(PERSONA_NAME)
        assert persona is not None, f"Persona '{PERSONA_NAME}' not loaded"
        assert persona.memory is not None, "Persona must have short-term memory enabled"

        _drive_utterance(mc, sess, "what is privacy", timeout=30)

        history = persona.memory.get_history(sess.session_id)
        contents = [m.content for m in history]
        assert any("privacy" in c for c in contents), (
            f"User utterance not found in memory for session {sess.session_id}. "
            f"History: {contents}"
        )

    def test_assistant_response_recorded_in_memory(self, mc):
        from ovos_plugin_manager.templates.agents import MessageRole

        svc = _get_persona_service(mc)
        sess = Session(session_id="ddg-e2e-mem-assistant")
        SessionManager.sessions[sess.session_id] = sess

        persona = svc.personas.get(PERSONA_NAME)
        assert persona is not None
        assert persona.memory is not None

        _drive_utterance(mc, sess, "tell me about search engines", timeout=30)

        history = persona.memory.get_history(sess.session_id)
        roles = [m.role for m in history]
        assert MessageRole.ASSISTANT in roles, (
            f"No ASSISTANT turn recorded in memory. History roles: {roles}"
        )

    def test_unknown_session_has_empty_history(self, mc):
        svc = _get_persona_service(mc)
        persona = svc.personas.get(PERSONA_NAME)
        assert persona is not None
        assert persona.memory is not None

        # Drive a known session first so the service is warmed up
        sess = Session(session_id="ddg-e2e-mem-known")
        SessionManager.sessions[sess.session_id] = sess
        _drive_utterance(mc, sess, "hello", timeout=30)

        unknown_history = persona.memory.get_history("session-that-never-existed-ddg")
        assert unknown_history == [], (
            f"Expected empty history for unknown session, got: {unknown_history}"
        )

    def test_same_session_accumulates_turns(self, mc):
        svc = _get_persona_service(mc)
        sess = Session(session_id="ddg-e2e-mem-accumulate")
        SessionManager.sessions[sess.session_id] = sess

        persona = svc.personas.get(PERSONA_NAME)
        assert persona is not None
        assert persona.memory is not None
        persona.memory.session2history.pop(sess.session_id, None)

        _drive_utterance(mc, sess, "first question about duckduckgo", timeout=30)
        _drive_utterance(mc, sess, "second question about privacy", timeout=30)

        history = persona.memory.get_history(sess.session_id)
        assert len(history) >= 2, (
            f"Expected at least 2 history entries after two turns, got {len(history)}: "
            f"{[m.content for m in history]}"
        )

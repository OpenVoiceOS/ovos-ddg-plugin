"""
Unit tests for ovos-ddg-plugin.

All network calls and OVOS config are mocked — no API key or running daemon required.
"""
import datetime
import unittest
from unittest.mock import MagicMock, patch, call

from ovos_ddg_plugin import (
    DuckDuckGoRetrievalEngine,
    DuckDuckGoToolbox,
    SearchDuckDuckGoArgs,
    SearchDuckDuckGoOutput,
    DDGImageArgs,
    DDGImageOutput,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine() -> DuckDuckGoRetrievalEngine:
    """Return a bare engine with __init__ bypassed (no locale files, no config)."""
    engine = DuckDuckGoRetrievalEngine.__new__(DuckDuckGoRetrievalEngine)
    engine.config = {}
    engine._kword_extractors = {}
    engine._intent_matchers = {}
    return engine


def _infobox_payload(label: str, value) -> dict:
    return {"Infobox": {"content": [{"label": label, "value": value}]}, "RelatedTopics": []}


# ---------------------------------------------------------------------------
# _fetch
# ---------------------------------------------------------------------------

class TestFetch(unittest.TestCase):

    def setUp(self):
        self.engine = _make_engine()

    def test_returns_json_on_success(self):
        payload = {"AbstractText": "Isaac Newton was a physicist."}
        with patch("ovos_ddg_plugin.requests.get") as mock_get:
            mock_get.return_value.json.return_value = payload
            result = self.engine._fetch("Isaac Newton", lang="en-US")
        self.assertEqual(result, payload)

    def test_sends_correct_kl_param(self):
        with patch("ovos_ddg_plugin.requests.get") as mock_get:
            mock_get.return_value.json.return_value = {}
            self.engine._fetch("test", lang="en-US")
        _, kwargs = mock_get.call_args
        self.assertEqual(kwargs["params"]["kl"], "us-en")

    def test_returns_empty_on_network_exception(self):
        with patch("ovos_ddg_plugin.requests.get", side_effect=OSError("timeout")):
            result = self.engine._fetch("Isaac Newton", lang="en-US")
        self.assertEqual(result, {})

    def test_returns_empty_on_json_exception(self):
        with patch("ovos_ddg_plugin.requests.get") as mock_get:
            mock_get.return_value.json.side_effect = ValueError("bad json")
            result = self.engine._fetch("Isaac Newton", lang="en-US")
        self.assertEqual(result, {})

    def test_returns_empty_for_unknown_locale(self):
        result = self.engine._fetch("query", lang="xx-XX")
        self.assertEqual(result, {})

    def test_known_locales_map_correctly(self):
        """Spot-check a handful of DDG's reversed locale codes."""
        cases = [("de-DE", "de-de"), ("pt-PT", "pt-pt"), ("fr-FR", "fr-fr")]
        for bcp47, expected_kl in cases:
            with patch("ovos_ddg_plugin.requests.get") as mock_get:
                mock_get.return_value.json.return_value = {}
                self.engine._fetch("test", lang=bcp47)
            _, kwargs = mock_get.call_args
            self.assertEqual(kwargs["params"]["kl"], expected_kl, bcp47)


# ---------------------------------------------------------------------------
# _search
# ---------------------------------------------------------------------------

class TestSearch(unittest.TestCase):

    def setUp(self):
        self.engine = _make_engine()

    def test_returns_direct_match_without_keyword_extraction(self):
        payload = {"AbstractText": "Newton invented calculus."}
        self.engine._fetch = MagicMock(return_value=payload)
        result = self.engine._search("Isaac Newton", lang="en-us")
        self.assertEqual(result, payload)
        self.engine._fetch.assert_called_once()

    def test_falls_back_to_keyword_extraction_on_empty_abstract(self):
        self.engine._fetch = MagicMock(side_effect=[
            {},
            {"AbstractText": "Gravity pulls objects downward."},
        ])
        kwx = MagicMock()
        kwx.extract.return_value = ["gravity"]
        self.engine._keyword_extractor = MagicMock(return_value=kwx)
        result = self.engine._search("what is the force of gravity", lang="en-us")
        self.assertEqual(result["AbstractText"], "Gravity pulls objects downward.")
        self.assertEqual(self.engine._fetch.call_count, 2)

    def test_returns_empty_when_no_keyword_extractor(self):
        self.engine._fetch = MagicMock(return_value={})
        self.engine._keyword_extractor = MagicMock(return_value=None)
        result = self.engine._search("gibberish query", lang="en-us")
        self.assertEqual(result, {})

    def test_returns_empty_when_keyword_extraction_finds_nothing(self):
        self.engine._fetch = MagicMock(return_value={})
        kwx = MagicMock()
        kwx.extract.return_value = []
        self.engine._keyword_extractor = MagicMock(return_value=kwx)
        result = self.engine._search("???", lang="en-us")
        self.assertEqual(result, {})

    def test_uses_highest_scoring_keyword(self):
        self.engine._fetch = MagicMock(side_effect=[{}, {}])
        kwx = MagicMock()
        kwx.extract.return_value = ["apple", "zebra", "mango"]
        self.engine._keyword_extractor = MagicMock(return_value=kwx)
        self.engine._search("some query", lang="en-us")
        second_call_query = self.engine._fetch.call_args_list[1][0][0]
        self.assertEqual(second_call_query, max(["apple", "zebra", "mango"]))


# ---------------------------------------------------------------------------
# get_infobox
# ---------------------------------------------------------------------------

class TestGetInfobox(unittest.TestCase):

    def setUp(self):
        self.engine = _make_engine()

    def test_returns_plain_infobox_field(self):
        self.engine._search = MagicMock(return_value=_infobox_payload("Known for", "Calculus"))
        with patch("ovos_ddg_plugin.Configuration", return_value={}):
            infobox, _ = self.engine.get_infobox("Newton", lang="en-us")
        self.assertEqual(infobox["known for"], "Calculus")

    def test_label_normalised_to_lowercase(self):
        self.engine._search = MagicMock(return_value=_infobox_payload("Alma Mater", "Cambridge"))
        with patch("ovos_ddg_plugin.Configuration", return_value={}):
            infobox, _ = self.engine.get_infobox("Newton", lang="en-us")
        self.assertIn("alma mater", infobox)
        self.assertNotIn("Alma Mater", infobox)

    def test_date_fields_formatted_via_nice_date(self):
        dt_str = "+1942-01-08T00:00:00Z"
        self.engine._search = MagicMock(return_value=_infobox_payload("Born", {"time": dt_str}))
        fake_date = "January 8, 1942"
        with patch("ovos_ddg_plugin.Configuration", return_value={}), \
             patch("ovos_ddg_plugin.nice_date", return_value=fake_date) as mock_nd:
            infobox, _ = self.engine.get_infobox("Hawking", lang="en-us")
        mock_nd.assert_called_once()
        self.assertEqual(infobox["born"], fake_date)

    def test_died_field_also_formatted(self):
        dt_str = "+2018-03-14T00:00:00Z"
        self.engine._search = MagicMock(return_value=_infobox_payload("Died", {"time": dt_str}))
        with patch("ovos_ddg_plugin.Configuration", return_value={}), \
             patch("ovos_ddg_plugin.nice_date", return_value="March 14, 2018"):
            infobox, _ = self.engine.get_infobox("Hawking", lang="en-us")
        self.assertIn("died", infobox)

    def test_malformed_entry_skipped_silently(self):
        # entry whose value raises on strptime (bad time string) should be skipped
        payload = {"Infobox": {"content": [{"label": "Born", "value": {"time": "not-a-date"}}]}, "RelatedTopics": []}
        self.engine._search = MagicMock(return_value=payload)
        with patch("ovos_ddg_plugin.Configuration", return_value={}):
            infobox, _ = self.engine.get_infobox("thing", lang="en-us")
        self.assertEqual(infobox, {})

    def test_returns_related_topics(self):
        payload = {
            "Infobox": {"content": []},
            "RelatedTopics": [{"Text": "gravity"}, {"Text": "optics"}],
        }
        self.engine._search = MagicMock(return_value=payload)
        with patch("ovos_ddg_plugin.Configuration", return_value={}):
            _, related = self.engine.get_infobox("Newton", lang="en-us")
        self.assertEqual(related, ["gravity", "optics"])

    def test_returns_empty_on_no_data(self):
        self.engine._search = MagicMock(return_value={})
        with patch("ovos_ddg_plugin.Configuration", return_value={}):
            infobox, related = self.engine.get_infobox("xyzzy", lang="en-us")
        self.assertEqual(infobox, {})
        self.assertEqual(related, [])

    def test_lang_defaults_from_config(self):
        self.engine._search = MagicMock(return_value={})
        with patch("ovos_ddg_plugin.Configuration", return_value={"lang": "de-de"}):
            self.engine.get_infobox("test")
        self.engine._search.assert_called_once_with("test", "de-de")


# ---------------------------------------------------------------------------
# get_image
# ---------------------------------------------------------------------------

class TestGetImage(unittest.TestCase):

    def setUp(self):
        self.engine = _make_engine()

    def test_returns_absolute_url_unchanged(self):
        self.engine._search = MagicMock(return_value={"Image": "https://duckduckgo.com/i/abc.jpg"})
        with patch("ovos_ddg_plugin.Configuration", return_value={}):
            result = self.engine.get_image("Newton")
        self.assertEqual(result, "https://duckduckgo.com/i/abc.jpg")

    def test_prepends_ddg_domain_for_relative_path(self):
        self.engine._search = MagicMock(return_value={"Image": "/i/abc.jpg"})
        with patch("ovos_ddg_plugin.Configuration", return_value={}):
            result = self.engine.get_image("Newton")
        self.assertEqual(result, "https://duckduckgo.com/i/abc.jpg")

    def test_returns_none_when_image_field_absent(self):
        self.engine._search = MagicMock(return_value={"AbstractText": "Some text."})
        with patch("ovos_ddg_plugin.Configuration", return_value={}):
            result = self.engine.get_image("Newton")
        self.assertIsNone(result)

    def test_returns_none_when_image_field_empty_string(self):
        self.engine._search = MagicMock(return_value={"Image": ""})
        with patch("ovos_ddg_plugin.Configuration", return_value={}):
            result = self.engine.get_image("Newton")
        self.assertIsNone(result)

    def test_lang_passed_to_search(self):
        self.engine._search = MagicMock(return_value={})
        with patch("ovos_ddg_plugin.Configuration", return_value={}):
            self.engine.get_image("Newton", lang="de-de")
        self.engine._search.assert_called_once_with("Newton", "de-de")


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------

class TestQuery(unittest.TestCase):

    def setUp(self):
        self.engine = _make_engine()

    def test_returns_list_of_tuples(self):
        self.engine._match_infobox_intent = MagicMock(return_value=(None, "Newton"))
        self.engine._search = MagicMock(return_value={"AbstractText": "Newton was a physicist."})
        with patch("ovos_ddg_plugin.Configuration", return_value={}):
            results = self.engine.query("Newton", lang="en-us")
        self.assertIsInstance(results, list)
        self.assertTrue(all(isinstance(r, tuple) and len(r) == 2 for r in results))

    def test_respects_k_limit(self):
        self.engine._match_infobox_intent = MagicMock(return_value=(None, "Newton"))
        self.engine._search = MagicMock(return_value={"AbstractText": "A. B. C. D. E. F."})
        with patch("ovos_ddg_plugin.Configuration", return_value={}):
            results = self.engine.query("Newton", lang="en-us", k=2)
        self.assertLessEqual(len(results), 2)

    def test_returns_empty_list_when_no_data(self):
        self.engine._match_infobox_intent = MagicMock(return_value=(None, "xyzzy"))
        self.engine._search = MagicMock(return_value={})
        with patch("ovos_ddg_plugin.Configuration", return_value={}):
            results = self.engine.query("xyzzy", lang="en-us")
        self.assertEqual(results, [])

    def test_infobox_match_returns_single_result_score_09(self):
        self.engine._match_infobox_intent = MagicMock(return_value=("born", "Hawking"))
        self.engine.get_infobox = MagicMock(return_value=({"born": "8 January 1942"}, []))
        with patch("ovos_ddg_plugin.Configuration", return_value={}):
            results = self.engine.query("when was Hawking born", lang="en-us")
        self.assertEqual(results, [("8 January 1942", 0.9)])

    def test_infobox_miss_falls_through_to_abstract(self):
        self.engine._match_infobox_intent = MagicMock(return_value=("born", "Hawking"))
        self.engine.get_infobox = MagicMock(return_value=({}, []))  # no "born" field
        self.engine._search = MagicMock(return_value={"AbstractText": "Hawking was a physicist."})
        with patch("ovos_ddg_plugin.Configuration", return_value={}):
            results = self.engine.query("when was Hawking born", lang="en-us")
        self.assertTrue(len(results) > 0)
        self.assertNotEqual(results[0][0], "8 January 1942")

    def test_abstract_scores_decrease_with_sentence_position(self):
        self.engine._match_infobox_intent = MagicMock(return_value=(None, "Newton"))
        self.engine._search = MagicMock(
            return_value={"AbstractText": "Newton was a physicist. He invented calculus. He studied at Cambridge."}
        )
        with patch("ovos_ddg_plugin.Configuration", return_value={}):
            results = self.engine.query("Newton", lang="en-us", k=3)
        scores = [s for _, s in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_empty_sentences_filtered_out(self):
        self.engine._match_infobox_intent = MagicMock(return_value=(None, "Newton"))
        with patch("ovos_ddg_plugin.sentence_tokenize", return_value=["Real sentence.", "", "  "]), \
             patch("ovos_ddg_plugin.Configuration", return_value={}):
            self.engine._search = MagicMock(return_value={"AbstractText": "anything"})
            results = self.engine.query("Newton", lang="en-us", k=5)
        self.assertTrue(all(text.strip() for text, _ in results))

    def test_lang_defaults_from_config_when_not_passed(self):
        self.engine._match_infobox_intent = MagicMock(return_value=(None, "x"))
        self.engine._search = MagicMock(return_value={})
        with patch("ovos_ddg_plugin.Configuration", return_value={"lang": "pt-pt"}):
            self.engine.query("x")
        self.engine._match_infobox_intent.assert_called_once_with("x", "pt-pt")


# ---------------------------------------------------------------------------
# Intent matching
# ---------------------------------------------------------------------------

class TestIntentMatching(unittest.TestCase):

    def setUp(self):
        self.engine = _make_engine()

    def test_no_matchers_returns_utterance_and_none_intent(self):
        intent, kw = self.engine._match_infobox_intent("when was Einstein born", "en")
        self.assertIsNone(intent)
        self.assertEqual(kw, "when was Einstein born")

    def test_register_and_match_extracts_entity(self):
        self.engine._register_intent("born", ["when was {query} born"], "en")
        intent, kw = self.engine._match_infobox_intent("when was Einstein born", "en")
        self.assertEqual(intent, "born")
        self.assertEqual(kw, "Einstein")

    def test_lang_uses_only_base_code(self):
        """en-US and en-GB should both hit the 'en' matcher."""
        self.engine._register_intent("born", ["when was {query} born"], "en-US")
        intent_gb, _ = self.engine._match_infobox_intent("when was Newton born", "en-GB")
        self.assertEqual(intent_gb, "born")

    def test_no_match_returns_full_utterance_as_keyword(self):
        self.engine._register_intent("born", ["when was {query} born"], "en")
        intent, kw = self.engine._match_infobox_intent("tell me about gravity", "en")
        self.assertIsNone(intent)
        self.assertEqual(kw, "tell me about gravity")

    def test_multiple_intents_registered_for_same_lang(self):
        self.engine._register_intent("born", ["when was {query} born"], "en")
        self.engine._register_intent("died", ["when did {query} die"], "en")
        intent, kw = self.engine._match_infobox_intent("when did Newton die", "en")
        self.assertEqual(intent, "died")
        self.assertEqual(kw, "Newton")

    def test_separate_matchers_per_language(self):
        self.engine._register_intent("born", ["when was {query} born"], "en")
        self.engine._register_intent("born", ["quando nasceu {query}"], "pt")
        intent_en, kw_en = self.engine._match_infobox_intent("when was Newton born", "en")
        intent_pt, kw_pt = self.engine._match_infobox_intent("quando nasceu Newton", "pt")
        self.assertEqual(intent_en, "born")
        self.assertEqual(intent_pt, "born")
        self.assertEqual(kw_en, "Newton")
        self.assertEqual(kw_pt, "Newton")


# ---------------------------------------------------------------------------
# DuckDuckGoToolbox
# ---------------------------------------------------------------------------

class TestDuckDuckGoToolbox(unittest.TestCase):

    def _make_toolbox(self) -> DuckDuckGoToolbox:
        with patch("ovos_ddg_plugin.DuckDuckGoRetrievalEngine"):
            tb = DuckDuckGoToolbox(config={})
        tb._engine = MagicMock()
        return tb

    def test_toolbox_id_is_correct(self):
        self.assertEqual(DuckDuckGoToolbox.toolbox_id, "ovos-ddg-tools")

    def test_discover_tools_returns_exactly_two_tools(self):
        tb = self._make_toolbox()
        tools = tb.discover_tools()
        self.assertEqual(len(tools), 2)

    def test_discover_tools_has_search_and_image_tools(self):
        tb = self._make_toolbox()
        names = {t.name for t in tb.discover_tools()}
        self.assertIn("search_duckduckgo", names)
        self.assertIn("duckduckgo_image", names)

    def test_search_ddg_returns_first_query_result(self):
        tb = self._make_toolbox()
        tb._engine.query.return_value = [("Newton invented calculus.", 0.9), ("He studied optics.", 0.8)]
        result = tb.search_ddg(SearchDuckDuckGoArgs(query="Isaac Newton", lang="en-us"))
        self.assertIsInstance(result, SearchDuckDuckGoOutput)
        self.assertEqual(result.result, "Newton invented calculus.")

    def test_search_ddg_passes_lang_to_engine(self):
        tb = self._make_toolbox()
        tb._engine.query.return_value = []
        tb.search_ddg(SearchDuckDuckGoArgs(query="test", lang="de-de"))
        tb._engine.query.assert_called_once_with("test", lang="de-de", k=1)

    def test_search_ddg_returns_empty_string_when_no_results(self):
        tb = self._make_toolbox()
        tb._engine.query.return_value = []
        result = tb.search_ddg(SearchDuckDuckGoArgs(query="xyzzy", lang="en-us"))
        self.assertEqual(result.result, "")

    def test_ddg_image_returns_url(self):
        tb = self._make_toolbox()
        tb._engine.get_image.return_value = "https://duckduckgo.com/i/abc.jpg"
        result = tb.ddg_image(DDGImageArgs(query="Newton", lang="en-us"))
        self.assertIsInstance(result, DDGImageOutput)
        self.assertEqual(result.url, "https://duckduckgo.com/i/abc.jpg")

    def test_ddg_image_passes_lang_to_engine(self):
        tb = self._make_toolbox()
        tb._engine.get_image.return_value = None
        tb.ddg_image(DDGImageArgs(query="Newton", lang="fr-fr"))
        tb._engine.get_image.assert_called_once_with("Newton", lang="fr-fr")

    def test_ddg_image_returns_none_when_no_image(self):
        tb = self._make_toolbox()
        tb._engine.get_image.return_value = None
        result = tb.ddg_image(DDGImageArgs(query="xyzzy", lang="en-us"))
        self.assertIsNone(result.url)

    def test_tool_schemas_are_wired_correctly(self):
        tb = self._make_toolbox()
        tools = {t.name: t for t in tb.discover_tools()}
        self.assertIs(tools["search_duckduckgo"].argument_schema, SearchDuckDuckGoArgs)
        self.assertIs(tools["search_duckduckgo"].output_schema, SearchDuckDuckGoOutput)
        self.assertIs(tools["duckduckgo_image"].argument_schema, DDGImageArgs)
        self.assertIs(tools["duckduckgo_image"].output_schema, DDGImageOutput)


# ---------------------------------------------------------------------------
# Plugin loading / entry points
# ---------------------------------------------------------------------------

class TestPluginLoading(unittest.TestCase):

    def test_public_api_importable(self):
        from ovos_ddg_plugin import (
            DuckDuckGoRetrievalEngine,
            DuckDuckGoToolbox,
            SearchDuckDuckGoArgs,
            SearchDuckDuckGoOutput,
            DDGImageArgs,
            DDGImageOutput,
        )

    def test_engine_is_retrieval_engine_subclass(self):
        from ovos_plugin_manager.templates.agents import RetrievalEngine
        self.assertTrue(issubclass(DuckDuckGoRetrievalEngine, RetrievalEngine))

    def test_toolbox_is_toolbox_subclass(self):
        from ovos_plugin_manager.templates.agent_tools import ToolBox
        self.assertTrue(issubclass(DuckDuckGoToolbox, ToolBox))


if __name__ == "__main__":
    unittest.main()

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
    DDGInfoboxArgs,
    DDGInfoboxOutput,
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
        # Labels are lowercased and spaces replaced with underscores
        self.assertEqual(infobox["known_for"], "Calculus")

    def test_label_normalised_to_lowercase_and_underscores(self):
        self.engine._search = MagicMock(return_value=_infobox_payload("Alma Mater", "Cambridge"))
        with patch("ovos_ddg_plugin.Configuration", return_value={}):
            infobox, _ = self.engine.get_infobox("Newton", lang="en-us")
        self.assertIn("alma_mater", infobox)
        self.assertNotIn("Alma Mater", infobox)
        self.assertNotIn("alma mater", infobox)

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

    def test_possessive_s_stripped_before_matching(self):
        # "Darwin's father" should match the "father" intent and extract "Darwin"
        self.engine._load_intents()
        intent, kw = self.engine._match_infobox_intent("Darwin's father", "en")
        self.assertEqual(intent, "father")
        self.assertEqual(kw.lower(), "darwin")

    def test_field_alias_alma_mater_resolves_via_education(self):
        # alma_mater intent → DDG stores as "education" key
        self.engine._match_infobox_intent = MagicMock(return_value=("alma_mater", "Hawking"))
        self.engine.get_infobox = MagicMock(return_value=({"education": "University of Cambridge"}, []))
        with patch("ovos_ddg_plugin.Configuration", return_value={}):
            results = self.engine.query("what is Hawking alma mater", lang="en-us")
        self.assertEqual(results, [("University of Cambridge", 0.9)])

    def test_field_alias_resting_place_resolves_via_burial(self):
        # resting_place intent → DDG sometimes stores as "burial"
        self.engine._match_infobox_intent = MagicMock(return_value=("resting_place", "Queen Elizabeth II"))
        self.engine.get_infobox = MagicMock(return_value=({"burial": "Windsor Castle"}, []))
        with patch("ovos_ddg_plugin.Configuration", return_value={}):
            results = self.engine.query("where is Queen Elizabeth II buried", lang="en-us")
        self.assertEqual(results, [("Windsor Castle", 0.9)])

    def test_non_keyword_entity_label_extracted(self):
        # Intent files may use {person}, {movie}, etc. — code must extract any entity, not just {keyword}
        self.engine._load_intents()
        intent, kw = self.engine._match_infobox_intent("who directed The Godfather", "en")
        self.assertEqual(intent, "director")
        self.assertEqual(kw.lower(), "the godfather")

    def test_infobox_key_spaces_normalised_to_underscores(self):
        # "age at death" from DDG → stored as "age_at_death" in infobox
        self.engine._search = MagicMock(return_value={"Infobox": {"content": [
            {"label": "Age at death", "value": "76 years"}
        ]}, "RelatedTopics": []})
        with patch("ovos_ddg_plugin.Configuration", return_value={}):
            infobox, _ = self.engine.get_infobox("Hawking", lang="en-us")
        self.assertEqual(infobox.get("age_at_death"), "76 years")
        self.assertNotIn("age at death", infobox)

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
        self.engine._register_intent("born", ["when was {keyword} born"], "en")
        intent, kw = self.engine._match_infobox_intent("when was Einstein born", "en")
        self.assertEqual(intent, "born")
        self.assertEqual(kw, "Einstein")

    def test_lang_uses_only_base_code(self):
        """en-US and en-GB should both hit the 'en' matcher."""
        self.engine._register_intent("born", ["when was {keyword} born"], "en-US")
        intent_gb, _ = self.engine._match_infobox_intent("when was Newton born", "en-GB")
        self.assertEqual(intent_gb, "born")

    def test_no_match_returns_full_utterance_as_keyword(self):
        self.engine._register_intent("born", ["when was {keyword} born"], "en")
        intent, kw = self.engine._match_infobox_intent("tell me about gravity", "en")
        self.assertIsNone(intent)
        self.assertEqual(kw, "tell me about gravity")

    def test_multiple_intents_registered_for_same_lang(self):
        self.engine._register_intent("born", ["when was {keyword} born"], "en")
        self.engine._register_intent("died", ["when did {keyword} die"], "en")
        intent, kw = self.engine._match_infobox_intent("when did Newton die", "en")
        self.assertEqual(intent, "died")
        self.assertEqual(kw, "Newton")

    def test_separate_matchers_per_language(self):
        self.engine._register_intent("born", ["when was {keyword} born"], "en")
        self.engine._register_intent("born", ["quando nasceu {keyword}"], "pt")
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

    def test_discover_tools_returns_exactly_three_tools(self):
        tb = self._make_toolbox()
        tools = tb.discover_tools()
        self.assertEqual(len(tools), 3)

    def test_discover_tools_has_all_three_tools(self):
        tb = self._make_toolbox()
        names = {t.name for t in tb.discover_tools()}
        self.assertIn("search_duckduckgo", names)
        self.assertIn("duckduckgo_infobox", names)
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

    def test_ddg_infobox_returns_infobox_and_related(self):
        tb = self._make_toolbox()
        tb._engine.get_infobox.return_value = ({"born": "November 7, 1867"}, ["Radioactivity"])
        result = tb.ddg_infobox(DDGInfoboxArgs(query="Marie Curie", lang="en-us"))
        self.assertIsInstance(result, DDGInfoboxOutput)
        self.assertEqual(result.infobox["born"], "November 7, 1867")
        self.assertEqual(result.related_topics, ["Radioactivity"])

    def test_ddg_infobox_passes_lang_to_engine(self):
        tb = self._make_toolbox()
        tb._engine.get_infobox.return_value = ({}, [])
        tb.ddg_infobox(DDGInfoboxArgs(query="Newton", lang="de-de"))
        tb._engine.get_infobox.assert_called_once_with("Newton", lang="de-de")

    def test_ddg_infobox_returns_empty_on_no_data(self):
        tb = self._make_toolbox()
        tb._engine.get_infobox.return_value = ({}, [])
        result = tb.ddg_infobox(DDGInfoboxArgs(query="xyzzy", lang="en-us"))
        self.assertEqual(result.infobox, {})
        self.assertEqual(result.related_topics, [])

    def test_tool_schemas_are_wired_correctly(self):
        tb = self._make_toolbox()
        tools = {t.name: t for t in tb.discover_tools()}
        self.assertIs(tools["search_duckduckgo"].argument_schema, SearchDuckDuckGoArgs)
        self.assertIs(tools["search_duckduckgo"].output_schema, SearchDuckDuckGoOutput)
        self.assertIs(tools["duckduckgo_infobox"].argument_schema, DDGInfoboxArgs)
        self.assertIs(tools["duckduckgo_infobox"].output_schema, DDGInfoboxOutput)
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
            DDGInfoboxArgs,
            DDGInfoboxOutput,
        )

    def test_engine_is_retrieval_engine_subclass(self):
        from ovos_plugin_manager.templates.agents import RetrievalEngine
        self.assertTrue(issubclass(DuckDuckGoRetrievalEngine, RetrievalEngine))

    def test_toolbox_is_toolbox_subclass(self):
        from ovos_plugin_manager.templates.agent_tools import ToolBox
        self.assertTrue(issubclass(DuckDuckGoToolbox, ToolBox))


# ---------------------------------------------------------------------------
# Intent parsing — all languages
# ---------------------------------------------------------------------------

# (utterance, expected_intent, expected_keyword)
# Only intents with actual content in the locale files are included.
# Ambiguous utterances (tied intents) are avoided; specific forms are used instead.
_INTENT_CASES = {
    "ca": [
        ("quan va néixer Albert Einstein", "born", "Albert Einstein"),
        ("quan va morir Stephen Hawking", "died", "Stephen Hawking"),
        ("per a què es coneix Marie Curie", "known_for", "Marie Curie"),
        ("on està enterrat Newton", "resting_place", "Newton"),
        ("quants fills té Darwin", "children", "Darwin"),
        ("quina és l'alma mater de Hawking", "alma_mater", "Hawking"),
        ("on va estudiar Einstein", "education", "Einstein"),
        ("quin és el lloc web oficial de Newton", "official_website", "Newton"),
        ("quants anys tenia Darwin en morir", "age_at_death", "Darwin"),
    ],
    "da": [
        ("hvornår blev Albert Einstein født", "born", "Albert Einstein"),
        ("hvornår døde Stephen Hawking", "died", "Stephen Hawking"),
        ("hvad er Marie Curie berømt for", "known_for", "Marie Curie"),
        ("hvor ligger Newton begravet", "resting_place", "Newton"),
        ("hvor mange børn har Darwin har", "children", "Darwin"),
        ("hvad er Hawking alma mater", "alma_mater", "Hawking"),
        ("hvor studerede Einstein", "education", "Einstein"),
        ("hvad er Newton officielle hjemmeside", "official_website", "Newton"),
        ("hvad er Darwin specialeemne", "thesis", "Darwin"),
        ("hvor gammel var Hawking ved hans død", "age_at_death", "Hawking"),
    ],
    "de": [
        ("wann wurde Albert Einstein geboren", "born", "Albert Einstein"),
        ("wann ist Stephen Hawking gestorben", "died", "Stephen Hawking"),
        ("wofür ist Marie Curie bekannt", "known_for", "Marie Curie"),
        ("wo ist Newton begraben", "resting_place", "Newton"),
        ("wie viele Kinder hatte Darwin", "children", "Darwin"),
        ("was ist die Universität von Hawking", "alma_mater", "Hawking"),
        ("wo hat Einstein studiert", "education", "Einstein"),
        ("was ist die offizielle Website von Newton", "official_website", "Newton"),
        ("was ist Darwin Diplomarbeitsthema", "thesis", "Darwin"),
        ("wie alt war Hawking als sie starb", "age_at_death", "Hawking"),
    ],
    "en": [
        ("when was Albert Einstein born", "born", "Albert Einstein"),
        ("when did Stephen Hawking die", "died", "Stephen Hawking"),
        ("what is Marie Curie known for", "known_for", "Marie Curie"),
        ("where is Newton resting place", "resting_place", "Newton"),
        ("how many children did Darwin have", "children", "Darwin"),
        ("what is Hawking alma mater", "alma_mater", "Hawking"),
        ("where did Einstein study", "education", "Einstein"),
        ("what is Newton official website", "official_website", "Newton"),
        ("what is Darwin thesis subject", "thesis", "Darwin"),
        ("how old was Hawking on his death", "age_at_death", "Hawking"),
        ("where did Turing work", "institutions", "Turing"),
        ("what field did Newton work in", "fields", "Newton"),
        ("what awards did Curie win", "notable_awards", "Curie"),
        ("what is Picasso most famous work", "notable_work", "Picasso"),
        ("what movement did Picasso belong to", "movement", "Picasso"),
        ("what is Hawking profession", "occupation", "Hawking"),
        ("who preceded Elizabeth II", "predecessor", "Elizabeth II"),
        ("who succeeded Elizabeth II", "successor", "Elizabeth II"),
        ("what religion does Einstein follow", "religion", "Einstein"),
        ("who were Hawking doctoral students", "doctoral_students", "Hawking"),
        ("when was Newton baptised", "baptised", "Newton"),
        ("who is the father of Darwin", "father", "Darwin"),
        ("who is the mother of Darwin", "mother", "Darwin"),
        # new intents
        ("what political party is Biden in", "political_party", "Biden"),
        ("how old is Elon Musk", "age", "Elon Musk"),
        ("who directed The Godfather", "director", "The Godfather"),
        ("who stars in Inception", "starring", "Inception"),
        ("when was The Godfather released", "released", "The Godfather"),
        ("what is the running time of Inception", "running_time", "Inception"),
        ("what was the budget of Avatar", "budget", "Avatar"),
        ("how much did Titanic make at the box office", "box_office", "Titanic"),
        ("when was Apple founded", "founded", "Apple"),
        ("who founded Google", "founders", "Google"),
        ("what industry is Tesla in", "industry", "Tesla"),
        ("who is the ceo of Microsoft", "ceo", "Microsoft"),
        ("who owns Twitter", "owner", "Twitter"),
        ("who invented the telephone", "inventor", "the telephone"),
        ("what is Apple revenue", "revenue", "Apple"),
        ("how tall is LeBron James", "height", "LeBron James"),
        ("what country is the Eiffel Tower in", "country", "the Eiffel Tower"),
        ("what is the elevation of Mount Everest", "elevation", "Mount Everest"),
        ("what is the length of the Amazon River", "length", "the Amazon River"),
        ("what are Shakespeare other names", "other_names", "Shakespeare"),
        ("what sport does Serena Williams play", "sport", "Serena Williams"),
        ("when was Serena Williams active", "years_active", "Serena Williams"),
        ("who coaches Real Madrid", "coached_by", "Real Madrid"),
        ("who is Einstein partner", "partner", "Einstein"),
        ("what is Obama citizenship", "citizenship", "Obama"),
        ("where is the Eiffel Tower located", "location", "the Eiffel Tower"),
        ("what license is Linux under", "license", "Linux"),
        ("who designed the Eiffel Tower", "designed_by", "the Eiffel Tower"),
        # round 2 new intents
        ("who wrote Pride and Prejudice", "author", "Pride and Prejudice"),
        ("what genre is Inception", "genre", "Inception"),
        ("what language is Don Quixote written in", "language", "Don Quixote"),
        ("what is the original title of The Odyssey", "original_title", "The Odyssey"),
        ("who published War and Peace", "publisher", "War and Peace"),
        ("who is the head coach of FC Barcelona", "head_coach", "FC Barcelona"),
        ("what stadium does Real Madrid use", "stadium", "Real Madrid"),
        ("how many championships has the Lakers won", "championships", "the Lakers"),
        ("what platforms is Minecraft on", "platforms", "Minecraft"),
        ("who developed Python", "developer", "Python"),
        ("who produced The Godfather", "produced_by", "The Godfather"),
        ("who wrote the screenplay of Inception", "written_by", "Inception"),
        ("what is Tesla nickname", "nickname", "Tesla"),
        ("what is the capital of France", "capital", "France"),
        ("what is the population of Tokyo", "population", "Tokyo"),
        ("what is the area of Canada", "area", "Canada"),
        ("where is Apple headquartered", "headquarters", "Apple"),
        ("what is Elon Musk net worth", "net_worth", "Elon Musk"),
        ("who is Einstein spouse", "spouse", "Einstein"),
        ("what is the scientific name of the gray wolf", "scientific_name", "the gray wolf"),
        ("where does the polar bear live", "habitat", "the polar bear"),
        ("what does the lion eat", "diet", "the lion"),
        ("is the tiger endangered", "conservation_status", "the tiger"),
        ("when was the Eiffel Tower opened", "opened", "the Eiffel Tower"),
        ("what record label is Taylor Swift on", "label", "Taylor Swift"),
        ("who are the members of The Beatles", "members", "The Beatles"),
        ("who composed the music of Inception", "composers", "Inception"),
        ("how many employees does Google have", "number_of_employees", "Google"),
    ],
    "eu": [
        ("noiz jaio zen Albert Einstein", "born", "Albert Einstein"),
        ("noiz hil zen Stephen Hawking", "died", "Stephen Hawking"),
        ("zergatik da ezagun Marie Curie", "known_for", "Marie Curie"),
        ("non dago lurperatuta Newton", "resting_place", "Newton"),
        ("zenbat haur izan ditu Darwin", "children", "Darwin"),
        ("zer da Hawking alma mater", "alma_mater", "Hawking"),
        ("non ikasi zuen Einstein", "education", "Einstein"),
        ("zein da Newton webgune ofiziala", "official_website", "Newton"),
        ("zein da Darwin tesiaren gaia", "thesis", "Darwin"),
        ("zenbat urte zituen Hawking hil zenean", "age_at_death", "Hawking"),
    ],
    "fr": [
        ("quand Albert Einstein est-il né", "born", "Albert Einstein"),
        ("quand Stephen Hawking est-il mort", "died", "Stephen Hawking"),
        ("pour quoi Marie Curie est-il connu", "known_for", "Marie Curie"),
        ("où est enterré Newton", "resting_place", "Newton"),
        ("combien d'enfants a Darwin", "children", "Darwin"),
        ("dans quelle université Hawking a-t-il étudié", "alma_mater", "Hawking"),
        ("où Einstein a fait ses études", "education", "Einstein"),
        ("quel est le site officiel de Newton", "official_website", "Newton"),
        ("quel est le sujet de thèse de Darwin", "thesis", "Darwin"),
        ("quel âge avait Hawking lorsqu'il est mort", "age_at_death", "Hawking"),
    ],
    "gl": [
        ("cando finou Stephen Hawking", "died", "Stephen Hawking"),
        ("por que se coñece Marie Curie", "known_for", "Marie Curie"),
        ("onde está enterrado Newton", "resting_place", "Newton"),
        ("cantos fillos tivo Darwin", "children", "Darwin"),
        ("onde se formou Einstein", "education", "Einstein"),
        ("cal é o sitio web oficial de Newton", "official_website", "Newton"),
        ("que é o tema da tese Darwin", "thesis", "Darwin"),
        ("cantos anos tiña Hawking cando morreu", "age_at_death", "Hawking"),
    ],
    "it": [
        ("quando è nata Albert Einstein", "born", "Albert Einstein"),
        ("qual è il sito ufficiale di Newton", "official_website", "Newton"),
    ],
    "pt": [
        ("quando foi a morte de Stephen Hawking", "died", "Stephen Hawking"),
        ("pelo que é Marie Curie conhecida", "known_for", "Marie Curie"),
        ("onde se formou Einstein", "education", "Einstein"),
        ("quantos filhos tinha Darwin", "children", "Darwin"),
        ("quantos anos tinha Hawking quando morreu", "age_at_death", "Hawking"),
    ],
}


class TestIntentParsingAllLanguages(unittest.TestCase):
    """Integration tests: load real locale files and verify intent matching for all languages."""

    @classmethod
    def setUpClass(cls):
        cls.engine = DuckDuckGoRetrievalEngine.__new__(DuckDuckGoRetrievalEngine)
        cls.engine.config = {}
        cls.engine._kword_extractors = {}
        cls.engine._intent_matchers = {}
        cls.engine._load_intents()

    def _assert_matches(self, lang: str, utterance: str, expected_intent: str, expected_keyword: str):
        intent, kw = self.engine._match_infobox_intent(utterance, lang)
        self.assertEqual(
            intent, expected_intent,
            f"[{lang}] {utterance!r} → expected intent {expected_intent!r}, got {intent!r}",
        )
        self.assertEqual(
            kw.lower(), expected_keyword.lower(),
            f"[{lang}] {utterance!r} → expected keyword {expected_keyword!r}, got {kw!r}",
        )


def _make_intent_test(lang, utterance, intent, keyword):
    def test(self):
        self._assert_matches(lang, utterance, intent, keyword)
    test.__name__ = f"test_{lang}_{intent}_{keyword.lower().replace(' ', '_')}"
    return test

for _lang, _cases in _INTENT_CASES.items():
    for _utterance, _intent, _keyword in _cases:
        _name = f"test_{_lang}_{_intent}_{_keyword.lower().replace(' ', '_')}"
        setattr(TestIntentParsingAllLanguages, _name, _make_intent_test(_lang, _utterance, _intent, _keyword))


if __name__ == "__main__":
    unittest.main()

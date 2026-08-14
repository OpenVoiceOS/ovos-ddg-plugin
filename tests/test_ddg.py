"""
Unit tests for ovos-ddg-plugin.

All network calls and OVOS config are mocked — no API key or running daemon required.
"""
import datetime
import unittest
from unittest.mock import MagicMock, patch, call

from ovos_spec_tools import normalize_for_match

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
        self.engine._match_infobox_intents = MagicMock(return_value=[])
        self.engine._search = MagicMock(return_value={"AbstractText": "Newton was a physicist."})
        with patch("ovos_ddg_plugin.Configuration", return_value={}):
            results = self.engine.query("Newton", lang="en-us")
        self.assertIsInstance(results, list)
        self.assertTrue(all(isinstance(r, tuple) and len(r) == 2 for r in results))

    def test_respects_k_limit(self):
        self.engine._match_infobox_intents = MagicMock(return_value=[])
        self.engine._search = MagicMock(return_value={"AbstractText": "A. B. C. D. E. F."})
        with patch("ovos_ddg_plugin.Configuration", return_value={}):
            results = self.engine.query("Newton", lang="en-us", k=2)
        self.assertLessEqual(len(results), 2)

    def test_returns_empty_list_when_no_data(self):
        self.engine._match_infobox_intents = MagicMock(return_value=[])
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
        self.engine._match_infobox_intents = MagicMock(return_value=[("alma_mater", "Hawking", "person")])
        self.engine.get_infobox = MagicMock(return_value=({"education": "University of Cambridge"}, []))
        with patch("ovos_ddg_plugin.Configuration", return_value={}):
            results = self.engine.query("what is Hawking alma mater", lang="en-us")
        self.assertEqual(results, [("University of Cambridge", 0.9)])

    def test_field_alias_resting_place_resolves_via_burial(self):
        # resting_place intent → DDG sometimes stores as "burial"
        self.engine._match_infobox_intents = MagicMock(return_value=[("resting_place", "Queen Elizabeth II", "person")])
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
        self.engine._match_infobox_intents = MagicMock(return_value=[("born", "Hawking", "person")])
        self.engine.get_infobox = MagicMock(return_value=({"born": "8 January 1942"}, []))
        with patch("ovos_ddg_plugin.Configuration", return_value={}):
            results = self.engine.query("when was Hawking born", lang="en-us")
        self.assertEqual(results, [("8 January 1942", 0.9)])

    def test_infobox_miss_falls_through_to_abstract(self):
        self.engine._match_infobox_intents = MagicMock(return_value=[("born", "Hawking", "person")])
        self.engine.get_infobox = MagicMock(return_value=({}, []))  # no "born" field
        self.engine._search = MagicMock(return_value={"AbstractText": "Hawking was a physicist."})
        with patch("ovos_ddg_plugin.Configuration", return_value={}):
            results = self.engine.query("when was Hawking born", lang="en-us")
        self.assertTrue(len(results) > 0)
        self.assertNotEqual(results[0][0], "8 January 1942")

    def test_abstract_scores_decrease_with_sentence_position(self):
        self.engine._match_infobox_intents = MagicMock(return_value=[])
        self.engine._search = MagicMock(
            return_value={"AbstractText": "Newton was a physicist. He invented calculus. He studied at Cambridge."}
        )
        with patch("ovos_ddg_plugin.Configuration", return_value={}):
            results = self.engine.query("Newton", lang="en-us", k=3)
        scores = [s for _, s in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_empty_sentences_filtered_out(self):
        self.engine._match_infobox_intents = MagicMock(return_value=[])
        with patch("ovos_ddg_plugin.sentence_tokenize", return_value=["Real sentence.", "", "  "]), \
             patch("ovos_ddg_plugin.Configuration", return_value={}):
            self.engine._search = MagicMock(return_value={"AbstractText": "anything"})
            results = self.engine.query("Newton", lang="en-us", k=5)
        self.assertTrue(all(text.strip() for text, _ in results))

    def test_lang_defaults_from_config_when_not_passed(self):
        self.engine._match_infobox_intents = MagicMock(return_value=[])
        self.engine._search = MagicMock(return_value={})
        with patch("ovos_ddg_plugin.Configuration", return_value={"lang": "pt-pt"}):
            self.engine.query("x")
        self.engine._match_infobox_intents.assert_called_once_with("x", "pt-pt")

    def test_ambiguous_intent_resolved_by_infobox(self):
        # "wie lang ist Amazon" ties between length({place}) and running_time({movie}).
        # The infobox for "Amazon" has "length" but not "running_time" → length wins.
        self.engine._match_infobox_intents = MagicMock(return_value=[
            ("running_time", "Amazon", "movie"),
            ("length", "Amazon", "place"),
        ])
        self.engine.get_infobox = MagicMock(return_value=({"length": "6,400 km"}, []))
        with patch("ovos_ddg_plugin.Configuration", return_value={}):
            results = self.engine.query("wie lang ist Amazon", lang="de-DE")
        self.assertEqual(results, [("6,400 km", 0.9)])

    def test_ambiguous_intent_first_candidate_wins_when_both_missing(self):
        # When neither candidate has an infobox value, fall through to abstract.
        self.engine._match_infobox_intents = MagicMock(return_value=[
            ("running_time", "Amazon", "movie"),
            ("length", "Amazon", "place"),
        ])
        self.engine.get_infobox = MagicMock(return_value=({}, []))
        self.engine._search = MagicMock(return_value={"AbstractText": "Amazon is a river."})
        with patch("ovos_ddg_plugin.Configuration", return_value={}):
            results = self.engine.query("wie lang ist Amazon", lang="de-DE")
        self.assertIn("Amazon", results[0][0])


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
        self.assertEqual(normalize_for_match(kw), normalize_for_match("Einstein"))

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
        self.assertEqual(normalize_for_match(kw), normalize_for_match("Newton"))

    def test_separate_matchers_per_language(self):
        self.engine._register_intent("born", ["when was {keyword} born"], "en")
        self.engine._register_intent("born", ["quando nasceu {keyword}"], "pt")
        intent_en, kw_en = self.engine._match_infobox_intent("when was Newton born", "en")
        intent_pt, kw_pt = self.engine._match_infobox_intent("quando nasceu Newton", "pt")
        self.assertEqual(intent_en, "born")
        self.assertEqual(intent_pt, "born")
        self.assertEqual(normalize_for_match(kw_en), normalize_for_match("Newton"))
        self.assertEqual(normalize_for_match(kw_pt), normalize_for_match("Newton"))


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

    def test_constructs_the_way_the_persona_server_loader_does(self):
        # ovos_persona_server.tools._load_toolboxes calls cls(config=cfg, bus=bus).
        # A plugin whose __init__ does not accept bus raises a TypeError there
        # and the loader only logs a warning, so the toolbox silently vanishes.
        with patch("ovos_ddg_plugin.DuckDuckGoRetrievalEngine"):
            cfg = {"some": "config"}
            tb = DuckDuckGoToolbox(config=cfg, bus=None)
        self.assertEqual(tb.config, cfg)
        self.assertTrue(tb.tools)

    def test_constructor_forwards_bus_so_bind_actually_runs(self):
        # Passing bus=None (as the loader does when it has no bus of its own)
        # must not crash, and passing a real bus must reach ToolBox.bind(),
        # which is only possible now that __init__ forwards it to super().
        calls = []

        class FakeBus:
            def on(self, topic, handler):
                calls.append(topic)

        with patch("ovos_ddg_plugin.DuckDuckGoRetrievalEngine"):
            tb = DuckDuckGoToolbox(config={}, bus=FakeBus())
        self.assertEqual(sorted(calls), sorted([
            "ovos.persona.tools.discover",
            f"ovos.persona.tools.{tb.toolbox_id}.call",
        ]))

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
# Every .intent file in every locale has at least one test case here.
# Ambiguous utterances (tied intents) are avoided; specific forms are used.
_INTENT_CASES = {
    "ca": [
        # person / biography
        ("quants anys té Einstein", "age", "Einstein"),
        ("quants anys tenia Darwin en morir", "age_at_death", "Darwin"),
        ("quina és l'alma mater de Hawking", "alma_mater", "Hawking"),
        ("quan va ser batejat Newton", "baptised", "Newton"),
        ("quan va néixer Albert Einstein", "born", "Albert Einstein"),
        ("quants fills té Darwin", "children", "Darwin"),
        ("quina és la ciutadania de Obama", "citizenship", "Obama"),
        ("qui entrena Real Madrid", "coached_by", "Real Madrid"),
        ("quan va morir Stephen Hawking", "died", "Stephen Hawking"),
        ("qui eren els doctorands de Hawking", "doctoral_students", "Hawking"),
        ("on va estudiar Einstein", "education", "Einstein"),
        ("qui és el pare de Darwin", "father", "Darwin"),
        ("quin és l'àmbit d'expertesa de Newton", "fields", "Newton"),
        ("quant mesura LeBron James", "height", "LeBron James"),
        ("on va treballar Turing", "institutions", "Turing"),
        ("per a què es coneix Marie Curie", "known_for", "Marie Curie"),
        ("qui és la mare de Darwin", "mother", "Darwin"),
        ("a quin moviment artístic pertanyia Picasso", "movement", "Picasso"),
        ("quin és el patrimoni net de Elon Musk", "net_worth", "Elon Musk"),
        ("quins premis va guanyar Curie", "notable_awards", "Curie"),
        ("quina és l'obra més famosa de Picasso", "notable_work", "Picasso"),
        ("quina és la professió de Hawking", "occupation", "Hawking"),
        ("quin és el lloc web oficial de Newton", "official_website", "Newton"),
        ("qui és la parella de Einstein", "partner", "Einstein"),
        ("a quin partit polític pertany Biden", "political_party", "Biden"),
        ("qui va precedir Elizabeth II", "predecessor", "Elizabeth II"),
        ("quina religió practica Einstein", "religion", "Einstein"),
        ("on està enterrat Newton", "resting_place", "Newton"),
        ("qui és el cònjuge de Einstein", "spouse", "Einstein"),
        ("quin esport practica Serena Williams", "sport", "Serena Williams"),
        ("qui va succeir Elizabeth II", "successor", "Elizabeth II"),
        ("quin és el tema de la tesi de Darwin", "thesis", "Darwin"),
        ("quan va estar actiu Serena Williams", "years_active", "Serena Williams"),
        # movie
        ("quant va recaptar Inception", "box_office", "Inception"),
        ("quin era el pressupost de Inception", "budget", "Inception"),
        ("qui va compondre la música de Inception", "composers", "Inception"),
        ("qui va dirigir Inception", "director", "Inception"),
        ("quin gènere és Inception", "genre", "Inception"),
        ("en quina llengua és escrit Don Quixote", "language", "Don Quixote"),
        ("quin és el títol original de The Odyssey", "original_title", "The Odyssey"),
        ("qui va produir Inception", "produced_by", "Inception"),
        ("quan es va estrenar Inception", "released", "Inception"),
        ("quant dura Inception", "running_time", "Inception"),
        ("qui actua a Inception", "starring", "Inception"),
        ("qui va escriure el guió de Inception", "written_by", "Inception"),
        # book
        ("qui va escriure Don Quixote", "author", "Don Quixote"),
        ("qui va publicar War and Peace", "publisher", "War and Peace"),
        # company / organization
        ("qui és el ceo de Apple", "ceo", "Apple"),
        ("quan va ser fundat Apple", "founded", "Apple"),
        ("qui va fundar Google", "founders", "Google"),
        ("en quin sector és Tesla", "industry", "Tesla"),
        ("on té la seu Apple", "headquarters", "Apple"),
        ("quants empleats té Google", "number_of_employees", "Google"),
        ("quins són els ingressos de Apple", "revenue", "Apple"),
        # sports team
        ("quants campionats ha guanyat Real Madrid", "championships", "Real Madrid"),
        ("qui és l entrenador de FC Barcelona", "head_coach", "FC Barcelona"),
        ("en quin estadi juga Real Madrid", "stadium", "Real Madrid"),
        # geography
        ("quina és la capital de França", "capital", "França"),
        ("a quin país és la Torre Eiffel", "country", "la Torre Eiffel"),
        ("quina és l altitud de Everest", "elevation", "Everest"),
        ("quina és la longitud de Amazon", "length", "Amazon"),
        ("on es troba la Torre Eiffel", "location", "la Torre Eiffel"),
        ("quina és l àrea de Canadà", "area", "Canadà"),
        ("quina és la població de Tòquio", "population", "Tòquio"),
        # animal
        ("tigre estat de conservació", "conservation_status", "tigre"),
        ("que menja el lleó", "diet", "el lleó"),
        ("on viu ós polar", "habitat", "ós polar"),
        ("quin és el nom científic de llop gris", "scientific_name", "llop gris"),
        # music
        ("a quin segell discogràfic pertany Taylor Swift", "label", "Taylor Swift"),
        ("qui són els membres de The Beatles", "members", "The Beatles"),
        # technology
        ("qui va desenvolupar Linux", "developer", "Linux"),
        ("sota quina llicència és Linux", "license", "Linux"),
        ("en quines plataformes és Minecraft", "platforms", "Minecraft"),
        # place / general
        ("qui va dissenyar la Torre Eiffel", "designed_by", "la Torre Eiffel"),
        ("quin és el sobrenom de Tesla", "nickname", "Tesla"),
        ("quan va ser inaugurat la Torre Eiffel", "opened", "la Torre Eiffel"),
        ("quins altres noms té Shakespeare", "other_names", "Shakespeare"),
        ("qui és el propietari de Twitter", "owner", "Twitter"),
        ("qui va inventar el telèfon", "inventor", "el telèfon"),
    ],
    "da": [
        # person / biography
        ("hvor gammel er Einstein", "age", "Einstein"),
        ("hvor gammel var Hawking ved hans død", "age_at_death", "Hawking"),
        ("hvad er Hawking alma mater", "alma_mater", "Hawking"),
        ("hvornår blev Newton døbt", "baptised", "Newton"),
        ("hvornår blev Albert Einstein født", "born", "Albert Einstein"),
        ("hvor mange børn har Darwin har", "children", "Darwin"),
        ("hvad er Obama statsborgerskab", "citizenship", "Obama"),
        ("hvem træner Real Madrid", "coached_by", "Real Madrid"),
        ("hvornår døde Stephen Hawking", "died", "Stephen Hawking"),
        ("hvem vejledte Hawking i ph.d.-arbejde", "doctoral_students", "Hawking"),
        ("hvor studerede Einstein", "education", "Einstein"),
        ("hvem er far til Darwin", "father", "Darwin"),
        ("hvad er Newton ekspertiseområde", "fields", "Newton"),
        ("hvor høj er LeBron James", "height", "LeBron James"),
        ("hvor arbejdede Turing", "institutions", "Turing"),
        ("hvad er Marie Curie berømt for", "known_for", "Marie Curie"),
        ("hvem er mor til Darwin", "mother", "Darwin"),
        ("hvilken kunstnerisk bevægelse var Picasso tilknyttet", "movement", "Picasso"),
        ("hvad er Elon Musk formue", "net_worth", "Elon Musk"),
        ("hvilke priser vandt Curie", "notable_awards", "Curie"),
        ("hvad skabte Picasso", "notable_work", "Picasso"),
        ("hvad lavede Hawking", "occupation", "Hawking"),
        ("hvad er Newton officielle hjemmeside", "official_website", "Newton"),
        ("hvem er Einstein partner", "partner", "Einstein"),
        ("hvilket politisk parti tilhører Biden", "political_party", "Biden"),
        ("hvem gik forud for Elizabeth II", "predecessor", "Elizabeth II"),
        ("hvilken religion praktiserer Einstein", "religion", "Einstein"),
        ("hvor ligger Newton begravet", "resting_place", "Newton"),
        ("hvem er Einstein ægtefælle", "spouse", "Einstein"),
        ("hvilken sport dyrker Serena Williams", "sport", "Serena Williams"),
        ("hvem tog over fra Elizabeth II", "successor", "Elizabeth II"),
        ("hvad er Darwin specialeemne", "thesis", "Darwin"),
        ("hvornår var Serena Williams aktiv", "years_active", "Serena Williams"),
        # movie
        ("hvor meget tjente Inception", "box_office", "Inception"),
        ("hvad var budgettet for Inception", "budget", "Inception"),
        ("hvem komponerede musikken til Inception", "composers", "Inception"),
        ("hvem instruerede Inception", "director", "Inception"),
        ("hvilket genre er Inception", "genre", "Inception"),
        ("hvilket sprog er Don Quixote skrevet på", "language", "Don Quixote"),
        ("hvad er originaltitlen på The Odyssey", "original_title", "The Odyssey"),
        ("hvem producerede Inception", "produced_by", "Inception"),
        ("hvornår udkom Inception", "released", "Inception"),
        ("hvor lang tid varer Inception", "running_time", "Inception"),
        ("hvem spiller i Inception", "starring", "Inception"),
        ("hvem skrev manuskriptet til Inception", "written_by", "Inception"),
        # book
        ("hvem har skrevet Don Quixote", "author", "Don Quixote"),
        ("hvem udgav War and Peace", "publisher", "War and Peace"),
        # company / organization
        ("hvem er ceo for Apple", "ceo", "Apple"),
        ("hvornår blev Apple grundlagt", "founded", "Apple"),
        ("hvem grundlagde Google", "founders", "Google"),
        ("hvilken branche er Tesla i", "industry", "Tesla"),
        ("hvor er Apple hovedkontor", "headquarters", "Apple"),
        ("hvor mange ansatte har Google", "number_of_employees", "Google"),
        ("hvad er Apple omsætning", "revenue", "Apple"),
        # sports team
        ("hvor mange mesterskaber har Real Madrid vundet", "championships", "Real Madrid"),
        ("hvem er cheftræner for FC Barcelona", "head_coach", "FC Barcelona"),
        ("hvilken stadion bruger Real Madrid", "stadium", "Real Madrid"),
        # geography
        ("hvad er hovedstaden i Frankrig", "capital", "Frankrig"),
        ("i hvilket land er Eiffeltårnet", "country", "Eiffeltårnet"),
        ("hvad er højden på Everest", "elevation", "Everest"),
        ("hvor lang er Amazon", "length", "Amazon"),
        ("hvor ligger Eiffeltårnet", "location", "Eiffeltårnet"),
        ("hvad er arealet af Canada", "area", "Canada"),
        ("hvad er befolkningstallet i Tokyo", "population", "Tokyo"),
        # animal
        ("tiger bevaringsstatus", "conservation_status", "tiger"),
        ("hvad spiser løven", "diet", "løven"),
        ("hvor lever isbjørnen", "habitat", "isbjørnen"),
        ("hvad er det videnskabelige navn for gråulven", "scientific_name", "gråulven"),
        # music
        ("hvilket pladeselskab er Taylor Swift tilknyttet", "label", "Taylor Swift"),
        ("hvem er medlemmerne i The Beatles", "members", "The Beatles"),
        # technology
        ("hvem udviklede Linux", "developer", "Linux"),
        ("under hvilken licens er Linux", "license", "Linux"),
        ("hvilke platforme kører Minecraft på", "platforms", "Minecraft"),
        # place / general
        ("hvem designede Eiffeltårnet", "designed_by", "Eiffeltårnet"),
        ("hvad er Tesla kaldenavn", "nickname", "Tesla"),
        ("hvornår blev Eiffeltårnet åbnet", "opened", "Eiffeltårnet"),
        ("hvad er Shakespeare andre navne", "other_names", "Shakespeare"),
        ("hvem ejer Twitter", "owner", "Twitter"),
        ("hvem opfandt telefonen", "inventor", "telefonen"),
    ],
    "de": [
        # person / biography
        ("wie alt ist Einstein", "age", "Einstein"),
        ("wie alt war Hawking als sie starb", "age_at_death", "Hawking"),
        ("was ist die Universität von Hawking", "alma_mater", "Hawking"),
        ("wann wurde Newton getauft", "baptised", "Newton"),
        ("wann wurde Albert Einstein geboren", "born", "Albert Einstein"),
        ("wie viele Kinder hatte Darwin", "children", "Darwin"),
        ("was ist Obama Staatsbürgerschaft", "citizenship", "Obama"),
        ("wer trainiert Real Madrid", "coached_by", "Real Madrid"),
        ("wann ist Stephen Hawking gestorben", "died", "Stephen Hawking"),
        ("Hawking Doktoranden und Schüler", "doctoral_students", "Hawking"),
        ("wo hat Einstein studiert", "education", "Einstein"),
        ("wer ist der Vater von Darwin", "father", "Darwin"),
        ("was ist das Fachgebiet von Newton", "fields", "Newton"),
        ("wie groß ist LeBron James", "height", "LeBron James"),
        ("wo hat Turing gearbeitet", "institutions", "Turing"),
        ("wofür ist Marie Curie bekannt", "known_for", "Marie Curie"),
        ("wer ist die Mutter von Darwin", "mother", "Darwin"),
        ("welcher künstlerischen Bewegung gehörte Picasso an", "movement", "Picasso"),
        ("was ist Elon Musk Nettovermögen", "net_worth", "Elon Musk"),
        ("welche Auszeichnungen hat Curie erhalten", "notable_awards", "Curie"),
        ("Picasso bekanntestes Werk", "notable_work", "Picasso"),
        ("was ist der Beruf von Hawking", "occupation", "Hawking"),
        ("was ist die offizielle Website von Newton", "official_website", "Newton"),
        ("wer ist Einstein Partner", "partner", "Einstein"),
        ("welcher Partei gehört Biden an", "political_party", "Biden"),
        ("wer war der Vorgänger von Elizabeth II", "predecessor", "Elizabeth II"),
        ("welcher Religion gehört Einstein an", "religion", "Einstein"),
        ("wo ist Newton begraben", "resting_place", "Newton"),
        ("wer ist Einstein Ehepartner", "spouse", "Einstein"),
        ("welche Sportart betreibt Serena Williams", "sport", "Serena Williams"),
        ("wer war der Nachfolger von Elizabeth II", "successor", "Elizabeth II"),
        ("was ist Darwin Diplomarbeitsthema", "thesis", "Darwin"),
        ("wann war Serena Williams aktiv", "years_active", "Serena Williams"),
        # movie
        ("wie viel hat Inception eingespielt", "box_office", "Inception"),
        ("was war das Budget von Inception", "budget", "Inception"),
        ("wer hat die Musik von Inception komponiert", "composers", "Inception"),
        ("wer hat Inception gedreht", "director", "Inception"),
        ("welches Genre ist Inception", "genre", "Inception"),
        ("in welcher Sprache ist Don Quixote geschrieben", "language", "Don Quixote"),
        ("was ist der Originaltitel von The Odyssey", "original_title", "The Odyssey"),
        ("wer hat Inception produziert", "produced_by", "Inception"),
        ("wann wurde Inception veröffentlicht", "released", "Inception"),
        ("wie lange dauert Inception", "running_time", "Inception"),
        ("wer spielt in Inception", "starring", "Inception"),
        ("wer hat das drehbuch zu Inception geschrieben", "written_by", "Inception"),
        # book
        ("wer hat Don Quixote geschrieben", "author", "Don Quixote"),
        ("wer hat War and Peace veröffentlicht", "publisher", "War and Peace"),
        # company / organization
        ("wer ist der CEO von Apple", "ceo", "Apple"),
        ("wann wurde Apple gegründet", "founded", "Apple"),
        ("wer hat Google gegründet", "founders", "Google"),
        ("in welcher Branche ist Tesla", "industry", "Tesla"),
        ("wo hat Apple seinen Hauptsitz", "headquarters", "Apple"),
        ("wie viele Mitarbeiter hat Google", "number_of_employees", "Google"),
        ("was ist Apple Umsatz", "revenue", "Apple"),
        # sports team
        ("wie viele Meisterschaften hat Real Madrid gewonnen", "championships", "Real Madrid"),
        ("wer ist der Trainer von FC Barcelona", "head_coach", "FC Barcelona"),
        ("welches Stadion hat Real Madrid", "stadium", "Real Madrid"),
        # geography
        ("was ist die Hauptstadt von Frankreich", "capital", "Frankreich"),
        ("in welchem Land ist der Eiffelturm", "country", "der Eiffelturm"),
        ("was ist die Höhe von Everest", "elevation", "Everest"),
        ("der Rhein länge", "length", "der Rhein"),
        ("wo befindet sich der Eiffelturm", "location", "der Eiffelturm"),
        ("Kanada fläche", "area", "Kanada"),
        ("wie viele Einwohner hat Tokio", "population", "Tokio"),
        # animal
        ("Tiger Schutzstatus", "conservation_status", "Tiger"),
        ("was frisst der Löwe", "diet", "der Löwe"),
        ("wo lebt der Eisbär", "habitat", "der Eisbär"),
        ("was ist der wissenschaftliche Name von Grauwolf", "scientific_name", "Grauwolf"),
        # music
        ("bei welchem Label ist Taylor Swift unter Vertrag", "label", "Taylor Swift"),
        ("wer sind die Mitglieder von The Beatles", "members", "The Beatles"),
        # technology
        ("wer hat Linux entwickelt", "developer", "Linux"),
        ("unter welcher Lizenz steht Linux", "license", "Linux"),
        ("auf welchen Plattformen läuft Minecraft", "platforms", "Minecraft"),
        # place / general
        ("wer hat den Eiffelturm entworfen", "designed_by", "den Eiffelturm"),
        ("was ist Tesla Spitzname", "nickname", "Tesla"),
        ("wann wurde der Eiffelturm eröffnet", "opened", "der Eiffelturm"),
        ("was sind andere Namen für Shakespeare", "other_names", "Shakespeare"),
        ("wem gehört Twitter", "owner", "Twitter"),
        ("wer hat das Telefon erfunden", "inventor", "das Telefon"),
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
    "es": [
        # person / biography
        ("cuántos años tiene Einstein", "age", "Einstein"),
        ("cuántos años tenía Hawking cuando murió", "age_at_death", "Hawking"),
        ("cuál es el alma mater de Hawking", "alma_mater", "Hawking"),
        ("cuándo fue bautizado Newton", "baptised", "Newton"),
        ("cuándo nació Albert Einstein", "born", "Albert Einstein"),
        ("cuántos hijos tuvo Darwin", "children", "Darwin"),
        ("cuál es la ciudadanía de Obama", "citizenship", "Obama"),
        ("quién es el entrenador de Real Madrid", "coached_by", "Real Madrid"),
        ("cuándo murió Stephen Hawking", "died", "Stephen Hawking"),
        ("quiénes fueron los doctorandos de Hawking", "doctoral_students", "Hawking"),
        ("dónde se educó Einstein", "education", "Einstein"),
        ("quién es el padre de Darwin", "father", "Darwin"),
        ("cuál es el campo de especialización de Newton", "fields", "Newton"),
        ("cuánto mide LeBron James", "height", "LeBron James"),
        ("dónde trabajó Turing", "institutions", "Turing"),
        ("por qué se conoce a Marie Curie", "known_for", "Marie Curie"),
        ("quién es la madre de Darwin", "mother", "Darwin"),
        ("a qué movimiento artístico pertenece Picasso", "movement", "Picasso"),
        ("cuál es el patrimonio neto de Elon Musk", "net_worth", "Elon Musk"),
        ("qué premios ganó Curie", "notable_awards", "Curie"),
        ("cuál es la obra más famosa de Picasso", "notable_work", "Picasso"),
        ("cuál es la profesión de Hawking", "occupation", "Hawking"),
        ("cuál es el sitio web oficial de Newton", "official_website", "Newton"),
        ("quién es la pareja de Einstein", "partner", "Einstein"),
        ("a qué partido político pertenece Biden", "political_party", "Biden"),
        ("quién precedió a Elizabeth II", "predecessor", "Elizabeth II"),
        ("qué religión practica Einstein", "religion", "Einstein"),
        ("dónde está enterrado Newton", "resting_place", "Newton"),
        ("quién es el cónyuge de Einstein", "spouse", "Einstein"),
        ("qué deporte practica Serena Williams", "sport", "Serena Williams"),
        ("quién sucedió a Elizabeth II", "successor", "Elizabeth II"),
        ("cuál es el tema de la tesis de Darwin", "thesis", "Darwin"),
        ("cuándo estuvo activo Serena Williams", "years_active", "Serena Williams"),
        # movie
        ("cuánto recaudó Inception", "box_office", "Inception"),
        ("cuál fue el presupuesto de Inception", "budget", "Inception"),
        ("quién compuso la música de Inception", "composers", "Inception"),
        ("quién dirigió Inception", "director", "Inception"),
        ("qué género es Inception", "genre", "Inception"),
        ("en qué idioma está escrito Don Quijote", "language", "Don Quijote"),
        ("cuál es el título original de The Odyssey", "original_title", "The Odyssey"),
        ("quién produjo Inception", "produced_by", "Inception"),
        ("cuándo se estrenó Inception", "released", "Inception"),
        ("cuánto dura Inception", "running_time", "Inception"),
        ("quiénes actúan en Inception", "starring", "Inception"),
        ("quién escribió el guión de Inception", "written_by", "Inception"),
        # book
        ("quién escribió Don Quijote", "author", "Don Quijote"),
        ("quién publicó War and Peace", "publisher", "War and Peace"),
        # company / organization
        ("quién es el ceo de Apple", "ceo", "Apple"),
        ("cuándo fue fundado Apple", "founded", "Apple"),
        ("quién fundó Google", "founders", "Google"),
        ("en qué industria está Tesla", "industry", "Tesla"),
        ("dónde está la sede de Apple", "headquarters", "Apple"),
        ("cuántos empleados tiene Google", "number_of_employees", "Google"),
        ("cuáles son los ingresos de Apple", "revenue", "Apple"),
        # sports team
        ("cuántos campeonatos ha ganado Real Madrid", "championships", "Real Madrid"),
        ("quién es el director técnico de FC Barcelona", "head_coach", "FC Barcelona"),
        ("en qué estadio juega Real Madrid", "stadium", "Real Madrid"),
        # geography
        ("cuál es la capital de Francia", "capital", "Francia"),
        ("en qué país está la Torre Eiffel", "country", "la Torre Eiffel"),
        ("cuál es la altitud de Everest", "elevation", "Everest"),
        ("cuál es la longitud de Amazon", "length", "Amazon"),
        ("dónde se encuentra la Torre Eiffel", "location", "la Torre Eiffel"),
        ("cuál es el área de Canadá", "area", "Canadá"),
        ("cuál es la población de Tokio", "population", "Tokio"),
        # animal
        ("tigre estado de conservación", "conservation_status", "tigre"),
        ("qué come el león", "diet", "el león"),
        ("dónde vive el oso polar", "habitat", "el oso polar"),
        ("cuál es el nombre científico de lobo gris", "scientific_name", "lobo gris"),
        # music
        ("a qué sello discográfico pertenece Taylor Swift", "label", "Taylor Swift"),
        ("quiénes son los integrantes de The Beatles", "members", "The Beatles"),
        # technology
        ("quién desarrolló Linux", "developer", "Linux"),
        ("bajo qué licencia está Linux", "license", "Linux"),
        ("en qué plataformas está Minecraft", "platforms", "Minecraft"),
        # place / general
        ("quién diseñó la Torre Eiffel", "designed_by", "la Torre Eiffel"),
        ("cuál es el apodo de Tesla", "nickname", "Tesla"),
        ("cuándo fue inaugurado la Torre Eiffel", "opened", "la Torre Eiffel"),
        ("cuáles son otros nombres de Shakespeare", "other_names", "Shakespeare"),
        ("quién es el dueño de Twitter", "owner", "Twitter"),
        ("quién inventó el teléfono", "inventor", "el teléfono"),
    ],
    "eu": [
        # person / biography
        ("zenbat urte ditu Einstein", "age", "Einstein"),
        ("zenbat urte zituen Hawking hil zenean", "age_at_death", "Hawking"),
        ("zer da Hawking alma mater", "alma_mater", "Hawking"),
        ("noiz bataiatu zuten Newton", "baptised", "Newton"),
        ("noiz jaio zen Albert Einstein", "born", "Albert Einstein"),
        ("zenbat haur izan ditu Darwin", "children", "Darwin"),
        ("zein da Obama herritartasuna", "citizenship", "Obama"),
        ("nork entrenatzen du Real Madrid", "coached_by", "Real Madrid"),
        ("noiz hil zen Stephen Hawking", "died", "Stephen Hawking"),
        ("Hawking doktore-ikasleak", "doctoral_students", "Hawking"),
        ("non ikasi zuen Einstein", "education", "Einstein"),
        ("Darwin aita", "father", "Darwin"),
        ("zein da Einstein espezializazio-arloa", "fields", "Einstein"),
        ("zenbat neuritzen du LeBron James", "height", "LeBron James"),
        ("non lan egin zuen Turing", "institutions", "Turing"),
        ("zergatik da ezagun Marie Curie", "known_for", "Marie Curie"),
        ("Darwin ama", "mother", "Darwin"),
        ("Picasso arte mugimendua", "movement", "Picasso"),
        ("zein da Elon Musk ondare garbia", "net_worth", "Elon Musk"),
        ("zein sari irabazi zituen Curie", "notable_awards", "Curie"),
        ("zer sortu zuen Picasso", "notable_work", "Picasso"),
        ("Hawking lanbidea", "occupation", "Hawking"),
        ("zein da Newton webgune ofiziala", "official_website", "Newton"),
        ("nor da Einstein bikotekidea", "partner", "Einstein"),
        ("zein alderdi politikotakoa da Biden", "political_party", "Biden"),
        ("nor etorri zen Elizabeth II baino lehen", "predecessor", "Elizabeth II"),
        ("Einstein erlijioa", "religion", "Einstein"),
        ("non dago lurperatuta Newton", "resting_place", "Newton"),
        ("nor da Einstein ezkontidea", "spouse", "Einstein"),
        ("zein kirol egiten du Serena Williams", "sport", "Serena Williams"),
        ("nork ordezkatu zuen Elizabeth II", "successor", "Elizabeth II"),
        ("zein da Darwin tesiaren gaia", "thesis", "Darwin"),
        ("noiz egon zen Serena Williams aktibo", "years_active", "Serena Williams"),
        # movie
        ("zenbat irabazi zuen Inception", "box_office", "Inception"),
        ("zein zen Inception aurrekontua", "budget", "Inception"),
        ("nork konposatu zuen Inception musika", "composers", "Inception"),
        ("nork zuzendu zuen Inception", "director", "Inception"),
        ("zein genero da Inception", "genre", "Inception"),
        ("zein hizkuntzatan idatzita dago Don Quixote", "language", "Don Quixote"),
        ("zein da The Odyssey jatorrizko izenburua", "original_title", "The Odyssey"),
        ("nork ekoiztu zuen Inception", "produced_by", "Inception"),
        ("noiz argitaratu zen Inception", "released", "Inception"),
        ("zenbat irauten du Inception", "running_time", "Inception"),
        ("nork jokatzen du Inception filmean", "starring", "Inception"),
        ("Inception gidoilaria", "written_by", "Inception"),
        # book
        ("Don Quixote egilea", "author", "Don Quixote"),
        ("nork argitaratu zuen War and Peace", "publisher", "War and Peace"),
        # company / organization
        ("nor da Apple ceo", "ceo", "Apple"),
        ("noiz sortu zen Apple", "founded", "Apple"),
        ("Google sortzailea", "founders", "Google"),
        ("zein sektoretan dago Tesla", "industry", "Tesla"),
        ("non dago Apple egoitza nagusia", "headquarters", "Apple"),
        ("zenbat langile ditu Google", "number_of_employees", "Google"),
        ("zenbat diru egiten du Apple", "revenue", "Apple"),
        # sports team
        ("zenbat txapelketa irabazi ditu Real Madrid", "championships", "Real Madrid"),
        ("nor da FC Barcelona entrenatzaile nagusia", "head_coach", "FC Barcelona"),
        ("zein estadiotan jokatzen du Real Madrid", "stadium", "Real Madrid"),
        # geography
        ("zein da Frantzia hiriburua", "capital", "Frantzia"),
        ("zein herrialdetan dago Eiffel dorrea", "country", "Eiffel dorrea"),
        ("zein da Everest altitudea", "elevation", "Everest"),
        ("zenbat luzera du Amazon", "length", "Amazon"),
        ("non dago Eiffel dorrea", "location", "Eiffel dorrea"),
        ("zein da Kanada azalera", "area", "Kanada"),
        ("zenbat biztanle ditu Tokio", "population", "Tokio"),
        # animal
        ("tigrea kontserbazio egoera", "conservation_status", "tigrea"),
        ("zer jaten du lehoia", "diet", "lehoia"),
        ("non bizi da hartz polarra", "habitat", "hartz polarra"),
        ("zein da otso grisaren izen zientifikoa", "scientific_name", "otso grisaren"),
        # music
        ("zein diskografikatan dago Taylor Swift", "label", "Taylor Swift"),
        ("zein dira The Beatles kideak", "members", "The Beatles"),
        # technology
        ("nork garatu zuen Linux", "developer", "Linux"),
        ("zein lizentziapean dago Linux", "license", "Linux"),
        ("Minecraft plataformak", "platforms", "Minecraft"),
        # place / general
        ("nork diseinatu zuen Eiffel dorrea", "designed_by", "Eiffel dorrea"),
        ("zein da Tesla ezizena", "nickname", "Tesla"),
        ("noiz inauguratu zen Eiffel dorrea", "opened", "Eiffel dorrea"),
        ("zein dira Shakespeare beste izenak", "other_names", "Shakespeare"),
        ("noren jabea da Twitter", "owner", "Twitter"),
        ("nork asmatu zuen telefonoa", "inventor", "telefonoa"),
    ],
    "fr": [
        # person / biography
        ("quel âge a Einstein", "age", "Einstein"),
        ("quel âge avait Hawking lorsqu'il est mort", "age_at_death", "Hawking"),
        ("dans quelle université Hawking a-t-il étudié", "alma_mater", "Hawking"),
        ("quand Einstein a-t-il été baptisé", "baptised", "Einstein"),
        ("quand Albert Einstein est-il né", "born", "Albert Einstein"),
        ("combien d'enfants a Darwin", "children", "Darwin"),
        ("quelle est la nationalité de Obama", "citizenship", "Obama"),
        ("qui entraîne Real Madrid", "coached_by", "Real Madrid"),
        ("quand Stephen Hawking est-il mort", "died", "Stephen Hawking"),
        ("qui étaient les doctorants de Hawking", "doctoral_students", "Hawking"),
        ("où Einstein a fait ses études", "education", "Einstein"),
        ("qui est le père de Darwin", "father", "Darwin"),
        ("dans quel domaine Newton est-il spécialiste", "fields", "Newton"),
        ("quelle est la taille de LeBron James", "height", "LeBron James"),
        ("où Turing a-t-il travaillé", "institutions", "Turing"),
        ("pour quoi Marie Curie est-il connu", "known_for", "Marie Curie"),
        ("qui est la mère de Darwin", "mother", "Darwin"),
        ("à quel mouvement artistique Picasso était-il associé", "movement", "Picasso"),
        ("quelle est la fortune de Elon Musk", "net_worth", "Elon Musk"),
        ("quels prix Curie a-t-il remportés", "notable_awards", "Curie"),
        ("quelle est l'œuvre la plus célèbre de Picasso", "notable_work", "Picasso"),
        ("quelle est la profession de Hawking", "occupation", "Hawking"),
        ("quel est le site officiel de Newton", "official_website", "Newton"),
        ("qui est le partenaire de Einstein", "partner", "Einstein"),
        ("quel est le parti politique de Biden", "political_party", "Biden"),
        ("qui a précédé Elizabeth II", "predecessor", "Elizabeth II"),
        ("quelle religion pratique Einstein", "religion", "Einstein"),
        ("où est enterré Newton", "resting_place", "Newton"),
        ("qui est le conjoint de Einstein", "spouse", "Einstein"),
        ("quel sport pratique Serena Williams", "sport", "Serena Williams"),
        ("qui a succédé à Elizabeth II", "successor", "Elizabeth II"),
        ("quel est le sujet de thèse de Darwin", "thesis", "Darwin"),
        ("quand était Serena Williams actif", "years_active", "Serena Williams"),
        # movie
        ("combien a rapporté Inception", "box_office", "Inception"),
        ("quel était le budget de Inception", "budget", "Inception"),
        ("qui a composé la musique de Inception", "composers", "Inception"),
        ("qui a réalisé Inception", "director", "Inception"),
        ("quel genre est Inception", "genre", "Inception"),
        ("en quelle langue est écrit Don Quixote", "language", "Don Quixote"),
        ("quel est le titre original de The Odyssey", "original_title", "The Odyssey"),
        ("qui a produit Inception", "produced_by", "Inception"),
        ("quand est sorti Inception", "released", "Inception"),
        ("quelle est la durée de Inception", "running_time", "Inception"),
        ("qui joue dans Inception", "starring", "Inception"),
        ("qui a écrit le scénario de Inception", "written_by", "Inception"),
        # book
        ("qui a écrit Don Quixote", "author", "Don Quixote"),
        ("qui a publié War and Peace", "publisher", "War and Peace"),
        # company / organization
        ("qui est le pdg de Apple", "ceo", "Apple"),
        ("quand a été fondé Apple", "founded", "Apple"),
        ("qui a fondé Google", "founders", "Google"),
        ("dans quel secteur est Tesla", "industry", "Tesla"),
        ("où est le siège social de Apple", "headquarters", "Apple"),
        ("combien d employés a Google", "number_of_employees", "Google"),
        ("quel est le chiffre d affaires de Apple", "revenue", "Apple"),
        # sports team
        ("combien de championnats Real Madrid a remporté", "championships", "Real Madrid"),
        ("qui est l entraîneur de FC Barcelona", "head_coach", "FC Barcelona"),
        ("quel stade utilise Real Madrid", "stadium", "Real Madrid"),
        # geography
        ("quelle est la capitale de France", "capital", "France"),
        ("dans quel pays est la tour Eiffel", "country", "la tour Eiffel"),
        ("quelle est l altitude de Everest", "elevation", "Everest"),
        ("la Seine longueur", "length", "la Seine"),
        ("où se trouve la tour Eiffel", "location", "la tour Eiffel"),
        ("quelle est la superficie de Canada", "area", "Canada"),
        ("quelle est la population de Tokyo", "population", "Tokyo"),
        # animal
        ("tigre statut de conservation", "conservation_status", "tigre"),
        ("que mange le lion", "diet", "le lion"),
        ("où vit ours polaire", "habitat", "ours polaire"),
        ("quel est le nom scientifique de loup gris", "scientific_name", "loup gris"),
        # music
        ("à quel label appartient Taylor Swift", "label", "Taylor Swift"),
        ("qui sont les membres de The Beatles", "members", "The Beatles"),
        # technology
        ("qui a développé Linux", "developer", "Linux"),
        ("sous quelle licence est Linux", "license", "Linux"),
        ("sur quelles plateformes est disponible Minecraft", "platforms", "Minecraft"),
        # place / general
        ("la tour Eiffel concepteur", "designed_by", "la tour Eiffel"),
        ("quel est le surnom de Tesla", "nickname", "Tesla"),
        ("quand la tour Eiffel a été inauguré", "opened", "la tour Eiffel"),
        ("quels sont les autres noms de Shakespeare", "other_names", "Shakespeare"),
        ("qui possède Twitter", "owner", "Twitter"),
        ("qui a inventé le téléphone", "inventor", "le téléphone"),
    ],
    "gl": [
        # person / biography
        ("cantos anos ten Einstein", "age", "Einstein"),
        ("cantos anos tiña Hawking cando morreu", "age_at_death", "Hawking"),
        ("cal é a alma mater de Hawking", "alma_mater", "Hawking"),
        ("cando foi bautizado Newton", "baptised", "Newton"),
        ("cando naceu Albert Einstein", "born", "Albert Einstein"),
        ("cantos fillos tivo Darwin", "children", "Darwin"),
        ("cal é a cidadanía de Obama", "citizenship", "Obama"),
        ("quen adestra Real Madrid", "coached_by", "Real Madrid"),
        ("cando finou Stephen Hawking", "died", "Stephen Hawking"),
        ("quen foron os doutorandos de Hawking", "doctoral_students", "Hawking"),
        ("onde se formou Einstein", "education", "Einstein"),
        ("quen é o pai de Darwin", "father", "Darwin"),
        ("cal é o eido de especialización de Newton", "fields", "Newton"),
        ("canto mide LeBron James", "height", "LeBron James"),
        ("onde traballou Turing", "institutions", "Turing"),
        ("por que se coñece Marie Curie", "known_for", "Marie Curie"),
        ("quen é a nai de Darwin", "mother", "Darwin"),
        ("a que movemento artístico pertencía Picasso", "movement", "Picasso"),
        ("cal é o patrimonio neto de Elon Musk", "net_worth", "Elon Musk"),
        ("que premios gañou Curie", "notable_awards", "Curie"),
        ("cal é a obra máis famosa de Picasso", "notable_work", "Picasso"),
        ("cal é a profesión de Hawking", "occupation", "Hawking"),
        ("cal é o sitio web oficial de Newton", "official_website", "Newton"),
        ("quen é a parella de Einstein", "partner", "Einstein"),
        ("a que partido político pertence Biden", "political_party", "Biden"),
        ("quen precedeu a Elizabeth II", "predecessor", "Elizabeth II"),
        ("que relixión practica Einstein", "religion", "Einstein"),
        ("onde está enterrado Newton", "resting_place", "Newton"),
        ("quen é o cónxuxe de Einstein", "spouse", "Einstein"),
        ("que deporte practica Serena Williams", "sport", "Serena Williams"),
        ("quen sucedeu a Elizabeth II", "successor", "Elizabeth II"),
        ("que é o tema da tese Darwin", "thesis", "Darwin"),
        ("cando estivo activo Serena Williams", "years_active", "Serena Williams"),
        # movie
        ("canto recadou Inception", "box_office", "Inception"),
        ("cal foi o orzamento de Inception", "budget", "Inception"),
        ("quen compuxo a música de Inception", "composers", "Inception"),
        ("quen dirixiu Inception", "director", "Inception"),
        ("cal é o xénero de Inception", "genre", "Inception"),
        ("en que idioma está escrito Don Quixote", "language", "Don Quixote"),
        ("cal é o título orixinal de The Odyssey", "original_title", "The Odyssey"),
        ("quen produciu Inception", "produced_by", "Inception"),
        ("cando se estreou Inception", "released", "Inception"),
        ("canto dura Inception", "running_time", "Inception"),
        ("quen actúa en Inception", "starring", "Inception"),
        ("quen escribiu o guión de Inception", "written_by", "Inception"),
        # book
        ("quen escribiu Don Quixote", "author", "Don Quixote"),
        ("quen publicou War and Peace", "publisher", "War and Peace"),
        # company / organization
        ("quen é o ceo de Apple", "ceo", "Apple"),
        ("cando foi fundado Apple", "founded", "Apple"),
        ("quen fundou Google", "founders", "Google"),
        ("en que sector está Tesla", "industry", "Tesla"),
        ("onde está a sede de Apple", "headquarters", "Apple"),
        ("cantos empregados ten Google", "number_of_employees", "Google"),
        ("cales son os ingresos de Apple", "revenue", "Apple"),
        # sports team
        ("cantos campeonatos gañou Real Madrid", "championships", "Real Madrid"),
        ("quen é o adestredor de FC Barcelona", "head_coach", "FC Barcelona"),
        ("en que estadio xoga Real Madrid", "stadium", "Real Madrid"),
        # geography
        ("cal é a capital de Francia", "capital", "Francia"),
        ("en que país está a Torre Eiffel", "country", "a Torre Eiffel"),
        ("cal é a altitude de Everest", "elevation", "Everest"),
        ("cal é a lonxitude de Amazon", "length", "Amazon"),
        ("onde se atopa a Torre Eiffel", "location", "a Torre Eiffel"),
        ("cal é a área de Canadá", "area", "Canadá"),
        ("cal é a poboación de Tokio", "population", "Tokio"),
        # animal
        ("tigre estado de conservación", "conservation_status", "tigre"),
        ("que come o león", "diet", "o león"),
        ("onde vive o oso polar", "habitat", "o oso polar"),
        ("cal é o nome científico de lobo gris", "scientific_name", "lobo gris"),
        # music
        ("a que sello discográfico pertence Taylor Swift", "label", "Taylor Swift"),
        ("quen son os membros de The Beatles", "members", "The Beatles"),
        # technology
        ("quen desenvolveu Linux", "developer", "Linux"),
        ("baixo que licenza está Linux", "license", "Linux"),
        ("en que plataformas está Minecraft", "platforms", "Minecraft"),
        # place / general
        ("quen deseñou a Torre Eiffel", "designed_by", "a Torre Eiffel"),
        ("cal é o alcume de Tesla", "nickname", "Tesla"),
        ("cando foi inaugurado a Torre Eiffel", "opened", "a Torre Eiffel"),
        ("cales son outros nomes de Shakespeare", "other_names", "Shakespeare"),
        ("quen é o propietario de Twitter", "owner", "Twitter"),
        ("quen inventou o teléfono", "inventor", "o teléfono"),
    ],
    "it": [
        # person / biography
        ("quanti anni ha Einstein", "age", "Einstein"),
        ("quanti anni aveva Hawking quando è morto", "age_at_death", "Hawking"),
        ("qual è l'alma mater di Hawking", "alma_mater", "Hawking"),
        ("quando è stato battezzato Newton", "baptised", "Newton"),
        ("quando è nata Albert Einstein", "born", "Albert Einstein"),
        ("quanti figli aveva Darwin", "children", "Darwin"),
        ("qual è la cittadinanza di Obama", "citizenship", "Obama"),
        ("chi allena Real Madrid", "coached_by", "Real Madrid"),
        ("quando è morto Stephen Hawking", "died", "Stephen Hawking"),
        ("chi erano i dottorandi di Hawking", "doctoral_students", "Hawking"),
        ("dove si è formato Einstein", "education", "Einstein"),
        ("chi è il padre di Darwin", "father", "Darwin"),
        ("qual è il campo di specializzazione di Newton", "fields", "Newton"),
        ("quanto è alto LeBron James", "height", "LeBron James"),
        ("dove ha lavorato Turing", "institutions", "Turing"),
        ("per cosa è famosa Marie Curie", "known_for", "Marie Curie"),
        ("chi è la madre di Darwin", "mother", "Darwin"),
        ("a quale movimento artistico apparteneva Picasso", "movement", "Picasso"),
        ("qual è il patrimonio netto di Elon Musk", "net_worth", "Elon Musk"),
        ("quali premi ha vinto Curie", "notable_awards", "Curie"),
        ("qual è l'opera più famosa di Picasso", "notable_work", "Picasso"),
        ("qual è la professione di Hawking", "occupation", "Hawking"),
        ("qual è il sito ufficiale di Newton", "official_website", "Newton"),
        ("chi è il partner di Einstein", "partner", "Einstein"),
        ("a quale partito politico appartiene Biden", "political_party", "Biden"),
        ("chi ha preceduto Elizabeth II", "predecessor", "Elizabeth II"),
        ("quale religione pratica Einstein", "religion", "Einstein"),
        ("dove è sepolto Newton", "resting_place", "Newton"),
        ("chi è il coniuge di Einstein", "spouse", "Einstein"),
        ("quale sport pratica Serena Williams", "sport", "Serena Williams"),
        ("chi è succeduto a Elizabeth II", "successor", "Elizabeth II"),
        ("qual è il tema della tesi di Darwin", "thesis", "Darwin"),
        ("quando era attivo Serena Williams", "years_active", "Serena Williams"),
        # movie
        ("quanto ha incassato Inception", "box_office", "Inception"),
        ("qual era il budget di Inception", "budget", "Inception"),
        ("chi ha composto la musica di Inception", "composers", "Inception"),
        ("chi ha diretto Inception", "director", "Inception"),
        ("che genere è Inception", "genre", "Inception"),
        ("in che lingua è scritto Don Chisciotte", "language", "Don Chisciotte"),
        ("qual è il titolo originale di The Odyssey", "original_title", "The Odyssey"),
        ("chi ha prodotto Inception", "produced_by", "Inception"),
        ("quando è uscito Inception", "released", "Inception"),
        ("quanto dura Inception", "running_time", "Inception"),
        ("chi recita in Inception", "starring", "Inception"),
        ("chi ha scritto la sceneggiatura di Inception", "written_by", "Inception"),
        # book
        ("chi ha scritto Don Chisciotte", "author", "Don Chisciotte"),
        ("chi ha pubblicato War and Peace", "publisher", "War and Peace"),
        # company / organization
        ("chi è il ceo di Apple", "ceo", "Apple"),
        ("quando è stato fondato Apple", "founded", "Apple"),
        ("chi ha fondato Google", "founders", "Google"),
        ("in quale settore opera Tesla", "industry", "Tesla"),
        ("dove ha sede Apple", "headquarters", "Apple"),
        ("quanti dipendenti ha Google", "number_of_employees", "Google"),
        ("qual è il fatturato di Apple", "revenue", "Apple"),
        # sports team
        ("quanti campionati ha vinto Real Madrid", "championships", "Real Madrid"),
        ("chi è l allenatore di FC Barcelona", "head_coach", "FC Barcelona"),
        ("in quale stadio gioca Real Madrid", "stadium", "Real Madrid"),
        # geography
        ("qual è la capitale di Francia", "capital", "Francia"),
        ("in quale paese si trova la Torre Eiffel", "country", "la Torre Eiffel"),
        ("qual è l altitudine di Everest", "elevation", "Everest"),
        ("qual è la lunghezza di Amazon", "length", "Amazon"),
        ("dove si trova la Torre Eiffel", "location", "la Torre Eiffel"),
        ("qual è l area di Canada", "area", "Canada"),
        ("qual è la popolazione di Tokyo", "population", "Tokyo"),
        # animal
        ("tigre stato di conservazione", "conservation_status", "tigre"),
        ("cosa mangia il leone", "diet", "il leone"),
        ("dove vive orso polare", "habitat", "orso polare"),
        ("qual è il nome scientifico di lupo grigio", "scientific_name", "lupo grigio"),
        # music
        ("a quale etichetta discografica appartiene Taylor Swift", "label", "Taylor Swift"),
        ("chi sono i membri di The Beatles", "members", "The Beatles"),
        # technology
        ("chi ha sviluppato Linux", "developer", "Linux"),
        ("sotto quale licenza è Linux", "license", "Linux"),
        ("su quali piattaforme è disponibile Minecraft", "platforms", "Minecraft"),
        # place / general
        ("chi ha progettato la Torre Eiffel", "designed_by", "la Torre Eiffel"),
        ("qual è il soprannome di Tesla", "nickname", "Tesla"),
        ("quando è stato inaugurato la Torre Eiffel", "opened", "la Torre Eiffel"),
        ("quali sono gli altri nomi di Shakespeare", "other_names", "Shakespeare"),
        ("chi possiede Twitter", "owner", "Twitter"),
        ("chi ha inventato il telefono", "inventor", "il telefono"),
    ],
    "pt": [
        # person / biography
        ("quantos anos tem Einstein", "age", "Einstein"),
        ("quantos anos tinha Hawking quando morreu", "age_at_death", "Hawking"),
        ("qual é a alma mater de Hawking", "alma_mater", "Hawking"),
        ("quando foi batizado Newton", "baptised", "Newton"),
        ("quando nasceu Albert Einstein", "born", "Albert Einstein"),
        ("quantos filhos tinha Darwin", "children", "Darwin"),
        ("qual é a cidadania de Obama", "citizenship", "Obama"),
        ("quem treina Real Madrid", "coached_by", "Real Madrid"),
        ("quando foi a morte de Stephen Hawking", "died", "Stephen Hawking"),
        ("quem foram os doutorandos de Hawking", "doctoral_students", "Hawking"),
        ("onde se formou Einstein", "education", "Einstein"),
        ("quem é o pai de Darwin", "father", "Darwin"),
        ("qual é o campo de especialização de Newton", "fields", "Newton"),
        ("qual é a altura de LeBron James", "height", "LeBron James"),
        ("onde trabalhou Turing", "institutions", "Turing"),
        ("pelo que é Marie Curie conhecida", "known_for", "Marie Curie"),
        ("quem é a mãe de Darwin", "mother", "Darwin"),
        ("a que movimento artístico pertencia Picasso", "movement", "Picasso"),
        ("qual é o patrimônio de Elon Musk", "net_worth", "Elon Musk"),
        ("que prémios ganhou Curie", "notable_awards", "Curie"),
        ("qual é a obra mais famosa de Picasso", "notable_work", "Picasso"),
        ("qual é a profissão de Hawking", "occupation", "Hawking"),
        ("qual é o site oficial de Newton", "official_website", "Newton"),
        ("quem é o parceiro de Einstein", "partner", "Einstein"),
        ("a que partido político pertence Biden", "political_party", "Biden"),
        ("quem precedeu Elizabeth II", "predecessor", "Elizabeth II"),
        ("que religião pratica Einstein", "religion", "Einstein"),
        ("onde está enterrado Newton", "resting_place", "Newton"),
        ("quem é o cônjuge de Einstein", "spouse", "Einstein"),
        ("que desporto pratica Serena Williams", "sport", "Serena Williams"),
        ("quem sucedeu a Elizabeth II", "successor", "Elizabeth II"),
        ("qual é o tema da tese de Darwin", "thesis", "Darwin"),
        ("quando estava ativo Serena Williams", "years_active", "Serena Williams"),
        # movie
        ("quanto arrecadou Inception", "box_office", "Inception"),
        ("qual foi o orçamento de Inception", "budget", "Inception"),
        ("quem compôs a música de Inception", "composers", "Inception"),
        ("quem dirigiu Inception", "director", "Inception"),
        ("qual é o género de Inception", "genre", "Inception"),
        ("em que língua está escrito Don Quixote", "language", "Don Quixote"),
        ("qual é o título original de The Odyssey", "original_title", "The Odyssey"),
        ("quem produziu Inception", "produced_by", "Inception"),
        ("quando foi lançado Inception", "released", "Inception"),
        ("qual é a duração de Inception", "running_time", "Inception"),
        ("quem atua em Inception", "starring", "Inception"),
        ("quem escreveu o roteiro de Inception", "written_by", "Inception"),
        # book
        ("quem escreveu Dom Quixote", "author", "Dom Quixote"),
        ("quem publicou War and Peace", "publisher", "War and Peace"),
        # company / organization
        ("quem é o ceo de Apple", "ceo", "Apple"),
        ("quando foi fundado Apple", "founded", "Apple"),
        ("quem fundou Google", "founders", "Google"),
        ("em que setor está Tesla", "industry", "Tesla"),
        ("onde fica a sede de Apple", "headquarters", "Apple"),
        ("quantos funcionários tem Google", "number_of_employees", "Google"),
        ("qual é a receita de Apple", "revenue", "Apple"),
        # sports team
        ("quantos campeonatos Real Madrid ganhou", "championships", "Real Madrid"),
        ("quem é o treinador de FC Barcelona", "head_coach", "FC Barcelona"),
        ("em que estádio joga Real Madrid", "stadium", "Real Madrid"),
        # geography
        ("qual é a capital de França", "capital", "França"),
        ("em que país está a Torre Eiffel", "country", "a Torre Eiffel"),
        ("qual é a elevação de Everest", "elevation", "Everest"),
        ("qual é o comprimento de Amazon", "length", "Amazon"),
        ("onde se encontra a Torre Eiffel", "location", "a Torre Eiffel"),
        ("qual é a área de Canadá", "area", "Canadá"),
        ("qual é a população de Tóquio", "population", "Tóquio"),
        # animal
        ("tigre estado de conservação", "conservation_status", "tigre"),
        ("o que come o leão", "diet", "o leão"),
        ("onde vive o urso polar", "habitat", "o urso polar"),
        ("qual é o nome científico de lobo cinzento", "scientific_name", "lobo cinzento"),
        # music
        ("a que gravadora pertence Taylor Swift", "label", "Taylor Swift"),
        ("quem são os membros de The Beatles", "members", "The Beatles"),
        # technology
        ("quem desenvolveu Linux", "developer", "Linux"),
        ("sob que licença está Linux", "license", "Linux"),
        ("em que plataformas está Minecraft", "platforms", "Minecraft"),
        # place / general
        ("quem projetou a Torre Eiffel", "designed_by", "a Torre Eiffel"),
        ("qual é a alcunha de Tesla", "nickname", "Tesla"),
        ("quando a Torre Eiffel foi inaugurado", "opened", "a Torre Eiffel"),
        ("quais são outros nomes de Shakespeare", "other_names", "Shakespeare"),
        ("quem é o dono de Twitter", "owner", "Twitter"),
        ("quem inventou o telefone", "inventor", "o telefone"),
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
            normalize_for_match(kw), normalize_for_match(expected_keyword),
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


# ---------------------------------------------------------------------------
# Ambiguous intent cases
# Each entry: (lang, utterance, [candidate_intent, ...], keyword)
# These utterances genuinely match multiple intents with equal confidence.
# The matcher returns one arbitrarily; real disambiguation happens downstream
# by querying the infobox and picking the field that has a non-empty value.
# ---------------------------------------------------------------------------

_AMBIGUOUS_INTENT_CASES = [
    # "wie lang ist Amazon" — ties between length({place}) and running_time({movie})
    ("de-DE", "wie lang ist Amazon", ["length", "running_time"], "Amazon"),
    # "quelle est la longueur de Amazon" — same tie in French
    ("fr-FR", "quelle est la longueur de Amazon", ["length", "running_time"], "Amazon"),
]


def _make_ambiguous_test(lang, utterance, intents, keyword):
    def test(self):
        intent, kw = self.engine._match_infobox_intent(utterance, lang)
        self.assertIn(
            intent, intents,
            f"[{lang}] {utterance!r} → expected one of {intents!r}, got {intent!r}",
        )
        self.assertEqual(
            kw.lower(), keyword.lower(),
            f"[{lang}] {utterance!r} → expected keyword {keyword!r}, got {kw!r}",
        )
    slug = keyword.lower().replace(" ", "_")
    test.__name__ = f"test_ambiguous_{lang}_{'_or_'.join(intents)}_{slug}"
    return test


for _lang, _utterance, _intents, _keyword in _AMBIGUOUS_INTENT_CASES:
    _name = f"test_ambiguous_{_lang}_{'_or_'.join(_intents)}_{_keyword.lower().replace(' ', '_')}"
    setattr(TestIntentParsingAllLanguages, _name, _make_ambiguous_test(_lang, _utterance, _intents, _keyword))


if __name__ == "__main__":
    unittest.main()

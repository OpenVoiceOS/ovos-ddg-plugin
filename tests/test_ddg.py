"""
Unit tests for ovos-ddg-solver-plugin.

All network calls and OVOS config are mocked — no API key required.
"""
import unittest
from unittest.mock import MagicMock, patch

from ovos_ddg_solver import (
    DuckDuckGoRetrievalEngine,
    DuckDuckGoToolbox,
    SearchDuckDuckGoArgs,
    SearchDuckDuckGoOutput,
)


def _make_engine():
    with patch("ovos_ddg_solver.Configuration", return_value={}):
        engine = DuckDuckGoRetrievalEngine.__new__(DuckDuckGoRetrievalEngine)
        engine.config = {}
        engine._kword_extractors = {}
        engine._intent_matchers = {}
        return engine


# ---------------------------------------------------------------------------
# _fetch
# ---------------------------------------------------------------------------

class TestFetch(unittest.TestCase):

    def setUp(self):
        self.engine = _make_engine()

    def test_returns_json_on_success(self):
        payload = {"AbstractText": "Isaac Newton was a physicist."}
        with patch("ovos_ddg_solver.requests.get") as mock_get, \
             patch("ovos_ddg_solver.Configuration", return_value={}):
            mock_get.return_value.json.return_value = payload
            result = self.engine._fetch("Isaac Newton", lang="en-US")
        self.assertEqual(result, payload)

    def test_returns_empty_on_exception(self):
        with patch("ovos_ddg_solver.requests.get", side_effect=Exception("network error")):
            result = self.engine._fetch("Isaac Newton", lang="en-US")
        self.assertEqual(result, {})

    def test_returns_empty_for_unknown_locale(self):
        result = self.engine._fetch("query", lang="xx-XX")
        self.assertEqual(result, {})


# ---------------------------------------------------------------------------
# _search
# ---------------------------------------------------------------------------

class TestSearch(unittest.TestCase):

    def setUp(self):
        self.engine = _make_engine()

    def test_returns_direct_match(self):
        payload = {"AbstractText": "Newton invented calculus."}
        self.engine._fetch = MagicMock(return_value=payload)
        result = self.engine._search("Isaac Newton", lang="en-us")
        self.assertEqual(result, payload)
        self.engine._fetch.assert_called_once()

    def test_falls_back_to_keyword_extraction(self):
        self.engine._fetch = MagicMock(side_effect=[
            {},
            {"AbstractText": "Gravity answer."},
        ])
        kwx = MagicMock()
        kwx.extract.return_value = ["gravity"]
        self.engine._keyword_extractor = MagicMock(return_value=kwx)
        result = self.engine._search("what is gravity", lang="en-us")
        self.assertEqual(result["AbstractText"], "Gravity answer.")


# ---------------------------------------------------------------------------
# get_infobox
# ---------------------------------------------------------------------------

class TestGetInfobox(unittest.TestCase):

    def setUp(self):
        self.engine = _make_engine()

    def test_returns_infobox_and_related(self):
        self.engine._search = MagicMock(return_value={
            "Infobox": {"content": [{"label": "Born", "value": "1643"}]},
            "RelatedTopics": [{"Text": "gravity"}],
        })
        with patch("ovos_ddg_solver.Configuration", return_value={}):
            infobox, related = self.engine.get_infobox("Newton", lang="en-us")
        self.assertEqual(infobox.get("born"), "1643")
        self.assertIn("gravity", related)

    def test_returns_empty_when_no_data(self):
        self.engine._search = MagicMock(return_value={})
        with patch("ovos_ddg_solver.Configuration", return_value={}):
            infobox, related = self.engine.get_infobox("xyzzy", lang="en-us")
        self.assertEqual(infobox, {})
        self.assertEqual(related, [])


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------

class TestQuery(unittest.TestCase):

    def setUp(self):
        self.engine = _make_engine()

    def test_returns_sentences_from_abstract(self):
        self.engine._match_infobox_intent = MagicMock(return_value=(None, "Isaac Newton"))
        self.engine._search = MagicMock(
            return_value={"AbstractText": "Newton was born in 1643. He invented calculus."}
        )
        with patch("ovos_ddg_solver.Configuration", return_value={}):
            results = self.engine.query("Isaac Newton", lang="en-us", k=5)
        self.assertIsInstance(results, list)
        self.assertTrue(len(results) > 0)
        self.assertIsInstance(results[0], tuple)
        self.assertEqual(len(results[0]), 2)

    def test_respects_k_limit(self):
        self.engine._match_infobox_intent = MagicMock(return_value=(None, "Newton"))
        self.engine._search = MagicMock(return_value={"AbstractText": "A. B. C. D. E. F."})
        with patch("ovos_ddg_solver.Configuration", return_value={}):
            results = self.engine.query("Newton", lang="en-us", k=2)
        self.assertLessEqual(len(results), 2)

    def test_returns_empty_when_no_data(self):
        self.engine._match_infobox_intent = MagicMock(return_value=(None, "xyzzy"))
        self.engine._search = MagicMock(return_value={})
        with patch("ovos_ddg_solver.Configuration", return_value={}):
            results = self.engine.query("xyzzy", lang="en-us")
        self.assertEqual(results, [])

    def test_returns_infobox_answer_with_high_score(self):
        self.engine._match_infobox_intent = MagicMock(return_value=("born", "Hawking"))
        self.engine.get_infobox = MagicMock(return_value=({"born": "8 January 1942"}, []))
        with patch("ovos_ddg_solver.Configuration", return_value={}):
            results = self.engine.query("when was Hawking born", lang="en-us")
        self.assertEqual(results, [("8 January 1942", 0.9)])

    def test_abstract_sentences_have_lower_score(self):
        self.engine._match_infobox_intent = MagicMock(return_value=(None, "Newton"))
        self.engine._search = MagicMock(return_value={"AbstractText": "Newton was a physicist."})
        with patch("ovos_ddg_solver.Configuration", return_value={}):
            results = self.engine.query("Newton", lang="en-us")
        self.assertTrue(all(score == 0.7 for _, score in results))


# ---------------------------------------------------------------------------
# Intent matching
# ---------------------------------------------------------------------------

class TestIntentMatching(unittest.TestCase):

    def setUp(self):
        self.engine = _make_engine()

    def test_no_matchers_returns_utterance_unchanged(self):
        intent, kw = self.engine._match_infobox_intent("when was Einstein born", "en")
        self.assertIsNone(intent)
        self.assertEqual(kw, "when was Einstein born")

    def test_register_and_match(self):
        self.engine._register_intent("born", ["when was {query} born"], "en")
        intent, kw = self.engine._match_infobox_intent("when was Einstein born", "en")
        self.assertEqual(intent, "born")
        self.assertEqual(kw, "Einstein")


# ---------------------------------------------------------------------------
# DuckDuckGoToolbox
# ---------------------------------------------------------------------------

class TestDuckDuckGoToolbox(unittest.TestCase):

    def _make_toolbox(self):
        with patch("ovos_ddg_solver.DuckDuckGoRetrievalEngine"):
            return DuckDuckGoToolbox(config={})

    def test_discover_tools_returns_one_tool(self):
        tb = self._make_toolbox()
        tools = tb.discover_tools()
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].name, "search_duckduckgo")

    def test_search_ddg_returns_output(self):
        tb = self._make_toolbox()
        tb._engine = MagicMock()
        tb._engine.query.return_value = [("Newton invented calculus.", 0.7)]
        args = SearchDuckDuckGoArgs(query="Isaac Newton", lang="en-us")
        result = tb.search_ddg(args)
        self.assertIsInstance(result, SearchDuckDuckGoOutput)
        self.assertEqual(result.result, "Newton invented calculus.")

    def test_search_ddg_returns_empty_string_on_no_answer(self):
        tb = self._make_toolbox()
        tb._engine = MagicMock()
        tb._engine.query.return_value = []
        args = SearchDuckDuckGoArgs(query="xyzzy", lang="en-us")
        result = tb.search_ddg(args)
        self.assertEqual(result.result, "")

    def test_toolbox_id(self):
        self.assertEqual(DuckDuckGoToolbox.toolbox_id, "ovos-ddg-tools")


# ---------------------------------------------------------------------------
# Plugin loading
# ---------------------------------------------------------------------------

class TestPluginLoading(unittest.TestCase):

    def test_imports(self):
        from ovos_ddg_solver import (
            DuckDuckGoRetrievalEngine,
            DuckDuckGoToolbox,
            SearchDuckDuckGoArgs,
            SearchDuckDuckGoOutput,
        )


if __name__ == "__main__":
    unittest.main()

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


# ---------------------------------------------------------------------------
# DuckDuckGoRetrievalEngine
# ---------------------------------------------------------------------------

def _make_engine():
    with patch("ovos_ddg_solver.Configuration", return_value={}):
        engine = DuckDuckGoRetrievalEngine.__new__(DuckDuckGoRetrievalEngine)
        engine.config = {}
        engine.kword_extractors = {}
        engine.intent_matchers = {}
        return engine


class TestDuckDuckGoRetrievalEngine(unittest.TestCase):

    def setUp(self):
        self.engine = _make_engine()

    # ------------------------------------------------------------------
    # get_data
    # ------------------------------------------------------------------

    def test_get_data_returns_json_on_success(self):
        payload = {"AbstractText": "Isaac Newton was a physicist."}
        with patch("ovos_ddg_solver.requests.get") as mock_get, \
             patch("ovos_ddg_solver.Configuration", return_value={"lang": "en-us", "system_unit": "metric"}):
            mock_get.return_value.json.return_value = payload
            result = self.engine.get_data("Isaac Newton", lang="en-US")
        self.assertEqual(result, payload)

    def test_get_data_returns_empty_on_exception(self):
        with patch("ovos_ddg_solver.requests.get", side_effect=Exception("network error")), \
             patch("ovos_ddg_solver.Configuration", return_value={}):
            result = self.engine.get_data("Isaac Newton", lang="en-US")
        self.assertEqual(result, {})

    def test_get_data_returns_empty_for_unknown_locale(self):
        with patch("ovos_ddg_solver.Configuration", return_value={}):
            result = self.engine.get_data("query", lang="xx-XX")
        self.assertEqual(result, {})

    # ------------------------------------------------------------------
    # extract_and_search
    # ------------------------------------------------------------------

    def test_extract_and_search_returns_direct_match(self):
        payload = {"AbstractText": "Newton invented calculus."}
        self.engine.get_data = MagicMock(return_value=payload)
        result = self.engine.extract_and_search("Isaac Newton")
        self.assertEqual(result, payload)
        self.engine.get_data.assert_called_once()

    def test_extract_and_search_falls_back_to_keyword_extraction(self):
        self.engine.get_data = MagicMock(side_effect=[
            {},  # first call — no AbstractText
            {"AbstractText": "Gravity answer."},  # second call after keyword extraction
        ])
        kwx = MagicMock()
        kwx.extract.return_value = ["gravity"]
        self.engine.get_keyword_extractor = MagicMock(return_value=kwx)
        result = self.engine.extract_and_search("what is gravity", lang="en")
        self.assertEqual(result["AbstractText"], "Gravity answer.")

    # ------------------------------------------------------------------
    # get_spoken_answer
    # ------------------------------------------------------------------

    def test_get_spoken_answer_returns_abstract(self):
        self.engine.match_infobox_intent = MagicMock(return_value=(None, "Isaac Newton"))
        self.engine.extract_and_search = MagicMock(return_value={"AbstractText": "Newton was a physicist."})
        with patch("ovos_ddg_solver.Configuration", return_value={}):
            result = self.engine.get_spoken_answer("Isaac Newton", lang="en-us")
        self.assertEqual(result, "Newton was a physicist.")

    def test_get_spoken_answer_returns_infobox_field(self):
        self.engine.match_infobox_intent = MagicMock(return_value=("born", "Stephen Hawking"))
        self.engine.get_infobox = MagicMock(return_value=({"born": "8 January 1942"}, []))
        with patch("ovos_ddg_solver.Configuration", return_value={}):
            result = self.engine.get_spoken_answer("when was Stephen Hawking born", lang="en-us")
        self.assertEqual(result, "8 January 1942")

    def test_get_spoken_answer_returns_none_when_no_data(self):
        self.engine.match_infobox_intent = MagicMock(return_value=(None, "xyzzy"))
        self.engine.extract_and_search = MagicMock(return_value={})
        with patch("ovos_ddg_solver.Configuration", return_value={}):
            result = self.engine.get_spoken_answer("xyzzy", lang="en-us")
        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # query
    # ------------------------------------------------------------------

    def test_query_returns_sentences(self):
        self.engine.match_infobox_intent = MagicMock(return_value=(None, "Isaac Newton"))
        self.engine.extract_and_search = MagicMock(
            return_value={"AbstractText": "Newton was born in 1643. He invented calculus."}
        )
        with patch("ovos_ddg_solver.Configuration", return_value={}):
            results = self.engine.query("Isaac Newton", lang="en-us", k=5)
        self.assertIsInstance(results, list)
        self.assertTrue(len(results) > 0)
        self.assertIsInstance(results[0], tuple)
        self.assertEqual(len(results[0]), 2)

    def test_query_respects_k_limit(self):
        self.engine.match_infobox_intent = MagicMock(return_value=(None, "Newton"))
        self.engine.extract_and_search = MagicMock(
            return_value={"AbstractText": "A. B. C. D. E. F."}
        )
        with patch("ovos_ddg_solver.Configuration", return_value={}):
            results = self.engine.query("Newton", lang="en-us", k=2)
        self.assertLessEqual(len(results), 2)

    def test_query_returns_empty_when_no_data(self):
        self.engine.match_infobox_intent = MagicMock(return_value=(None, "xyzzy"))
        self.engine.extract_and_search = MagicMock(return_value={})
        with patch("ovos_ddg_solver.Configuration", return_value={}):
            results = self.engine.query("xyzzy", lang="en-us")
        self.assertEqual(results, [])

    def test_query_returns_infobox_answer_with_high_score(self):
        self.engine.match_infobox_intent = MagicMock(return_value=("born", "Hawking"))
        self.engine.get_infobox = MagicMock(return_value=({"born": "8 January 1942"}, []))
        with patch("ovos_ddg_solver.Configuration", return_value={}):
            results = self.engine.query("when was Hawking born", lang="en-us")
        self.assertEqual(results, [("8 January 1942", 0.9)])

    # ------------------------------------------------------------------
    # get_image
    # ------------------------------------------------------------------

    def test_get_image_prepends_ddg_domain(self):
        self.engine.extract_and_search = MagicMock(return_value={"Image": "/i/abc.jpg"})
        result = self.engine.get_image("Newton", lang="en-us")
        self.assertTrue(result.startswith("https://duckduckgo.com"))

    def test_get_image_returns_full_url_unchanged(self):
        self.engine.extract_and_search = MagicMock(
            return_value={"Image": "https://duckduckgo.com/i/abc.jpg"}
        )
        result = self.engine.get_image("Newton", lang="en-us")
        self.assertEqual(result, "https://duckduckgo.com/i/abc.jpg")

    # ------------------------------------------------------------------
    # intent matching
    # ------------------------------------------------------------------

    def test_match_infobox_intent_no_matchers_returns_utterance(self):
        intent, kw = self.engine.match_infobox_intent("when was Einstein born", "en")
        self.assertIsNone(intent)
        self.assertEqual(kw, "when was Einstein born")

    def test_register_infobox_intent_creates_matcher(self):
        self.engine.register_infobox_intent("born.intent", ["when was {query} born"], "en")
        self.assertIn("en", self.engine.intent_matchers)


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
        tb._engine.get_spoken_answer.return_value = "Newton invented calculus."
        args = SearchDuckDuckGoArgs(query="Isaac Newton", lang="en-us")
        result = tb.search_ddg(args)
        self.assertIsInstance(result, SearchDuckDuckGoOutput)
        self.assertEqual(result.result, "Newton invented calculus.")

    def test_search_ddg_returns_empty_string_on_no_answer(self):
        tb = self._make_toolbox()
        tb._engine = MagicMock()
        tb._engine.get_spoken_answer.return_value = None
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

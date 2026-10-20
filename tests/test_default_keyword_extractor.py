"""The default keyword extractor must resolve from the plugin's declared dependencies."""
import unittest
from unittest.mock import patch

from ovos_plugin_manager.keywords import load_keyword_extract_plugin

from ovos_ddg_plugin import DuckDuckGoRetrievalEngine

DEFAULT_EXTRACTOR = "ovos-rake-keyword-extractor"
ABSTRACT = "Alan Mathison Turing was an English mathematician and logician."


class TestDefaultKeywordExtractor(unittest.TestCase):

    def test_default_extractor_plugin_is_installed(self):
        cls = load_keyword_extract_plugin(DEFAULT_EXTRACTOR)
        self.assertIsNotNone(cls)
        self.assertEqual(cls().extract("who was Alan Turing", lang="en-US"),
                         {"alan turing": 4.0})

    def test_conversational_question_is_answered_through_default_extractor(self):
        requested = []

        def fake_fetch(query, lang):
            requested.append(query)
            return {"AbstractText": ABSTRACT} if query == "alan turing" else {}

        engine = DuckDuckGoRetrievalEngine()
        with patch.object(engine, "_fetch", side_effect=fake_fetch):
            answers = engine.query("who was Alan Turing", lang="en-US")
        self.assertEqual(answers, [(ABSTRACT, 0.9)])
        self.assertEqual(requested[-1], "alan turing")

    def test_highest_scoring_keyword_is_fetched(self):
        requested = []

        def fake_fetch(query, lang):
            requested.append(query)
            return {}

        engine = DuckDuckGoRetrievalEngine()
        with patch.object(engine, "_fetch", side_effect=fake_fetch):
            engine.query("who painted the Mona Lisa", lang="en-US")
        self.assertEqual(requested[-1], "mona lisa")

    def test_unsupported_language_yields_no_answer(self):
        engine = DuckDuckGoRetrievalEngine()
        with patch.object(engine, "_fetch", return_value={}):
            self.assertEqual(engine.query("who painted the Mona Lisa", lang="kab"), [])


if __name__ == "__main__":
    unittest.main()

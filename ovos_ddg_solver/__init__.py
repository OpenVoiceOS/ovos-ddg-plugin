# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import datetime
import os.path
from typing import Optional, List, Tuple, Dict, Any

import requests
from langcodes import closest_match
from ovos_config import Configuration
from ovos_date_parser import nice_date
from ovos_plugin_manager.keywords import load_keyword_extract_plugin
from ovos_plugin_manager.templates.keywords import KeywordExtractor
from ovos_plugin_manager.templates.agent_tools import AgentTool, ToolBox, ToolOutput, ToolArguments
from ovos_plugin_manager.templates.agents import RetrievalEngine
from ovos_utils.log import LOG
from padacioso import IntentContainer
from padacioso.bracket_expansion import expand_parentheses
from pydantic import Field
from quebra_frases import sentence_tokenize


class DuckDuckGoRetrievalEngine(RetrievalEngine):
    # DDG reverses the standard lang-region order in its locale codes
    LOCALE_MAPPING = {'ar-XA': 'xa-ar', 'en-XA': 'xa-en', 'es-AR': 'ar-es', 'en-AU': 'au-en', 'de-AT': 'at-de',
                      'fr-BE': 'be-fr', 'nl-BE': 'be-nl', 'pt-BR': 'br-pt', 'bg-BG': 'bg-bg', 'en-CA': 'ca-en',
                      'fr-CA': 'ca-fr', 'ca-KI': 'ct-ca', 'es-CL': 'cl-es', 'zh-CN': 'cn-zh', 'es-CO': 'co-es',
                      'hr-HR': 'hr-hr', 'cs-CZ': 'cz-cs', 'da-DK': 'dk-da', 'et-EE': 'ee-et', 'fi-FI': 'fi-fi',
                      'fr-FR': 'fr-fr', 'de-DE': 'de-de', 'el-GR': 'gr-el', 'tzh-HK': 'hk-tzh', 'hu-HU': 'hu-hu',
                      'en-IN': 'in-en', 'id-ID': 'id-id', 'en-ID': 'id-en', 'en-IE': 'ie-en', 'he-IL': 'il-he',
                      'it-IT': 'it-it', 'jp-JP': 'jp-jp', 'kr-KR': 'kr-kr', 'lv-LV': 'lv-lv', 'lt-LT': 'lt-lt',
                      'es-XL': 'xl-es', 'ms-MY': 'my-ms', 'en-MY': 'my-en', 'es-MX': 'mx-es', 'nl-NL': 'nl-nl',
                      'en-NZ': 'nz-en', 'no-NO': 'no-no', 'es-PE': 'pe-es', 'en-PH': 'ph-en', 'fil-PH': 'ph-tl',
                      'pl-PL': 'pl-pl', 'pt-PT': 'pt-pt', 'ro-RO': 'ro-ro', 'ru-RU': 'ru-ru', 'en-SG': 'sg-en',
                      'sk-SK': 'sk-sk', 'sl-SL': 'sl-sl', 'en-ZA': 'za-en', 'es-ES': 'es-es', 'sv-SE': 'se-sv',
                      'de-CH': 'ch-de', 'fr-CH': 'ch-fr', 'it-CH': 'ch-it', 'tzh-TW': 'tw-tzh', 'th-TH': 'th-th',
                      'tr-TR': 'tr-tr', 'uk-UA': 'ua-uk', 'en-GB': 'uk-en', 'en-US': 'us-en', 'es-UE': 'ue-es',
                      'es-VE': 've-es', 'vi-VN': 'vn-vi'}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config=config)
        self._kword_extractors: Dict[str, KeywordExtractor] = {}
        self._intent_matchers: Dict[str, IntentContainer] = {}
        self._load_intents()

    # ------------------------------------------------------------------
    # Intent matching for infobox fields
    # ------------------------------------------------------------------

    def _load_intents(self) -> None:
        files = [
            "known_for.intent", "resting_place.intent", "born.intent", "died.intent",
            "children.intent", "alma_mater.intent", "age_at_death.intent",
            "education.intent", "fields.intent", "thesis.intent", "official_website.intent",
        ]
        locale_dir = f"{os.path.dirname(__file__)}/locale"
        if not os.path.isdir(locale_dir):
            return
        for lang in os.listdir(locale_dir):
            for fn in files:
                path = f"{locale_dir}/{lang}/{fn}"
                if not os.path.isfile(path):
                    continue
                samples = []
                with open(path) as f:
                    for line in f.read().split("\n"):
                        if not line.strip() or line.startswith("#"):
                            continue
                        samples += expand_parentheses(line) if "(" in line else [line]
                self._register_intent(fn.split(".intent")[0], samples, lang)

    def _register_intent(self, key: str, samples: List[str], lang: str) -> None:
        lang = lang.split("-")[0]
        if lang not in self._intent_matchers:
            self._intent_matchers[lang] = IntentContainer()
        self._intent_matchers[lang].add_intent(key, samples)

    def _match_infobox_intent(self, utterance: str, lang: str) -> Tuple[Optional[str], str]:
        """Return (infobox_field, keyword) if the utterance targets a specific infobox field."""
        lang = lang.split("-")[0]
        if lang not in self._intent_matchers:
            return None, utterance
        match = self._intent_matchers[lang].calc_intent(utterance)
        kw = match.get("entities", {}).get("query")
        if kw:
            LOG.debug(f"DDG infobox intent: {match['name']} query={kw} conf={match['conf']}")
            return match["name"], kw
        return None, utterance

    # ------------------------------------------------------------------
    # DDG API helpers
    # ------------------------------------------------------------------

    def _keyword_extractor(self, lang: str) -> Optional[KeywordExtractor]:
        if lang not in self._kword_extractors:
            plugin_id = self.config.get("keyword_extractor") or "ovos-rake-keyword-extractor"
            cls = load_keyword_extract_plugin(plugin_id)
            if cls is None:
                return None
            self._kword_extractors[lang] = cls()
        return self._kword_extractors[lang]

    def _fetch(self, query: str, lang: str) -> Dict[str, Any]:
        """Raw DDG Instant Answers API call."""
        best_lang, distance = closest_match(lang, self.LOCALE_MAPPING)
        if distance > 10:
            LOG.debug(f"Unsupported DDG locale: {lang}")
            return {}
        try:
            return requests.get("https://api.duckduckgo.com",
                                params={"format": "json",
                                        "kl": self.LOCALE_MAPPING[best_lang],
                                        "q": query}).json()
        except Exception:
            return {}

    def _search(self, query: str, lang: str) -> Dict[str, Any]:
        """Fetch DDG data, falling back to keyword extraction if the direct query returns nothing."""
        data = self._fetch(query, lang)
        if data.get("AbstractText"):
            return data
        kwx = self._keyword_extractor(lang)
        if kwx:
            keywords = kwx.extract(query, lang=lang)
            if keywords:
                kw = max(keywords)
                LOG.debug(f"DDG keyword fallback: {kw!r}")
                return self._fetch(kw, lang)
        return {}

    def get_infobox(self, query: str, lang: Optional[str] = None) -> Tuple[Dict[str, Any], List[str]]:
        """Return (infobox_dict, related_topics) for a query."""
        lang = lang or Configuration().get("lang", "en-us")
        data = self._search(query, lang)
        related_topics = [t.get("Text") for t in data.get("RelatedTopics", [])]
        infobox: Dict[str, Any] = {}
        for entry in (data.get("Infobox") or {}).get("content", []):
            k = entry["label"].lower().strip()
            v = entry["value"]
            try:
                if k in ("born", "died") and isinstance(v, dict) and "time" in v:
                    dt = datetime.datetime.strptime(v["time"], "+%Y-%m-%dT%H:%M:%SZ")
                    infobox[k] = nice_date(dt, lang=lang)
                else:
                    infobox[k] = v
            except Exception:
                continue
        return infobox, related_topics

    # ------------------------------------------------------------------
    # RetrievalEngine interface
    # ------------------------------------------------------------------

    def query(self, query: str, lang: Optional[str] = None, k: int = 3) -> List[Tuple[str, float]]:
        """Return up to k (answer, score) tuples from DuckDuckGo."""
        lang = lang or Configuration().get("lang", "en-us")
        intent, kw = self._match_infobox_intent(query, lang)
        if intent:
            infobox = self.get_infobox(kw, lang=lang)[0]
            answer = infobox.get(intent)
            if answer:
                return [(answer, 0.9)]
        data = self._search(query, lang)
        abstract = data.get("AbstractText")
        if abstract:
            return [(s, 0.7) for s in sentence_tokenize(abstract) if s][:k]
        return []


class SearchDuckDuckGoArgs(ToolArguments):
    query: str = Field(..., description="The natural language query to look up on DuckDuckGo. Convert conversational phrasing to concise keywords (e.g. 'Stephen Hawking birthdate' not 'when was Stephen Hawking born').")
    lang: str = Field("en-us", description="BCP-47 language code for the response locale (e.g. 'en-us', 'de-de', 'pt-pt').")


class SearchDuckDuckGoOutput(ToolOutput):
    result: str = Field(..., description="The DuckDuckGo abstract or infobox answer for the query.")


class DuckDuckGoToolbox(ToolBox):
    toolbox_id = "ovos-ddg-tools"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self._engine = DuckDuckGoRetrievalEngine(config=self.config)
        super().__init__(toolbox_id=self.toolbox_id)

    def search_ddg(self, args: SearchDuckDuckGoArgs) -> SearchDuckDuckGoOutput:
        results = self._engine.query(args.query, lang=args.lang, k=1)
        return SearchDuckDuckGoOutput(result=results[0][0] if results else "")

    def discover_tools(self) -> List[AgentTool]:
        return [
            AgentTool(
                name="search_duckduckgo",
                description=(
                    "Query DuckDuckGo for factual answers: encyclopaedic information, "
                    "infobox facts (birthdate, death, known for, alma mater, …), and text summaries. "
                    "No API key required. Send queries as concise keywords."
                ),
                argument_schema=SearchDuckDuckGoArgs,
                output_schema=SearchDuckDuckGoOutput,
                tool_call=self.search_ddg,
            )
        ]


if __name__ == "__main__":
    LOG.set_level("DEBUG")

    d = DuckDuckGoRetrievalEngine()
    print(d.query("Quem foi Bartolomeu Dias", lang="pt"))
    print(d.get_infobox("Stephen Hawking", lang="pt")[0])
    print(d.query("who is Isaac Newton", lang="en"))

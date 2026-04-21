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
from pprint import pformat
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
    # DDG is weird and has lang-codes lang/region "backwards"
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
        self.kword_extractors: Dict[str, KeywordExtractor] = {}
        self.intent_matchers: Dict[str, IntentContainer] = {}
        self.register_from_file()

    def get_keyword_extractor(self, lang: str) -> Optional[KeywordExtractor]:
        if lang not in self.kword_extractors:
            kw_plugin: str = self.config.get("keyword_extractor") or "ovos-rake-keyword-extractor"
            kword_extractor_class = load_keyword_extract_plugin(kw_plugin)
            if not kword_extractor_class:
                return None
            self.kword_extractors[lang] = kword_extractor_class()
        return self.kword_extractors[lang]

    def register_infobox_intent(self, key: str, samples: List[str], lang: str) -> None:
        lang = lang.split("-")[0]
        if lang not in self.intent_matchers:
            self.intent_matchers[lang] = IntentContainer()
        self.intent_matchers[lang].add_intent(key.split(".intent")[0], samples)

    def match_infobox_intent(self, utterance: str, lang: str) -> Tuple[Optional[str], str]:
        lang = lang.split("-")[0]
        if lang not in self.intent_matchers:
            return None, utterance
        matcher: IntentContainer = self.intent_matchers[lang]
        match = matcher.calc_intent(utterance)
        kw = match.get("entities", {}).get("query")
        intent = None
        if kw:
            intent = match["name"]
            LOG.debug(f"DDG Intent: {intent} Query: {kw} - Confidence: {match['conf']}")
        else:
            LOG.debug(f"Could not match intent for '{lang}' from '{utterance}'")
        return intent, kw or utterance

    def register_from_file(self) -> None:
        files = [
            "known_for.intent",
            "resting_place.intent",
            "born.intent",
            "died.intent",
            "children.intent",
            "alma_mater.intent",
            "age_at_death.intent",
            "education.intent",
            "fields.intent",
            "thesis.intent",
            "official_website.intent"
        ]
        locale_dir = f"{os.path.dirname(__file__)}/locale"
        if not os.path.isdir(locale_dir):
            return
        for lang in os.listdir(locale_dir):
            for fn in files:
                filename = f"{locale_dir}/{lang}/{fn}"
                if not os.path.isfile(filename):
                    LOG.warning(f"{filename} not found for '{lang}'")
                    continue
                samples = []
                with open(filename) as f:
                    for line in f.read().split("\n"):
                        if not line.strip() or line.startswith("#"):
                            continue
                        if "(" in line:
                            samples += expand_parentheses(line)
                        else:
                            samples.append(line)
                self.register_infobox_intent(fn.split(".intent")[0], samples, lang)

    def get_infobox(self, query: str,
                    lang: Optional[str] = None,
                    units: Optional[str] = None) -> Tuple[Dict[str, Any], List[str]]:
        time_keys = ["died", "born"]
        data = self.extract_and_search(query, lang=lang, units=units)
        related_topics = [t.get("Text") for t in data.get("RelatedTopics", [])]
        infobox = {}
        infodict = data.get("Infobox") or {}
        for entry in infodict.get("content", []):
            k = entry["label"].lower().strip()
            v = entry["value"]
            try:
                if k in time_keys and "time" in v:
                    dt = datetime.datetime.strptime(v["time"], "+%Y-%m-%dT%H:%M:%SZ")
                    infobox[k] = nice_date(dt, lang=lang or Configuration().get("lang", "en-us"))
                else:
                    infobox[k] = v
            except Exception:
                continue
        return infobox, related_topics

    def extract_and_search(self, query: str,
                           lang: Optional[str] = None,
                           units: Optional[str] = None) -> Dict[str, Any]:
        data = self.get_data(query, lang=lang, units=units)
        if data.get("AbstractText"):
            return data
        kwx = self.get_keyword_extractor(lang)
        if kwx:
            keywords = kwx.extract(query, lang=lang)
            if keywords:
                kw = max(keywords)
                LOG.debug(f"DDG search: {kw}")
                return self.get_data(kw, lang=lang, units=units)
        return {}

    def get_data(self, query: str,
                 lang: Optional[str] = None,
                 units: Optional[str] = None) -> Dict[str, Any]:
        units = units or Configuration().get("system_unit", "metric")
        lang = lang or Configuration().get("lang", "en-us")
        best_lang, distance = closest_match(lang, self.LOCALE_MAPPING)
        if distance > 10:
            LOG.debug(f"Unsupported DDG locale: {lang}")
            return {}
        try:
            data = requests.get("https://api.duckduckgo.com",
                                params={"format": "json",
                                        "kl": self.LOCALE_MAPPING[best_lang],
                                        "q": query}).json()
        except Exception:
            return {}
        return data

    def get_image(self, query: str,
                  lang: Optional[str] = None,
                  units: Optional[str] = None) -> str:
        data = self.extract_and_search(query, lang, units)
        image = data.get("Image") or f"{os.path.dirname(__file__)}/logo.png"
        if image.startswith("/"):
            image = "https://duckduckgo.com" + image
        return image

    def get_spoken_answer(self, query: str,
                          lang: Optional[str] = None,
                          units: Optional[str] = None) -> Optional[str]:
        lang = lang or Configuration().get("lang", "en-us")
        intent, query = self.match_infobox_intent(query, lang=lang)
        LOG.info(f"DDG intent: {intent} keyword: {query}")
        if intent not in ["question", None]:
            infobox = self.get_infobox(query, lang=lang, units=units)[0] or {}
            LOG.debug(f"Parsing infobox: {infobox}")
            answer = infobox.get(intent)
            if answer:
                return answer
        data = self.extract_and_search(query, lang=lang, units=units)
        return data.get("AbstractText")

    def get_expanded_answer(self, query: str,
                            lang: Optional[str] = None,
                            units: Optional[str] = None) -> List[Dict[str, str]]:
        img = self.get_image(query, lang=lang, units=units)
        lang = lang or Configuration().get("lang", "en-us")
        intent, query = self.match_infobox_intent(query, lang)
        if intent and intent not in ["question"]:
            infobox = self.get_infobox(query, lang=lang, units=units)[0] or {}
            LOG.debug(pformat(infobox))
            answer = infobox.get(intent)
            if answer:
                return [{"title": query, "summary": answer, "img": img}]
        LOG.debug("DDG couldn't match infobox section, using text summary")
        data = self.extract_and_search(query, lang=lang, units=units)
        return [{"title": query, "summary": s, "img": img}
                for s in sentence_tokenize(data.get("AbstractText", "")) if s]

    def query(self, query: str, lang: Optional[str] = None, k: int = 3) -> List[Tuple[str, float]]:
        """Return up to k (answer, score) tuples from DuckDuckGo."""
        units = self.config.get("units") or Configuration().get("system_unit", "metric")
        lang = lang or Configuration().get("lang", "en-us")
        intent, kw = self.match_infobox_intent(query, lang=lang)
        if intent and intent not in ["question"]:
            infobox = self.get_infobox(kw, lang=lang, units=units)[0] or {}
            answer = infobox.get(intent)
            if answer:
                return [(answer, 0.9)]
        data = self.extract_and_search(query, lang=lang, units=units)
        abstract = data.get("AbstractText")
        if abstract:
            sentences = sentence_tokenize(abstract)
            results = [(s, 0.7) for s in sentences if s]
            return results[:k]
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
        """Query DuckDuckGo and return the best answer."""
        answer = self._engine.get_spoken_answer(args.query, lang=args.lang)
        return SearchDuckDuckGoOutput(result=answer or "")

    def discover_tools(self) -> List[AgentTool]:
        return [
            AgentTool(
                name="search_duckduckgo",
                description=(
                    "Query DuckDuckGo for factual answers: encyclopaedic information, "
                    "infobox facts (birthdate, death, known for, alma mater, …), and text summaries. "
                    "Works offline — no API key required. Send queries as concise keywords."
                ),
                argument_schema=SearchDuckDuckGoArgs,
                output_schema=SearchDuckDuckGoOutput,
                tool_call=self.search_ddg,
            )
        ]


if __name__ == "__main__":
    LOG.set_level("DEBUG")

    d = DuckDuckGoRetrievalEngine()

    ans = d.get_spoken_answer("Quem foi Bartolomeu Dias", lang="pt")
    print(ans)

    info = d.get_infobox("Stephen Hawking", lang="pt")[0]
    from pprint import pprint
    pprint(info)

    for sentence in d.get_expanded_answer("who is Isaac Newton", lang="en"):
        print(sentence["title"])
        print(sentence["summary"])
        print(sentence.get("img"))

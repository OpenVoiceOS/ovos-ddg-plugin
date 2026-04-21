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
    """Retrieval engine backed by the DuckDuckGo Instant Answers API.

    Supports text abstracts, structured infobox lookups, and image URLs.
    No API key required. Locale is matched automatically from any BCP-47
    language tag to the nearest DDG locale code.

    Config keys (all optional):
        keyword_extractor: OPM plugin ID used when a direct query returns no
            abstract (default: "ovos-rake-keyword-extractor").
    """

    # DDG reverses the standard BCP-47 lang-region order in its locale codes
    # (e.g. "en-US" → "us-en", "de-DE" → "de-de").
    LOCALE_MAPPING: Dict[str, str] = {
        'ar-XA': 'xa-ar', 'en-XA': 'xa-en', 'es-AR': 'ar-es', 'en-AU': 'au-en', 'de-AT': 'at-de',
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
        'es-VE': 've-es', 'vi-VN': 'vn-vi',
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config)
        self._kword_extractors: Dict[str, KeywordExtractor] = {}
        self._intent_matchers: Dict[str, IntentContainer] = {}
        self._load_intents()

    # ------------------------------------------------------------------
    # Infobox intent matching
    # ------------------------------------------------------------------

    def _load_intents(self) -> None:
        """Load Padacioso infobox intent files from the bundled locale directory."""
        files = [
            "known_for.intent", "resting_place.intent", "born.intent", "died.intent",
            "children.intent", "alma_mater.intent", "age_at_death.intent",
            "education.intent", "fields.intent", "thesis.intent", "official_website.intent",
        ]
        locale_dir = os.path.join(os.path.dirname(__file__), "locale")
        if not os.path.isdir(locale_dir):
            return
        for lang in os.listdir(locale_dir):
            for fn in files:
                path = os.path.join(locale_dir, lang, fn)
                if not os.path.isfile(path):
                    continue
                samples: List[str] = []
                with open(path) as f:
                    for line in f.read().splitlines():
                        if not line.strip() or line.startswith("#"):
                            continue
                        samples += expand_parentheses(line) if "(" in line else [line]
                self._register_intent(fn.removesuffix(".intent"), samples, lang)

    def _register_intent(self, key: str, samples: List[str], lang: str) -> None:
        """Add intent samples for *key* to the matcher for *lang* (base code only)."""
        lang = lang.split("-")[0]
        if lang not in self._intent_matchers:
            self._intent_matchers[lang] = IntentContainer()
        self._intent_matchers[lang].add_intent(key, samples)

    def _match_infobox_intent(self, utterance: str, lang: str) -> Tuple[Optional[str], str]:
        """Try to match *utterance* against a known infobox field intent.

        Returns:
            (field_name, keyword) when a match is found, or (None, utterance) otherwise.
        """
        lang = lang.split("-")[0]
        if lang not in self._intent_matchers:
            return None, utterance
        match = self._intent_matchers[lang].calc_intent(utterance)
        kw: Optional[str] = match.get("entities", {}).get("query")
        if kw:
            LOG.debug(f"DDG infobox intent: {match['name']} query={kw!r} conf={match['conf']:.2f}")
            return match["name"], kw
        return None, utterance

    # ------------------------------------------------------------------
    # DDG API
    # ------------------------------------------------------------------

    def _keyword_extractor(self, lang: str) -> Optional[KeywordExtractor]:
        """Return a cached keyword extractor for *lang*, loading it on first use."""
        if lang not in self._kword_extractors:
            plugin_id: str = self.config.get("keyword_extractor") or "ovos-rake-keyword-extractor"
            cls = load_keyword_extract_plugin(plugin_id)
            if cls is None:
                LOG.warning(f"DDG keyword extractor plugin not found: {plugin_id!r}")
                return None
            self._kword_extractors[lang] = cls()
        return self._kword_extractors[lang]

    def _fetch(self, query: str, lang: str) -> Dict[str, Any]:
        """Send one raw request to the DDG Instant Answers API.

        Returns an empty dict on network errors or unsupported locales.
        """
        best_lang, distance = closest_match(lang, self.LOCALE_MAPPING)
        if distance > 10:
            LOG.debug(f"DDG: no supported locale for {lang!r}")
            return {}
        try:
            return requests.get(
                "https://api.duckduckgo.com",
                params={"format": "json", "kl": self.LOCALE_MAPPING[best_lang], "q": query},
            ).json()
        except Exception:
            return {}

    def _search(self, query: str, lang: str) -> Dict[str, Any]:
        """Fetch DDG data, falling back to keyword extraction when the direct query yields no abstract."""
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_image(self, query: str, lang: Optional[str] = None) -> Optional[str]:
        """Return a DDG image URL for *query*, or ``None`` if DDG has no image.

        Args:
            query: Search term.
            lang: BCP-47 language tag. Falls back to the OVOS system language.
        """
        lang = lang or Configuration().get("lang", "en-us")
        data = self._search(query, lang)
        image: str = data.get("Image", "")
        if not image:
            return None
        if image.startswith("/"):
            image = "https://duckduckgo.com" + image
        return image

    def get_infobox(self, query: str, lang: Optional[str] = None) -> Tuple[Dict[str, Any], List[str]]:
        """Return structured infobox data for *query*.

        Date fields (``born``, ``died``) are formatted via ``ovos-date-parser``
        into a natural-language string for *lang*.

        Args:
            query: Entity name or search term.
            lang: BCP-47 language tag. Falls back to the OVOS system language.

        Returns:
            A tuple of:
            - ``infobox``: mapping of lowercased field labels → values.
            - ``related_topics``: list of related-topic text snippets.
        """
        lang = lang or Configuration().get("lang", "en-us")
        data = self._search(query, lang)
        related_topics: List[str] = [t.get("Text") for t in data.get("RelatedTopics", [])]
        infobox: Dict[str, Any] = {}
        for entry in (data.get("Infobox") or {}).get("content", []):
            k: str = entry["label"].lower().strip()
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
        """Return up to *k* ``(answer, score)`` tuples from DuckDuckGo.

        Scoring:
        - Infobox field match → single result at 0.9.
        - Abstract sentences → 0.9, 0.8, 0.7 … in order of appearance.

        Args:
            query: Natural-language or keyword query.
            lang: BCP-47 language tag. Falls back to the OVOS system language.
            k: Maximum number of results to return.
        """
        lang = lang or Configuration().get("lang", "en-us")
        intent, kw = self._match_infobox_intent(query, lang)
        if intent:
            infobox = self.get_infobox(kw, lang=lang)[0]
            answer = infobox.get(intent)
            if answer:
                return [(answer, 0.9)]
        data = self._search(query, lang)
        abstract: str = data.get("AbstractText", "")
        if abstract:
            return [(s, 0.9 - idx * 0.1) for idx, s in enumerate(sentence_tokenize(abstract)) if s.strip()][:k]
        return []


# ---------------------------------------------------------------------------
# Pydantic schemas for agent tools
# ---------------------------------------------------------------------------

class DDGImageArgs(ToolArguments):
    query: str = Field(..., description="Topic or entity to fetch an image for (e.g. 'Stephen Hawking', 'Eiffel Tower').")
    lang: str = Field("en-us", description="BCP-47 language code for the locale.")


class DDGImageOutput(ToolOutput):
    url: Optional[str] = Field(None, description="Image URL from DuckDuckGo, or null if none is available.")


class SearchDuckDuckGoArgs(ToolArguments):
    query: str = Field(..., description="Query to look up on DuckDuckGo. Prefer concise keywords over conversational phrasing (e.g. 'Stephen Hawking birthdate' not 'when was Stephen Hawking born').")
    lang: str = Field("en-us", description="BCP-47 language code for the response locale (e.g. 'en-us', 'de-de', 'pt-pt').")


class SearchDuckDuckGoOutput(ToolOutput):
    result: str = Field(..., description="Best DuckDuckGo answer — an infobox field value or the first abstract sentence.")


# ---------------------------------------------------------------------------
# Toolbox
# ---------------------------------------------------------------------------

class DuckDuckGoToolbox(ToolBox):
    """Agent toolbox exposing DuckDuckGo search and image lookup as callable tools."""

    toolbox_id = "ovos-ddg-tools"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = config or {}
        self._engine = DuckDuckGoRetrievalEngine(config=self.config)
        super().__init__(toolbox_id=self.toolbox_id)

    def search_ddg(self, args: SearchDuckDuckGoArgs) -> SearchDuckDuckGoOutput:
        """Return the single best DuckDuckGo answer for *args.query*."""
        results = self._engine.query(args.query, lang=args.lang, k=1)
        return SearchDuckDuckGoOutput(result=results[0][0] if results else "")

    def ddg_image(self, args: DDGImageArgs) -> DDGImageOutput:
        """Return a DDG image URL for *args.query*, or ``null`` if none is available."""
        return DDGImageOutput(url=self._engine.get_image(args.query, lang=args.lang))

    def discover_tools(self) -> List[AgentTool]:
        """Declare the two tools provided by this toolbox."""
        return [
            AgentTool(
                name="search_duckduckgo",
                description=(
                    "Query DuckDuckGo for factual answers: encyclopaedic information, "
                    "infobox facts (birthdate, death, known for, alma mater, …), and text summaries. "
                    "No API key required. Prefer concise keywords over conversational phrasing."
                ),
                argument_schema=SearchDuckDuckGoArgs,
                output_schema=SearchDuckDuckGoOutput,
                tool_call=self.search_ddg,
            ),
            AgentTool(
                name="duckduckgo_image",
                description="Fetch an image URL for a topic or entity from DuckDuckGo. Returns null if DDG has no image.",
                argument_schema=DDGImageArgs,
                output_schema=DDGImageOutput,
                tool_call=self.ddg_image,
            ),
        ]


if __name__ == "__main__":
    LOG.set_level("DEBUG")

    d = DuckDuckGoRetrievalEngine()
    print(d.query("Quem foi Bartolomeu Dias", lang="pt"))
    print(d.get_infobox("Stephen Hawking", lang="pt")[0])
    print(d.query("who is Isaac Newton", lang="en"))

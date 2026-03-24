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
#
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
from ovos_plugin_manager.templates.solvers import QuestionSolver
from ovos_utils.log import LOG
from padacioso import IntentContainer
from padacioso.bracket_expansion import expand_parentheses
from quebra_frases import sentence_tokenize


class DuckDuckGoSolver(QuestionSolver):
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
        super().__init__(config,
                         enable_tx=False,
                         priority=75)
        self.kword_extractors: Dict[str, KeywordExtractor] = {}
        self.intent_matchers: Dict[str, IntentContainer] = {}
        self.register_from_file()

    # utils to extract keywords from text
    def get_keyword_extractor(self, lang: str) -> Optional[KeywordExtractor]:
        """
        Get a keyword extractor instance for the given language, creating and caching a plugin instance when needed.

        Returns:
            KeywordExtractor | None: A `KeywordExtractor` configured for `lang`, or `None` if the configured plugin cannot be loaded.
        """
        if lang not in self.kword_extractors:
            kw_plugin: str = self.config.get("keyword_extractor") or "ovos-rake-keyword-extractor"
            kword_extractor_class = load_keyword_extract_plugin(kw_plugin)
            if not kword_extractor_class:
                return None
            kword_extractor = kword_extractor_class()
            if self.enable_tx:  # share objects to avoid re-init
                kword_extractor._detector = self.detector
                kword_extractor._translator = self.translator
                kword_extractor_class.enable_tx = self.enable_tx
            self.kword_extractors[lang] = kword_extractor
        return self.kword_extractors[lang]

    def register_infobox_intent(self, key: str, samples: List[str], lang: str) -> None:
        """Register infobox intents for a given language.

        Args:
            key: The key identifying the intent.
            samples: A list of intent samples.
            lang: Language code.
        """
        lang = lang.split("-")[0]
        if lang not in self.intent_matchers:
            self.intent_matchers[lang] = IntentContainer()
        self.intent_matchers[lang].add_intent(key.split(".intent")[0], samples)

    def match_infobox_intent(self, utterance: str, lang: str) -> Tuple[Optional[str], str]:
        """Match infobox intents in an utterance.

        Args:
            utterance: The utterance to match intents from.
            lang: Language code.

        Returns:
            A tuple of the matched intent and the extracted keyword or original utterance.
        """
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
        """Register internal Padacioso intents for DuckDuckGo."""
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
        for lang in os.listdir(f"{os.path.dirname(__file__)}/locale"):
            for fn in files:
                filename = f"{os.path.dirname(__file__)}/locale/{lang}/{fn}"
                if not os.path.isfile(filename):
                    LOG.warning(f"{filename} not found for '{lang}'")
                    continue
                samples = []
                with open(filename) as f:
                    for l in f.read().split("\n"):
                        if not l.strip() or l.startswith("#"):
                            continue
                        if "(" in l:
                            samples += expand_parentheses(l)
                        else:
                            samples.append(l)
                self.register_infobox_intent(fn.split(".intent")[0], samples, lang)

    def get_infobox(self, query: str,
                    lang: Optional[str] = None,
                    units: Optional[str] = None) -> Tuple[Dict[str, Any], List[str]]:
        """Retrieve infobox information and related topics for a query.

        Args:
            query: The search query.
            lang: Language code.
            units: Unit system (e.g., 'metric').

        Returns:
            A tuple of infobox data and related topics.
        """
        time_keys = ["died", "born"]
        data = self.extract_and_search(query, lang=lang, units=units)  # handles translation
        # parse infobox
        related_topics = [t.get("Text") for t in data.get("RelatedTopics", [])]
        infobox = {}
        infodict = data.get("Infobox") or {}
        for entry in infodict.get("content", []):
            k = entry["label"].lower().strip()
            v = entry["value"]
            try:
                if k in time_keys and "time" in v:
                    dt = datetime.datetime.strptime(v["time"], "+%Y-%m-%dT%H:%M:%SZ")
                    infobox[k] = nice_date(dt, lang=lang or self.default_lang)
                else:
                    infobox[k] = v
            except:  # probably a LF error
                continue
        return infobox, related_topics

    def extract_and_search(self, query: str,
                           lang: Optional[str] = None,
                           units: Optional[str] = None) -> Dict[str, Any]:
        """Extract search term from query and perform search.

        Args:
            query: The search query.
            lang: Language code.
            units: Unit system (e.g., 'metric').

        Returns:
            The search result data.
        """
        data = self.get_data(query, lang=lang, units=units)
        if data.get("AbstractText"):
            # direct match without extracting sub-keyword
            return data
        # extract the best keyword
        kwx = self.get_keyword_extractor(lang)
        keywords = kwx.extract(query, lang=lang)
        if keywords:
            kw = max(keywords)
            LOG.debug(f"DDG search: {kw}")
            return self.get_data(kw, lang=lang, units=units)
        return {}

    ########################################################
    # abstract methods all solver plugins need to implement
    def get_data(self, query: str,
                 lang: Optional[str] = None,
                 units: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve data from DuckDuckGo API.

        Args:
            query: The search query.
            lang: Language code.
            units: Unit system (e.g., 'metric').

        Returns:
            The search result data.
        """
        units = units or Configuration().get("system_unit", "metric")
        lang = lang or self.default_lang
        best_lang, distance = closest_match(lang, self.LOCALE_MAPPING)
        if distance > 10:
            LOG.debug(f"Unsupported DDG locale: {lang}")
            return {}

        # duck duck go api request
        try:
            data = requests.get("https://api.duckduckgo.com",
                                params={"format": "json",
                                        "kl": self.LOCALE_MAPPING[best_lang],
                                        "q": query}).json()
        except:
            return {}
        return data

    def get_image(self, query: str,
                  lang: Optional[str] = None,
                  units: Optional[str] = None) -> str:
        """Retrieve image URL for a query.

        Args:
            query: The search query.
            lang: Language code.
            units: Unit system (e.g., 'metric').

        Returns:
            The image URL.
        """
        data = self.extract_and_search(query, lang, units)
        image = data.get("Image") or f"{os.path.dirname(__file__)}/logo.png"
        if image.startswith("/"):
            image = "https://duckduckgo.com" + image
        return image

    def get_spoken_answer(self, query: str,
                          lang: Optional[str] = None,
                          units: Optional[str] = None) -> str:
        """Retrieve spoken answer for a query.

        Args:
            query: The search query.
            lang: Language code.
            units: Unit system (e.g., 'metric').

        Returns:
            The spoken answer.
        """
        lang = lang or self.default_lang
        # match an infobox field with some basic regexes
        # (primitive intent parsing)
        intent, query = self.match_infobox_intent(query, lang=lang)
        LOG.info(f"DDG intent: {intent} keyword: {query}")
        if intent not in ["question"]:
            infobox = self.get_infobox(query, lang=lang, units=units)[0] or {}
            LOG.debug(f"Parsing infobox: {infobox}")
            answer = infobox.get(intent)
            if answer:
                return answer

        # return summary
        data = self.extract_and_search(query, lang=lang, units=units)
        return data.get("AbstractText")

    def get_expanded_answer(self, query: str,
                            lang: Optional[str] = None,
                            units: Optional[str] = None) -> List[Dict[str, str]]:
        """
        query assured to be in self.default_lang
        return a list of ordered steps to expand the answer, eg, "tell me more"

        {
            "title": "optional",
            "summary": "speak this",
            "img": "optional/path/or/url
        }
        :return:
        """
        img = self.get_image(query, lang=lang, units=units)
        lang = lang or Configuration().get("lang", "en-us")
        # match an infobox field with some basic regexes
        # (primitive intent parsing)
        intent, query = self.match_infobox_intent(query, lang)
        if intent and intent not in ["question"]:
            infobox = self.get_infobox(query, lang=lang, units=units)[0] or {}
            LOG.debug(pformat(infobox))  # pretty print infobox in debug logs
            answer = infobox.get(intent)
            if answer:
                return [{
                    "title": query,
                    "summary": answer,
                    "img": img
                }]

        LOG.debug(f"DDG couldn't match infobox section, using text summary")
        data = self.extract_and_search(query, lang=lang, units=units)
        steps = [{
            "title": query,
            "summary": s,
            "img": img
        } for s in sentence_tokenize(data.get("AbstractText", "")) if s]
        return steps


DDG_PERSONA = {
    "name": "DuckDuckGo",
    "solvers": [
        "ovos-solver-plugin-ddg",
        "ovos-solver-failure-plugin"
    ]
}

if __name__ == "__main__":
    LOG.set_level("DEBUG")

    d = DuckDuckGoSolver()

    ans = d.spoken_answer("Quem foi Bartolomeu Dias", lang="pt")
    print(ans)
    # Bartolomeu Dias, OM, OMP foi um navegador português que ficou célebre por ter sido o primeiro europeu a navegar para além do extremo sul da África, contornando o Cabo da Boa Esperança e chegando ao Oceano Índico a partir do Atlântico, abrindo o caminho marítimo para a Índia. Dele não se conhecem os antepassados, mas mercês e armas a ele outorgadas passaram a seus descendentes. Seu irmão foi Diogo Dias, também experiente navegador. Foi o principal navegador da esquadra de Pedro Álvares Cabral em 1500. As terras do Brasil, até então desconhecidas pelos portugueses, confundiram os navegadores, que pensaram tratar-se de uma ilha, a que deram o nome de "Vera Cruz".

    info = d.get_infobox("Stephen Hawking", lang="pt")[0]
    from pprint import pprint

    pprint(info)
    # {'born': 'Quinta-feira, oito de Janeiro, mil novecentos e quarenta e dois',
    #  'died': 'Quarta-feira, catorze de Março, dois mil e dezoito',
    #  'facebook profile': 'stephenhawking',
    #  'imdb id': 'nm0370071',
    #  'instance of': {'entity-type': 'item', 'id': 'Q5', 'numeric-id': 5},
    #  'official website': 'https://hawking.org.uk',
    #  'rotten tomatoes id': 'celebrity/stephen_hawking',
    #  'wikidata aliases': ['Stephen Hawking',
    #                       'Hawking',
    #                       'Stephen William Hawking',
    #                       'S. W. Hawking'],
    #  'wikidata description': 'físico teórico, cosmólogo e autor inglês (1942–2018)',
    #  'wikidata id': 'Q17714',
    #  'wikidata label': 'Stephen Hawking',
    #  'youtube channel': 'UCPyd4mR0p8zHd8Z0HvHc0fw'}


    # chunked answer, "tell me more"
    for sentence in d.long_answer("who is Isaac Newton", lang="en"):
        print(sentence["title"])
        print(sentence["summary"])
        print(sentence.get("img"))

        # who is Isaac Newton
        # Sir Isaac Newton was an English polymath active as a mathematician, physicist, astronomer, alchemist, theologian, author, and inventor.
        # https://duckduckgo.com/i/401ff0bf4dfa0847.jpg

        # who is Isaac Newton
        # He was a key figure in the Scientific Revolution and the Enlightenment that followed.
        # https://duckduckgo.com/i/401ff0bf4dfa0847.jpg

        # who is Isaac Newton
        # His book Philosophiæ Naturalis Principia Mathematica, first published in 1687, achieved the first great unification in physics and established classical mechanics.
        # https://duckduckgo.com/i/401ff0bf4dfa0847.jpg

        # who is Isaac Newton
        # Newton also made seminal contributions to optics, and shares credit with German mathematician Gottfried Wilhelm Leibniz for formulating infinitesimal calculus, though he developed calculus years before Leibniz.
        # https://duckduckgo.com/i/401ff0bf4dfa0847.jpg

        # who is Isaac Newton
        # Newton contributed to and refined the scientific method, and his work is considered the most influential in bringing forth modern science.
        # https://duckduckgo.com/i/401ff0bf4dfa0847.jpg

        # who is Isaac Newton
        # In the Principia, Newton formulated the laws of motion and universal gravitation that formed the dominant scientific viewpoint for centuries until it was superseded by the theory of relativity.
        # https://duckduckgo.com/i/401ff0bf4dfa0847.jpg

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
from ovos_spec_tools import expand
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

    # Intent name → DDG infobox key(s) to try when the intent name differs from the stored key.
    # DDG labels are normalised to underscores (spaces replaced), so "age at death" → "age_at_death".
    # Aliases are tried in order; first non-None value wins.
    FIELD_ALIASES: Dict[str, List[str]] = {
        # person
        "alma_mater":          ["education"],
        "resting_place":       ["resting_place", "burial", "burial_place"],
        "notable_awards":      ["notable_awards", "awards"],
        "doctoral_students":   ["doctoral_students", "other_academic_advisors", "notable_students", "other_notable_students"],
        "predecessor":         ["predecessor", "preceded_by"],
        "successor":           ["successor", "succeeded_by", "followed_by"],
        "notable_work":        ["notable_work", "notable_works", "works"],
        "movement":            ["movement", "style"],
        "partner":             ["partner", "partner(s)"],
        "years_active":        ["years_active", "active"],
        "net_worth":           ["net_worth", "wealth"],
        "spouse":              ["spouse", "husband", "wife"],
        "nickname":            ["nickname", "nickname(s)", "nicknames", "also_known_as"],
        # organization / company
        "founders":            ["founders", "founder(s)"],
        "founded":             ["founded", "launched"],
        "head_coach":          ["head_coach", "coach", "manager"],
        "stadium":             ["stadium", "arena", "venue", "ground"],
        "championships":       ["championships", "championship_titles", "titles", "honours"],
        "number_of_employees": ["number_of_employees", "employees"],
        "developer":           ["developer", "developers"],
        "produced_by":         ["produced_by", "producer(s)", "producers"],
        "written_by":          ["written_by", "writer(s)", "writers", "screenplay"],
        "publisher":           ["publisher", "publishers"],
        "label":               ["label", "record_label", "music_label"],
        "members":             ["members", "band_members", "lineup", "current_members"],
        "composers":           ["composers", "composer", "music_by"],
        # geographic
        "country":             ["country", "countries"],
        "designed_by":         ["designed_by", "architect"],
        # works
        "released":            ["released", "first_appeared", "first_release"],
        "other_names":         ["other_names", "other_name(s)", "also_known_as", "other names"],
        "author":              ["author", "written_by"],
        "genre":               ["genre", "genres", "genre(s)"],
        "scientific_name":     ["scientific_name", "binomial_name", "latin_name", "taxon_name"],
        "habitat":             ["habitat", "range", "distribution"],
        "conservation_status": ["conservation_status", "iucn_status", "status"],
        "platforms":           ["platforms", "platform(s)", "system_requirements"],
        "opened":              ["opened", "completed", "inaugurated", "built"],
    }

    def _load_intents(self) -> None:
        """Load Padacioso infobox intent files from the bundled locale directory."""
        files = [
            # person: biography
            "born.intent", "died.intent", "known_for.intent", "age_at_death.intent",
            "resting_place.intent", "baptised.intent", "father.intent", "mother.intent",
            "children.intent", "partner.intent", "spouse.intent", "height.intent",
            "citizenship.intent", "net_worth.intent", "nickname.intent", "age.intent",
            # person: career / academic
            "occupation.intent", "fields.intent", "institutions.intent", "alma_mater.intent",
            "education.intent", "thesis.intent", "doctoral_students.intent",
            "notable_work.intent", "notable_awards.intent", "movement.intent",
            "religion.intent", "official_website.intent", "years_active.intent",
            # person: sports
            "sport.intent", "coached_by.intent",
            # person: politics
            "political_party.intent", "predecessor.intent", "successor.intent",
            # film / media
            "director.intent", "starring.intent", "released.intent", "running_time.intent",
            "budget.intent", "box_office.intent", "produced_by.intent", "written_by.intent",
            # book / literature
            "author.intent", "publisher.intent", "genre.intent", "language.intent",
            "original_title.intent",
            # music
            "label.intent", "members.intent", "composers.intent",
            # game / software
            "developer.intent", "platforms.intent", "license.intent",
            # sports team / organization / company
            "founded.intent", "founders.intent", "industry.intent", "ceo.intent",
            "owner.intent", "revenue.intent", "head_coach.intent", "stadium.intent",
            "championships.intent", "number_of_employees.intent", "headquarters.intent",
            # geography / place
            "country.intent", "location.intent", "elevation.intent", "length.intent",
            "capital.intent", "population.intent", "area.intent",
            # buildings / structures
            "opened.intent", "designed_by.intent",
            # biology / nature
            "scientific_name.intent", "habitat.intent", "diet.intent",
            "conservation_status.intent",
            # generic
            "other_names.intent", "inventor.intent",
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
                        # Normalise possessive 's in training samples to match normalised queries.
                        line = line.replace("'s ", " ").replace("\u2019s ", " ")
                        # Expand alternations/optionals to the sample set (OVOS-INTENT-1 \u00a74);
                        # emits single-spaced samples with slots left opaque.
                        expanded = expand(line)
                        # Collapse any double-spaces produced by empty alternatives like (word|).
                        samples += [" ".join(s.split()) for s in expanded]
                self._register_intent(fn.removesuffix(".intent"), samples, lang)

    def _register_intent(self, key: str, samples: List[str], lang: str) -> None:
        """Add intent samples for *key* to the matcher for *lang* (base code only)."""
        lang = lang.split("-")[0]
        if lang not in self._intent_matchers:
            self._intent_matchers[lang] = IntentContainer()
        self._intent_matchers[lang].add_intent(key, samples)

    def _match_infobox_intents(self, utterance: str, lang: str) -> List[Tuple[str, str, str]]:
        """Return all candidate infobox intent matches for *utterance*, ordered by confidence.

        Each entry is (intent_name, keyword, entity_type) where entity_type is the
        matched slot name (e.g. "movie", "place", "person") — useful for disambiguation.
        Returns an empty list when no match is found.
        """
        lang = lang.split("-")[0]
        if lang not in self._intent_matchers:
            return []
        normalised = utterance.replace("'s ", " ").replace("\u2019s ", " ")
        candidates = []
        for match in self._intent_matchers[lang].calc_intents(normalised):
            entities = match.get("entities", {})
            kw: Optional[str] = entities.get("keyword") or next(iter(entities.values()), None)
            if not kw:
                continue
            entity_type = next(iter(entities.keys()), "keyword")
            candidates.append((match["name"], kw, entity_type, match["conf"]))
        # Highest confidence first; on ties prefer the more specific match, i.e.
        # the one that captured the shorter keyword (more of the utterance was
        # matched literally by the template, e.g. "hvem skrev manuskriptet til
        # {movie}" over the greedier "hvem skrev {book}").
        candidates.sort(key=lambda x: (x[3], -len(x[1])), reverse=True)
        LOG.debug(f"DDG infobox candidates: {[(n, kw, et) for n, kw, et, _ in candidates]}")
        return [(name, kw, entity_type) for name, kw, entity_type, _ in candidates]

    def _match_infobox_intent(self, utterance: str, lang: str) -> Tuple[Optional[str], str]:
        """Return the single best infobox intent match (backwards-compatible wrapper).

        Returns (field_name, keyword) or (None, utterance).
        """
        candidates = self._match_infobox_intents(utterance, lang)
        if candidates:
            name, kw, _ = candidates[0]
            return name, kw
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
            # Normalise spaces to underscores so keys match intent names (e.g. "age at death" → "age_at_death").
            k: str = entry["label"].lower().strip().replace(" ", "_")
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
        candidates = self._match_infobox_intents(query, lang)
        if candidates:
            # Fetch the infobox once per unique keyword (candidates are already sorted by confidence).
            infoboxes: Dict[str, Dict[str, Any]] = {}
            for intent, kw, entity_type in candidates:
                if kw not in infoboxes:
                    infoboxes[kw] = self.get_infobox(kw, lang=lang)[0]
                infobox = infoboxes[kw]
                # Try intent name directly, then registered aliases.
                answer = infobox.get(intent)
                if answer is None:
                    for alias in self.FIELD_ALIASES.get(intent, []):
                        answer = infobox.get(alias)
                        if answer is not None:
                            break
                if answer:
                    LOG.debug(f"DDG infobox hit: intent={intent!r} entity_type={entity_type!r} kw={kw!r}")
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


class DDGInfoboxArgs(ToolArguments):
    query: str = Field(..., description="Entity name to look up (e.g. 'Stephen Hawking', 'Eiffel Tower').")
    lang: str = Field("en-us", description="BCP-47 language code for the locale.")


class DDGInfoboxOutput(ToolOutput):
    infobox: Dict[str, Any] = Field(default_factory=dict, description="Structured infobox fields (e.g. born, died, known for). Empty dict if DDG has no infobox.")
    related_topics: List[str] = Field(default_factory=list, description="List of related topic snippets.")


# ---------------------------------------------------------------------------
# Toolbox
# ---------------------------------------------------------------------------

class DuckDuckGoToolbox(ToolBox):
    """Agent toolbox exposing DuckDuckGo search, infobox, and image lookup as callable tools."""

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

    def ddg_infobox(self, args: DDGInfoboxArgs) -> DDGInfoboxOutput:
        """Return the full DDG infobox and related topics for *args.query*."""
        infobox, related = self._engine.get_infobox(args.query, lang=args.lang)
        return DDGInfoboxOutput(infobox=infobox, related_topics=related)

    def discover_tools(self) -> List[AgentTool]:
        """Declare the three tools provided by this toolbox."""
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
                name="duckduckgo_infobox",
                description=(
                    "Fetch the full structured infobox for an entity from DuckDuckGo "
                    "(born, died, known for, alma mater, fields, thesis, official website, …). "
                    "Returns all available fields at once plus related topic snippets."
                ),
                argument_schema=DDGInfoboxArgs,
                output_schema=DDGInfoboxOutput,
                tool_call=self.ddg_infobox,
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

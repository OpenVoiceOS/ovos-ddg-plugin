# ovos-ddg-plugin

[![PyPI](https://img.shields.io/pypi/v/ovos-ddg-plugin)](https://pypi.org/project/ovos-ddg-plugin/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](https://www.python.org/)

DuckDuckGo Instant Answers plugin for [OpenVoiceOS](https://openvoiceos.org).

Provides factual Q&A, structured infobox lookups, and image URLs via the [DDG Instant Answers API](https://duckduckgo.com/api). No API key required.

Implements the `opm.agents.retrieval` and `opm.agents.toolbox` entry points — usable both as a standalone Python library and as an OVOS agent plugin.

---

## Installation

```bash
pip install ovos-ddg-plugin
```

> **Legacy users:** the old `ovos-ddg-solver-plugin` package remains on PyPI. New code should import from `ovos_ddg_plugin`.

---

## Standalone usage

### Text answers

```python
from ovos_ddg_plugin import DuckDuckGoRetrievalEngine

ddg = DuckDuckGoRetrievalEngine()

# Returns up to k (sentence, score) tuples, scored 0.9 → 0.8 → 0.7 …
results = ddg.query("who is Isaac Newton", lang="en-us", k=3)
for sentence, score in results:
    print(f"[{score:.1f}] {sentence}")
# [0.9] Sir Isaac Newton was an English polymath active as a mathematician …
# [0.8] He was a key figure in the Scientific Revolution …
# [0.7] His book Philosophiæ Naturalis Principia Mathematica …

# Infobox field queries return a single result at score 0.9
results = ddg.query("when was Stephen Hawking born", lang="en-us")
print(results)
# [('January 8, 1942', 0.9)]
```

### Infobox

```python
infobox, related = ddg.get_infobox("Stephen Hawking", lang="en-us")
from pprint import pprint
pprint(infobox)
# {'born': 'January 8, 1942',
#  'died': 'March 14, 2018',
#  'known_for': 'Hawking radiation, A Brief History of Time …',
#  'alma_mater': 'University of Oxford; University of Cambridge',
#  ...}
print(related[:3])
# ['A Brief History of Time', 'Cambridge University', 'General relativity']
```

### Image URL

```python
url = ddg.get_image("Eiffel Tower", lang="en-us")
print(url)
# https://duckduckgo.com/i/...jpg  (or None if DDG has no image)
```

---

## Agent tools

`DuckDuckGoToolbox` exposes three tools for OVOS agent pipelines:

| Tool | Description |
|------|-------------|
| `search_duckduckgo` | Best text answer (infobox field or abstract sentence) |
| `duckduckgo_infobox` | Full structured infobox + related topics for an entity |
| `duckduckgo_image` | Image URL for a topic, or `null` |

```python
from ovos_ddg_plugin import DuckDuckGoToolbox, SearchDuckDuckGoArgs, DDGInfoboxArgs, DDGImageArgs

tb = DuckDuckGoToolbox()

answer = tb.search_ddg(SearchDuckDuckGoArgs(query="Marie Curie birthdate", lang="en-us"))
print(answer.result)
# November 7, 1867

box = tb.ddg_infobox(DDGInfoboxArgs(query="Marie Curie", lang="en-us"))
print(box.infobox.get("born"))
# November 7, 1867
print(box.related_topics[:2])
# ['Radioactivity', 'Nobel Prize in Physics']

img = tb.ddg_image(DDGImageArgs(query="Marie Curie", lang="en-us"))
print(img.url)
# https://duckduckgo.com/i/...jpg
```

---

## Infobox intent matching

When `query()` receives a natural-language question, it runs it through a bank of [Padacioso](https://github.com/OpenVoiceOS/padacioso) intent files to check whether the question targets a specific infobox field. If a match is found the engine fetches the infobox for the extracted entity and returns the exact field value at score 0.9, bypassing the abstract entirely.

### How it works

1. **Intent detection** — `_match_infobox_intents()` calls Padacioso's `calc_intents()` (plural) and returns *all* candidate matches sorted by confidence, each as `(intent_name, keyword, entity_type)`.

2. **Entity type** — the slot name used in the intent file becomes the `entity_type` (`"person"`, `"movie"`, `"place"`, `"animal"`, etc.). This contextualises the keyword for downstream use.

3. **Infobox-based disambiguation** — when two intents tie (e.g. `length` and `running_time` both match "how long is Amazon"), `query()` walks the candidate list, fetches the DDG infobox for the extracted keyword, and returns the first candidate whose field is non-empty in the infobox. The entity type associated with the winning slot does not need to be known ahead of time — the infobox result decides.

4. **Field aliases** — DDG does not always name its keys the same as the intent. `FIELD_ALIASES` maps intent names to the list of DDG keys to try in order (e.g. `alma_mater` → `["education"]`, `resting_place` → `["resting_place", "burial", "burial_place"]`).

5. **Fallback** — if no candidate's field is present in the infobox, `query()` falls through to the full-text abstract.

### Supported intents

79 intents across 10 locales. Entity-typed slots (`{person}`, `{movie}`, `{place}`, …) are used instead of a generic `{keyword}` wherever the context constrains the entity type.

**Person / biography**

| Intent | Sample pattern |
|--------|----------------|
| `born` | when was {person} born |
| `died` | when did {person} die |
| `age` | how old is {person} |
| `age_at_death` | how old was {person} when they died |
| `known_for` | what is {person} known for |
| `father` / `mother` / `children` | who is {person} father |
| `spouse` / `partner` | who is {person} married to |
| `nickname` | what is {person} nickname |
| `citizenship` | what is {person} nationality |
| `net_worth` | what is {person} net worth |
| `height` | how tall is {person} |
| `resting_place` | where is {person} buried |
| `notable_work` | what is {person} most famous work |
| `notable_awards` | what awards has {person} won |

**Person / career & academic**

| Intent | Sample pattern |
|--------|----------------|
| `occupation` | what does {person} do |
| `fields` | what field does {person} work in |
| `institutions` | what institution does {person} work for |
| `alma_mater` | where did {person} go to university |
| `education` | where did {person} study |
| `thesis` | {person} doctoral thesis |
| `doctoral_students` | who were {person} doctoral students |
| `movement` | what movement is {person} associated with |
| `religion` | what religion is {person} |
| `official_website` | {person} official website |
| `years_active` | when was {person} active |
| `political_party` | what party does {person} belong to |
| `predecessor` / `successor` | who came before / after {person} |
| `coached_by` | who coaches {person} |
| `sport` | what sport does {person} play |

**Film / media**

| Intent | Sample pattern |
|--------|----------------|
| `director` | who directed {movie} |
| `starring` | who stars in {movie} |
| `released` | when was {movie} released |
| `running_time` | how long is {movie} |
| `budget` | what was the budget of {movie} |
| `box_office` | how much did {movie} earn |
| `produced_by` | who produced {movie} |
| `written_by` | who wrote the screenplay for {movie} |
| `composers` | who composed the music for {movie} |
| `original_title` | what is the original title of {movie} |
| `genre` | what genre is {movie} |
| `language` | what language is {book} in |

**Book / literature**

| Intent | Sample pattern |
|--------|----------------|
| `author` | who wrote {book} |
| `publisher` | who published {book} |

**Music**

| Intent | Sample pattern |
|--------|----------------|
| `label` | what record label is {artist} on |
| `members` | who are the members of {group} |

**Game / software**

| Intent | Sample pattern |
|--------|----------------|
| `developer` | who developed {software} |
| `platforms` | what platforms is {software} on |
| `license` | what license is {software} under |

**Organisation / company**

| Intent | Sample pattern |
|--------|----------------|
| `founded` | when was {organization} founded |
| `founders` | who founded {organization} |
| `industry` | what industry is {company} in |
| `ceo` | who is the CEO of {company} |
| `owner` | who owns {keyword} |
| `revenue` | what is {company} revenue |
| `number_of_employees` | how many employees does {company} have |
| `headquarters` | where is {company} headquartered |
| `head_coach` | who is the head coach of {team} |
| `stadium` | what stadium does {team} play in |
| `championships` | how many championships has {team} won |

**Geography / place**

| Intent | Sample pattern |
|--------|----------------|
| `capital` | what is the capital of {place} |
| `country` | what country is {place} in |
| `location` | where is {place} located |
| `area` | what is the area of {place} |
| `population` | how many people live in {place} |
| `elevation` | how high is {mountain} |
| `length` | how long is {place} |

**Buildings / structures**

| Intent | Sample pattern |
|--------|----------------|
| `opened` | when was {place} opened |
| `designed_by` | who designed {place} |

**Biology / nature**

| Intent | Sample pattern |
|--------|----------------|
| `scientific_name` | what is the scientific name of {animal} |
| `habitat` | where does {animal} live |
| `diet` | what does {animal} eat |
| `conservation_status` | is {animal} endangered |

**Generic**

| Intent | Sample pattern |
|--------|----------------|
| `other_names` | what are {keyword} other names |
| `inventor` | who invented {keyword} |

### Supported locales

Intent files ship for 10 locales, each with 6–12 sample patterns per intent:

| Locale | Language |
|--------|----------|
| `en-US` | English |
| `ca-ES` | Catalan |
| `da-DK` | Danish |
| `de-DE` | German |
| `es-ES` | Spanish |
| `eu-ES` | Basque |
| `fr-FR` | French |
| `gl-ES` | Galician |
| `it-IT` | Italian |
| `pt-PT` | Portuguese |

Matching is case-insensitive and possessive `'s` is stripped before matching so "Darwin's father" and "Darwin father" both work.

### Adding or extending intent files

Intent files live in `ovos_ddg_plugin/locale/<locale>/`. Each file is named `<intent>.intent` and contains one Padacioso pattern per line:

```
who directed {movie}
{movie} director
who made {movie}
```

Slot names constrain the entity type and are used as the `entity_type` value returned by `_match_infobox_intents()`. Use specific types (`{person}`, `{movie}`, `{place}`, `{animal}`, `{company}`, `{software}`, `{book}`, `{artist}`, `{group}`, `{organization}`, `{mountain}`, `{team}`) instead of the generic `{keyword}` wherever the context restricts what the entity can be. Use `{keyword}` only for truly open-ended slots (e.g. `inventor`, `owner`, `other_names`).

If the DDG infobox key differs from the intent name, add an entry to `FIELD_ALIASES` in `__init__.py`:

```python
FIELD_ALIASES["my_intent"] = ["ddg_key_1", "ddg_key_2"]
```

---

## Supported locales (DDG API)

The engine maps any BCP-47 tag to the closest DDG locale code via `langcodes`. Infobox date fields (`born`, `died`) are formatted in the requested language using `ovos-date-parser`.

---

## Configuration

All keys are optional and read from the OVOS plugin config block for `ovos-ddg-plugin`:

| Key | Default | Description |
|-----|---------|-------------|
| `keyword_extractor` | `"ovos-rake-keyword-extractor"` | OPM keyword extractor plugin used when a direct query returns no result |

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

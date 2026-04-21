# ovos-ddg-plugin

[![PyPI](https://img.shields.io/pypi/v/ovos-ddg-plugin)](https://pypi.org/project/ovos-ddg-plugin/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](https://www.python.org/)

DuckDuckGo Instant Answers plugin for [OpenVoiceOS](https://openvoiceos.org).

Provides factual Q&A, infobox lookups (birthdate, death, alma mater, …), and image URLs via the [DDG Instant Answers API](https://duckduckgo.com/api). No API key required.

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

# Infobox fields are returned with score 0.9
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
#  'official website': 'https://hawking.org.uk',
#  'wikidata description': 'English theoretical physicist and cosmologist',
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

`DuckDuckGoToolbox` exposes two tools for OVOS agent pipelines:

| Tool | Description |
|------|-------------|
| `search_duckduckgo` | Best text answer (infobox field or abstract sentence) |
| `duckduckgo_image` | Image URL for a topic, or `null` |

```python
from ovos_ddg_plugin import DuckDuckGoToolbox, SearchDuckDuckGoArgs, DDGImageArgs

tb = DuckDuckGoToolbox()

answer = tb.search_ddg(SearchDuckDuckGoArgs(query="Marie Curie birthdate", lang="en-us"))
print(answer.result)
# November 7, 1867

img = tb.ddg_image(DDGImageArgs(query="Marie Curie", lang="en-us"))
print(img.url)
# https://duckduckgo.com/i/...jpg
```

---

## Supported locales

DDG Instant Answers supports ~70 locale codes. The engine maps any BCP-47 language tag to the closest supported DDG locale automatically.

Infobox date fields (`born`, `died`) are formatted in the requested language using `ovos-date-parser`. All other fields are returned as-is from the DDG API.

---

## Infobox intent matching

The engine ships Padacioso intent files for 10 languages that let `query()` detect when a question targets a specific infobox field, returning the precise value instead of the full abstract:

| Intent | Example |
|--------|---------|
| `born` | "when was {query} born" |
| `died` | "when did {query} die" |
| `known_for` | "what is {query} known for" |
| `alma_mater` | "what is {query} alma mater" |
| `children` | "how many children did {query} have" |
| `education` | "where did {query} study" |
| `fields` | "what field does {query} work in" |
| `thesis` | "{query} thesis" |
| `resting_place` | "where is {query} buried" |
| `official_website` | "{query} official website" |
| `age_at_death` | "how old was {query} when they died" |

---

## Configuration

All keys are optional and read from the OVOS plugin config block for `ovos-ddg-plugin`:

| Key | Default | Description |
|-----|---------|-------------|
| `keyword_extractor` | `"ovos-rake-keyword-extractor"` | OPM keyword extractor plugin used when a direct query returns no result |

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

# ovos-ddg-plugin

DuckDuckGo Instant Answers retrieval engine and agent toolbox for OpenVoiceOS — factual Q&A, structured infobox lookups, and image URLs. No API key required.

## Setup

```bash
pip install ovos-ddg-plugin
# from source:
pip install -e .
```

## Test

```bash
python -m unittest discover -s tests
```

Tests mock all network and config (`requests`, `ovos_config.Configuration`), so they run offline with no DDG access and no running daemon.

## Lint/Typecheck

No local lint/type config in-repo. A `lint.yml` workflow runs via gh-automations CI on PRs.

## Layout

- `ovos_ddg_plugin/__init__.py` — everything: `DuckDuckGoRetrievalEngine` (the `RetrievalEngine` impl), `DuckDuckGoToolbox` (the `ToolBox` impl) plus its pydantic arg/output schemas (`SearchDuckDuckGoArgs`/`Output`, `DDGInfoboxArgs`/`Output`, `DDGImageArgs`/`Output`).
- `ovos_ddg_plugin/locale/<locale>/<intent>.intent` — Padacioso infobox intent files; ~79 intents across 10 locales (en-US, ca-ES, da-DK, de-DE, es-ES, eu-ES, fr-FR, gl-ES, it-IT, pt-PT).
- `ovos_ddg_plugin/version.py` — version block (do not edit).
- `tests/test_ddg.py` — unit tests.

Entry-point groups (OPM agent plugin, two of them):
- `opm.agents.retrieval` → `ovos-ddg-plugin = ovos_ddg_plugin:DuckDuckGoRetrievalEngine`
- `opm.agents.toolbox` → `ovos-ddg-tools = ovos_ddg_plugin:DuckDuckGoToolbox`

### How query() works
`query()` first runs the utterance through per-language Padacioso matchers (`_match_infobox_intents`, using `calc_intents`). The slot name in each intent file becomes the `entity_type`. Tied candidates are disambiguated by fetching the DDG infobox and returning the first candidate whose field is non-empty. `FIELD_ALIASES` maps intent names to DDG infobox keys (tried in order). An infobox hit returns one result at score 0.9; otherwise it falls back to abstract sentences scored 0.9, 0.8, 0.7 …. `_search` retries via the configured OPM keyword extractor (`ovos-rake-keyword-extractor` by default) when a direct query has no abstract.

## Conventions (Org hard rules)

- Branches: `dev` for work, `master` for stable. NEVER `main`.
- Never edit `version.py` — gh-automations bumps semver from conventional-commit prefixes (`feat:`, `fix:`, `feat!:`).
- New repos private by default; do not make source public without asking.
- Commit identity: JarbasAi <jarbasai@mailfence.com>.
- Reference `OpenVoiceOS/gh-automations` reusable workflows at `@dev`.
- No Neon / `neon-*` references.
- No meta-commentary (no history, dates, or "design mistake" narration); describe current state only.
- CI is provided by OpenVoiceOS/gh-automations.

## Gotchas

- `LOCALE_MAPPING` uses DDG's reversed locale codes (`en-US` → `us-en`). BCP-47 tags are matched to the nearest supported code via `langcodes.closest_match`; distance > 10 means "unsupported locale" and returns `{}`.
- Infobox keys are normalised: label lowercased, spaces → underscores (`age at death` → `age_at_death`). Intent files and `FIELD_ALIASES` must follow that convention.
- Possessive `'s` (and the unicode `’s`) is stripped from both training samples and queries before matching.
- `born`/`died` infobox values are date-formatted via `ovos-date-parser` `nice_date` in the requested language; other fields pass through raw.
- Package name vs import: package is `ovos-ddg-plugin`, importable module `ovos_ddg_plugin`. A legacy `ovos-ddg-solver-plugin` package still exists on PyPI.
- The `Homepage` URL in `pyproject.toml` points at `OpenVoiceOS/ovos-ddg-solver-plugin`, not this repo.

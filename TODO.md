# TODO

## 1. Intent test coverage — every .intent file × every locale

Every `.intent` file in every locale must have at least one test case in `_INTENT_CASES`.

| Locale | Covered | Total | Status |
|--------|---------|-------|--------|
| en-US  | 79      | 79    | ✅ done |
| ca-ES  | 79      | 79    | ✅ done |
| da-DK  | 79      | 79    | ✅ done |
| de-DE  | 79      | 79    | ✅ done |
| es-ES  | 79      | 79    | ✅ done |
| eu-ES  | 79      | 79    | ✅ done |
| fr-FR  | 79      | 79    | ✅ done |
| gl-ES  | 79      | 79    | ✅ done |
| it-IT  | 79      | 79    | ✅ done |
| pt-PT  | 79      | 79    | ✅ done |

**Action**: Expand `_INTENT_CASES` in `tests/test_ddg.py` so every file is covered.

---

## 2. ✅ Fix trailing empty-alternative bug in non-English locales

`(word|)` with a trailing empty alternative allows the entity slot in the same pattern to over-capture query words, causing cross-intent false matches.
The fix was applied to en-US `conservation_status.intent`. The same bug exists in all 9 non-English `conservation_status.intent` files and possibly others.

**Action**:
- Strip trailing `|)` → `)` from `conservation_status.intent` in all non-English locales
- Audit `citizenship`, `religion`, `political_party` in non-English locales for the same pattern

---

## 3. ✅ Entity-type audit — replace `{keyword}` with specific types where justified

Per user guidance: use typed entity labels (`{person}`, `{movie}`, etc.) instead of `{keyword}` where the context clearly constrains the entity type.

| Intent file | Current | Verdict |
|-------------|---------|---------|
| `notable_work` | `{keyword}` | **Change → `{person}`** — "most famous work" always describes a creator |
| `scientific_name` | `{keyword}` | Keep — can be animal, plant, fungus, bacteria |
| `notable_awards` | `{keyword}` | Keep — awards go to persons AND films/companies |
| `known_for` | `{keyword}` | Keep — applies to persons, places, cities |
| `official_website` | `{keyword}` | Keep — persons, companies, teams |
| `genre` | `{keyword}` | Keep — films, books, games, music |
| `language` | `{keyword}` | Keep — films, books, software |
| `inventor` | `{keyword}` | Keep — the invented thing is generic |
| `owner` | `{keyword}` | Keep — any ownable entity |
| `length` | `{keyword}` | Keep — rivers, films, roads |
| `nickname` | `{keyword}` | Keep — persons, teams, places |
| `other_names` | `{keyword}` | Keep — any entity |
| `opened` | `{keyword}` | Keep — buildings, parks, bridges |
| `designed_by` | `{keyword}` | Keep — products, buildings, logos |
| `original_title` | `{keyword}` | Keep — films, books |
| `publisher` | `{keyword}` | Keep — books, games, magazines |
| `released` | `{keyword}` | Keep — films, albums, software |
| `composers` | `{keyword}` | Keep — films, games, musicals, operas |
| `location` | `{keyword}` | Keep — any locatable entity |

**Action**: Change `{keyword}` → `{person}` in `notable_work.intent` in all 10 locales.
Also note: future intents about linguistic/grammatical queries (e.g., word gender) should use `{word}`, not `{keyword}`.

---

## 4. ✅ Expand non-person intent files to 6-10 samples

All non-person intent files in 9 non-English locales expanded from 2-4 samples to 7-8 samples each.
~45 files × 9 locales covered: ca-ES, da-DK, de-DE, es-ES, eu-ES, fr-FR, gl-ES, it-IT, pt-PT.

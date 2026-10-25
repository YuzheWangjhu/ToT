# Evaluation protocol

This guide summarizes the implemented evaluation conditions, scoring, and statistical settings. To match the submitted results, use the following mapping: paper Track 1 Sparse/Concurrent = `1a`/`1b`, paper Track 2 = `1c`, and paper Track 3 = `2`. Tasks 3.1–3.5 map to `2A.1`, `2A.2`, `2A.3`, `2B.1`, and `2B.2`.

## Conditions and information access

| CLI condition | Information supplied |
| --- | --- |
| `active` | No initial audio. The model chooses bounded intervals and may stop early. |
| `full_context` | Complete episode attached; no active listening policy is scored. |
| `active_same_session_periodic_probe` | Conversational checkpoints stay in the continuing action-selection session. This is the paper's additional Track 2 condition. |

Paper Tracks 1 and 2 query one-second bins; paper Track 3 queries continuous start/end times in seconds. Use the controlled tracks' `binned` mode for paper comparisons. The implementation also contains continuous controlled-track variants; these are outside the paper's frozen evaluation.

For standard active evaluation on paper Tracks 1 and 2, probes use fresh sessions containing acquired crops in chronological order. Their questions and answers do not enter subsequent action selection. A conversational checkpoint occurs when the union of queried audio first fully covers a sentence's annotated span. In the same-session condition, the checkpoint questions and answers remain available to later actions.

Every paper Track 3 call starts a new session. The initial action request and final-answer request are text only. Each valid query supplies its crop as the next request's only audio, together with query history, remaining budgets, and a benchmark-managed canonical ledger. Ledger entries use `NOT_FOUND_YET`, `FOUND` with a predicted interval, or `LIKELY_ABSENT`. `SET` lines update individual entries; `NO_UPDATE` preserves the report. A correct final answer does not retrospectively create acquired evidence or repair the ledger.

## Frozen budgets

Replaying audio consumes the budget again. Out-of-bounds queries, excessive spans, and queries exceeding remaining budgets are invalid.

| Paper setting | Maximum query span | Action/query limit | Total listening budget |
| --- | --- | --- | --- |
| Track 1, both settings | 3 one-second bins | 8 action steps | 20 queried-bin units |
| Track 2, episode duration D, N = ceil(D) | 3 one-second bins | ceil(0.8N) action steps | 2N queried-bin units |
| Task 3.1 present | 384 seconds | 6 queries | 768 seconds |
| Task 3.1 absent | 384 seconds | 10 queries | 1,200 seconds |
| Task 3.2 | 384 seconds | 8 queries | 768 seconds |
| Tasks 3.3 and 3.4 | 384 seconds | 10 queries | 1,200 seconds |
| Task 3.5 | 384 seconds | 12 queries | 1,584 seconds |

Controlled-track priors are `Discovery`, `Count-known`, and `Inventory-known`. Discovery gives the vocabulary/task format and broad source information; Count-known adds the exact event or sentence count; Inventory-known adds the present event labels or conversation category. The primary scoreboard uses Discovery. Paper Track 3 has no prior-information levels.

## Responses, correction, and operational outcomes

The Track 1 timeline report has ten rows: label sets, `00` for silence, and `??` for unresolved bins. Semantic and format validity are separate checks. Paper Tracks 1 and 2 allow one text-only correction after an invalid action, within the originating session, followed by the same validity and budget checks.

Paper Track 3 validates updates individually. An invalid update leaves the referenced ledger entry unchanged and makes the action response semantically invalid. The final-answer parser can extract one complete JSON object from permitted surrounding Markdown fences or provider control tokens. A remaining parse failure permits one text-only syntax-conversion request in a new session. Conversion must preserve the original semantic claims and introduces no new evidence. Exact templates and task-specific JSON contracts are in `tot_benchmark/track2_protocol.py` and `tot_benchmark/track2.py`.

Successful completion, invalid-action termination, budget exhaustion, unsupported capability, and API failure are distinct outcomes. Unrecovered API failures and unsupported paper Track 3 conditions are unscored and excluded from performance averages. Invalid-action terminations otherwise remain included; missing, incomplete, unparsable, or semantically invalid required paper Track 3 final answers receive answer quality zero. Do not turn missing diagnostic observations into zero or discard protocol failures when recomputing paper results.

The runner uses bounded retries and records unrecovered API failures separately from task outcomes. A syntax correction is a protocol operation, distinct from an API request retry. Use the archived outputs to reproduce the submitted analyses; fresh calls exercise current provider behavior.

## Scores and aggregation

Scores are on [0, 1]. Protocol compliance P is the geometric mean of post-correction format validity and semantic validity, set to zero on invalid-action termination. Active composite score is `M = P^0.25 × E^0.35 × A^0.40`; full-context score is answer quality A. Any zero active component yields M = 0.

Evidence acquisition E combines relevant coverage with listening efficiency. Track 1 weights event coverage 0.70 and occupied-bin coverage 0.30, then multiplies by nonredundancy. Track 2 weights target-evidence recall 0.70 and sentence coverage 0.30, then multiplies by the mean of prefix continuity and nonredundancy. Track 3 uses present-target coverage, unique coverage for absence, or their task-appropriate mixture, multiplied by nonredundancy. Active runs with no listening receive zero acquisition quality.

Answer quality A combines task performance with temporal, uncertainty, and update diagnostics using the manuscript's track-specific weights. Inapplicable weighted components are omitted and remaining weights rescaled; zero denotes failure on an applicable observation. Track 3 answer quality weights core task performance 0.50, answer structure 0.20, confidence consistency 0.10, strict completion 0.15, and temporal localization 0.05. Confidence consistency is `1 − (c − y)^2`, where c is reported confidence and y is binary task correctness. An absence claim is evidentially supported only when correct and made after at least 80% unique audio coverage. Tolerant localization uses endpoint tolerances of 10 seconds for scenes and 1 second for utterances, with tolerant IoU ≥ 0.5 for a localization pass. Ordinary overlap measures use original endpoints.

Track 1 averages episodes within Sparse/Concurrent, then weights the two settings equally. Track 2 averages episodes. Track 3 averages instances within variants, variants within each of the five tasks, and then tasks, with equal weights at each aggregation level. Overall track weights are 0.20, 0.40, and 0.40. An unbalanced average over all 418 long-horizon instances does not implement this score.

## Statistical settings

The paper evaluates each applicable model/episode/condition/prior combination once; it does not estimate variability across repeated generations. For Track 3 the unit of inference is the task instance, clustered by its source episode.

| Analysis | Frozen setting |
| --- | --- |
| Paired 95% percentile intervals | 10,000 cluster-bootstrap resamples; seed 20260724 |
| Bootstrap units | Track 1 episodes; Track 2 base conversations with all four voice renderings; Track 3 episodes with all tasks |
| Rank analysis | 10,000 paired item-cluster resamples with common cross-model draws; seed 20260726 |
| Matched tests | Two-sided sign flips; exact enumeration for ≤16 units, otherwise 50,000 Monte Carlo flips; seed 20260725 + number of units |
| Multiple comparisons | Holm correction within predefined comparison families |

The [reproduction workflow](reproduction.md) fixes the input inventory and preserves these denominators, clusters, seeds, and model inclusion rules.

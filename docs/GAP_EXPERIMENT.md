# Finding missing frames with complete video context

This experiment asks whether Qwen can request useful evidence after the
least-important surgical candidate stills have been removed. Qwen first ranks
all candidates by their contribution to understanding the visible procedure,
then independent sessions receive the top 50%, 30%, or 10%. Every session receives
the complete native video. Only the candidate stills change between conditions.

The experiment uses Qwen's local llama.cpp native-video path. Its rankings,
reasons, and evidence questions are model-generated proposals, separate from
SOSpine's original tool-label/coordinate CSV rows. See [runtime setup and migration status](LLAMA_CPP.md#migration-from-ollama)
for the move from Ollama and ongoing validation.

The current protocol is `native-video-gap-experiment-v2`. It replaces the first
experiment's opposite omission policy, which supplied the least-important
candidates. Historical runs and their original condition labels remain intact.

The reference ranking, each of the three omission tests, and the all-candidate
control use separate conversations. Tests run sequentially on the local machine;
sharing the model weights does not share conversation history. Retrieval and
follow-up reviews retain the complete video and earlier evidence within their
own condition.

## Start, inspect, pause, and resume

Use a prepared contextual-selection run whose complete source and initial
candidate manifest are still available. The experiment uses **all initial
candidates**, including any that Qwen dropped in the earlier selection run. For
a run prepared with 24 candidates, this means all **24** initial candidates,
regardless of how many the earlier review kept.

Prepare the experiment without starting Qwen:

```sh
.venv/bin/python -m yasargil experiment-frame-gaps \
  --selection-run outputs/smart_selection/S6A3 \
  --output-dir outputs/gap_experiments/S6A3_surgical_importance_v2 \
  --prepare-only
```

Start the prepared experiment, or resume it later:

```sh
.venv/bin/python -m yasargil experiment-frame-gaps \
  --output-dir outputs/gap_experiments/S6A3_surgical_importance_v2 --resume
```

Inspect saved progress:

```sh
.venv/bin/python -m yasargil gap-experiment-status \
  --output-dir outputs/gap_experiments/S6A3_surgical_importance_v2
```

Request a pause:

```sh
.venv/bin/python -m yasargil pause-gap-experiment \
  --output-dir outputs/gap_experiments/S6A3_surgical_importance_v2
```

Pause takes effect after the current model call finishes and its result is
checkpointed. It does not cancel an in-flight response. Resume uses the saved
configuration and verifies the saved evidence before continuing.

Open `outputs/gap_experiments/S6A3_surgical_importance_v2/report.html` to inspect the reference,
condition sets, scores, and retrieval evidence. It works from local files and
uses no remote images, scripts, or styles. It also shows incomplete, running,
paused, and failed states. The JSON artifacts provide the full audit trail.

| Setting | Default |
| --- | --- |
| `--context-size` | 131,072 tokens |
| `--image-max-tokens` | 256 visual tokens per image budget |
| `--ranking-max-tokens` | 6,144 output tokens |
| `--review-max-tokens` | 4,096 output tokens per review |
| `--requests-per-round` | 6 evidence intervals per review |
| `--retrieval-frames` | 6 additional frames per retrieval round |
| `--max-retrieval-rounds` | 4, allowing up to 5 condition reviews |
| `--max-request-span-ms` | 10,000 milliseconds per interval |
| `--max-moment-span-ms` | 10,000 milliseconds between members of a reference moment group |
| `--tolerance-ms` | 1,000 milliseconds around an actual reference member timestamp |

The frame and request allowances must both accommodate the largest withheld set.
The effective interval limit is also capped at one half of the video duration,
so requesting almost an entire short video cannot earn full detection credit.
For S6A3 the effective limit remains 10 seconds.
For 24 candidates, six requests and six retrieved frames over four retrieval
rounds allow up to 24 executed intervals and 24 additional frames. A final review
can still request more evidence; those requests remain unfulfilled and visible
when the retrieval budget is exhausted. This is a
capacity allowance, not a guarantee that Qwen will spend it usefully. If the
complete evidence and output do not fit in context, the run fails explicitly.

## One ranking, four independent conditions

Qwen first sees the complete video and the whole initial candidate pool together.
The ranking prompt asks for **surgical importance**, using visible evidence such
as a consequential maneuver, an informative view of the operative target,
instrument–tissue interaction, a meaningful state change, or a useful view of the
result. Repeated views, obscured views, and frames that add little information
rank lower. The documented procedure context guides the task but cannot supply
findings that are not visible.

It produces one descending importance order, a score from 0 to 100 and a concrete
surgical-relevance reason for each candidate, plus moment groups for candidates
that depict equivalent nearby evidence. Candidate stills are presented in a
deterministically shuffled order for ranking; the complete video remains in its
original order. A ranking that merely copies chronological order, reverse
chronological order, or candidate presentation order fails the reference-quality
check before any condition audits begin. These checks can detect a copied order;
passing them does not establish clinical validity.

If the only raw-output problem is that Qwen listed valid scores out of order,
the runner derives a descending ranking by sorting those scores. It preserves
Qwen's original order within equal scores and changes no score, reason, moment
ID, or source identity. The raw response and original quality receipt remain
available; a separate normalization receipt records the exact derivation.
Ordering checks run again on the derived ranking. Chronological, reversed,
presentation-copy, flat-score, and incomplete rankings still stop the audits.
Cutoffs through tied scores are reproducible choices, not evidence that one tied
frame is more important than another.

Every candidate must appear exactly once. The software attaches timestamps,
paths, and hashes from the verified source manifest, rather than asking Qwen to
invent these locators.

The supplied groups are nested prefixes of this one ranking, from most important
downward. The withheld groups contain the least-important remaining candidates.
Counts use
nearest-integer rounding with halves rounded up; the actual counts and
percentages are saved. A non-control condition keeps at least one supplied and
one withheld candidate.

| Condition | Withheld from the 24-candidate pool | Supplied as stills |
| --- | --- | --- |
| Drop least-important 50% (`drop_50`) | Lowest-ranked 12 | Top 12, or 50% |
| Drop least-important 70% (`drop_70`) | Lowest-ranked 17 | Top 7, or approximately 29.2% |
| Drop least-important 90% (`drop_90`) | Lowest-ranked 22 | Top 2, or approximately 8.3% |
| All-candidate control | None | All 24 |

Each condition receives its supplied stills in chronological order and the same
review instructions. It receives no ranking, importance reasons, withheld IDs,
or answers from another condition. Protected temporal anchors from the original
embedding selection do not override this experiment's deliberate omissions or
the model's keep/drop decisions.

A withheld candidate does not necessarily create a missing moment. If a supplied
candidate belongs to the same reference moment group, that moment is already
represented and is excluded from the missing-moment denominator. The report
preserves both the withheld candidate IDs and the moments that are actually
missing according to the provisional reference.

## What is measured

There is no single accuracy number. The report separates these stages:

| Measure | Meaning |
| --- | --- |
| Initially requested | Fraction of initially missing reference moments covered by the first review's eligible requests |
| Cumulative requested | Fraction covered by eligible requests across all completed reviews |
| Cumulative retrieved | Fraction for which actual source evidence was returned |
| Final retained | Fraction represented by frames Qwen kept in its latest completed review |
| Exact versus temporal proxy | Whether recovery used a known reference-group member or a nearby source observation |
| Completion with unretained reference moments | Qwen declared completion while omitted reference moments remained outside its retained set; a neutral observation in v2 |
| Request cost | Request count, broad requests, total/unique time coverage, and returned-frame count |

When no reference moments are missing, recall is **not applicable**, not zero or
perfect. This applies to the all-candidate control, though that condition may
still request genuinely useful evidence outside the initial candidate pool.

An evidence request must specify an interval and a visible-evidence question.
Intervals longer than the configured maximum credited span receive **no detection
credit** and return no frames in this runner. Qwen receives a rejection receipt
and can narrow the request if retrieval rounds remain. The broad request and its
cost stay visible. This keeps asking for most of the video from looking like
precise gap detection.

Exact reference membership and temporal proximity are different evidence types.
A new frame receives a temporal-proxy match only if it is within the configured
tolerance of an actual member timestamp and matches one reference group.
Matching somewhere inside a long group's overall interval is insufficient.
Frames close to multiple groups are marked ambiguous and receive no automatic
coverage credit. Every temporal-proxy match needs visual review.

Retrieved frames outside all reference groups are preserved for inspection.
They may be discoveries that the original candidate pool missed; they are not
automatically counted as false positives. Frames returned after the latest model
decision remain unreviewed and receive no final-retention credit until Qwen
judges them.

The ranking and grouping are Qwen's judgments. These measurements test recovery
relative to that frozen model reference. They do not establish surgical
importance, anatomical accuracy, or clinical correctness. A human can inspect
the complete evidence before interpreting a high or low score.

In this version, omitted candidates were deliberately ranked least important.
Requesting every omitted moment is not necessarily the best result: a retained
set may adequately explain the procedure without those moments. A zero-request
answer, low recovery fraction, or completion with unretained reference moments
therefore needs inspection of the importance scores and visible evidence before
it is called a failure. The v2 score records this completion observation neutrally
and leaves the historical `false_completion_against_provisional_reference` field
not applicable. Historical v1 reports retain their original false-completion
flag. The report preserves these measurements instead of treating them as a
surgical quality score.

## Video coverage and provenance

This experiment uses the complete-video verification path from
[contextual frame selection](SMART_FRAME_SELECTION.md). It verifies the source
timeline, compares the ordered decoded RGB frames against the native decoder's
FPS-filter output, requires every expected decoded frame ID in each Qwen
request, and rejects context truncation. No lower-rate sample or text summary
replaces the video. The model's image-token budget still limits visual detail.

The pinned llama.cpp runtime's native video path currently supports timelines
that pass the constant-rate verification. Variable-frame-rate inputs, multiple
video streams, missing or duplicated frames, changed source bytes, and contexts
that do not fit are rejected. The workflow does not silently retime the source
or discard video frames to make it fit.

Every candidate, retrieved observation, and ranking entry retains its original
stable frame ID, frame index, timestamp, timestamp basis, source path and hash,
and image path and hash. Retrieval selects real source observations from the
requested interval; it never creates an image for an unsupported timestamp.

For SOSpine S6A3 the available source is **288 released JPEGs**, reconstructed
losslessly at a nominal 1 fps. Duration is `N / fps`, so this video lasts
288 seconds (4:48); the last frame starts at 287 seconds (4:47). Timestamps are
explicitly `reconstructed_nominal`. Ranking, retrieval, and scoring use this
media timeline, without rescaling it from recorded repair time or outcomes-CSV
duration fields. Qwen receives the same rule in the reference and condition
sessions. No 2× correction is inferred. Original video inputs retain their
verified PTS timeline and duration instead of assuming 1 fps.

These times locate released observations, not elapsed time in the original
repair. Seeing every available image does not establish end-to-end procedure
coverage or recover absent frames. Case outcomes remain available for separate
analysis. See the [shared timing policy](SMART_FRAME_SELECTION.md#timestamps-provenance-and-keepdrop-decisions).

The explicit media-timing prompts apply to newly prepared experiments.
Completed experiments keep their original requests and outputs; use a new
output directory to run the updated prompts.

## Review artifacts

| Artifact | Purpose |
| --- | --- |
| `summary.json` | Overall status, source identity, reference status, condition counts and provisional scores |
| `report.html` | Local visual comparison, ranking thumbnails, supplied/withheld sets, provenance and evidence links |
| `reference/reference.json` | Frozen scored ranking, surgical-relevance reasons, moment groups, and source-derived frame provenance |
| `reference/ranking-quality.json` | Accepted/rejected heuristic order and score checks; rejected rankings stop before condition audits |
| `reference/ranking-quality-raw.json` | Original checks before any score-order normalization |
| `reference/ranking-normalization.json`, `reference/normalized-output.json` | Reproducible descending score order, stable ties, and links to immutable raw output hashes |
| `reference/round-00/` | Exact reference request/response and complete-video verification |
| `conditions/<condition>/` | Separate condition state, selection, metrics, and runtime artifacts |
| `conditions/<condition>/rounds/round-NN/` | Candidate manifest, exact request/response, video verification, and retrieval receipts |
| `conditions/<condition>/metrics.json` | Matching receipts, per-round progress, exact/proxy recoveries, misses and request costs |

The report exposes original source file links and full hashes for traceability.
All model-written text is escaped as text, including reasons and evidence
questions. Reporting does not execute model-generated HTML.

## Validation limits

Automated checks cover omission direction, score validation, suspicious ranking
orders, isolated conversations, source provenance, retrieval, scoring,
pause/resume, recovery after interruption, and tampered state. Score normalization
checks preserve raw responses and judgments, keep equal-score order stable,
and resume accepted work without repeating inference. Original ranking-quality
rejections remain separate from checks on a derived ranking.

A synthetic fixture or successful decoding check does not establish successful
surgical judgments. The model may request no useful evidence, flag conflicting
context, or declare completion while still missing reference moments. Inspect
those outcomes in your own `report.html`, `summary.json`, condition metrics, and
retained responses. Source images, generated reports, and run artifacts are not
distributed with the repository. Use a new output directory when changing the
protocol; historical requests and responses retain their original meaning.

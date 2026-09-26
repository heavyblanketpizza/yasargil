# TODO: Qwen penalties and timestamp accuracy

**Priority: low. Status: deferred; controlled comparison not yet run.**

Investigate whether presence penalty contributes to timestamp errors or interacts
with thinking mode in Qwen video responses. This is a research hypothesis, not an
established cause or a proposed change to runtime defaults.

## Why investigate

Presence penalty acts on previously generated tokens, including numeric tokens
and punctuation. It can therefore affect generated timestamps without changing
the source video's timeline. The current runtime excludes prompt tokens from
penalty history, but includes generated thinking tokens and retains them when
the final answer starts. Equal penalty settings across thinking modes therefore
do not imply equal penalty history.

The original out-of-bounds timestamp error occurred with neutral penalties:
presence `0`, frequency `0`, and repetition `1`. An excessive penalty cannot
explain that failure. Later thinking and non-thinking comparisons retained
presence `1.5`; numerical frame citations were already valid before thinking
improved descriptions. The contribution of penalties remains unmeasured.

[Qwen's recommendations](https://huggingface.co/Qwen/Qwen3.8-27B#best-practices)
use presence `1.5` for non-thinking and `0` for thinking. The
[current runtime](LLAMA_CPP.md) retains the non-thinking numerical sampling
settings with thinking enabled, so it does not compare the two complete
recommended recipes.

## Controlled comparison

- [ ] Run the following four conditions on the same evaluation clips and targets.

| Condition | Thinking | Presence penalty |
|---|---|---|
| A | Off | `0` |
| B | Off | `1.5` |
| C | On | `0` |
| D | On | `1.5` |

- [ ] Hold model, quantization, runtime, video processing, user/system prompts,
  output schema, output-token budget, and other sampling controls fixed. Use
  temperature `0.7`, top-p `0.8`, top-k `20`, min-p `0`, frequency penalty `0`,
  repetition penalty `1`, and generated-only history covering the output budget.
- [ ] Repeat all conditions with the same prespecified seed set across multiple
  clips. Save exact requests, effective sampler settings, raw responses, and
  thinking/final token counts. Include truncations and failed answers in results.
- [ ] Record template instructions introduced by thinking mode. This comparison
  measures the mode switch and its template effects together; identical seeds
  do not imply identical sampling trajectories across modes.

## Evaluate separately

- [ ] **Numerical timestamps:** invalid, reversed, or out-of-bounds times, including
  free text. Distinguish model-written timestamps from times deterministically
  resolved from frame IDs; valid frame IDs alone do not test timestamp generation.
- [ ] **Event localization:** compare claimed event intervals with independently
  reviewed references. A valid point citation does not establish an event boundary.
- [ ] **Description accuracy:** unsupported or contradicted visual claims.
- [ ] **Repetition:** duplicate descriptions and claims across distinct targets.

- [ ] Report per-condition results, variation across seeds/clips, failures, and
  generation cost. Assess the penalty effect within each mode and whether it
  differs between modes. Keep conclusions bounded by the evaluated tasks.

If an interaction appears, a separate follow-up can isolate whether carrying
thinking tokens into penalty history explains it. Changing runtime defaults
requires evidence from the comparison; this TODO records pending research only.

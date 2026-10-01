# Live guidance replay

`yasargil.live` replays a SOSpine case as if it were a live microscope feed and
runs a guidance system on it: per-frame perception, an event log, three loops
that read the log at different speeds, and a speak gate that decides what
reaches the surgeon. It is a research harness for one question: **how far can a
local vision-language model go as the reasoning layer of real-time guidance?**

**Research replay only.** Outputs are not validated for clinical use. Software
that analyzes surgical images to guide a surgeon is a regulated medical device
in most jurisdictions.

## How it works

```text
frame ──> perception ──> event log ──┬──> reflex rules       (< 100 ms, no LLM)
           (every frame)  (state      ├──> procedure tracker  (seconds, spec-driven)
                           changes)   └──> Qwen agent         (seconds, on request)
                                                 │  tools: query events, current state,
                                                 │  view or perceive past frames,
                                                 │  ask the specialist
                                                 └──> citation check
                                       all three ──> speak gate ──> surgeon
```

- **Perception** turns each frame into detections: instruments with tip points
  and the durotomy region, with confidences, in normalized coordinates.
- **The event log** records state changes, not frames. An instrument enters
  after `enter_frames` consecutive confident detections and leaves after
  `exit_frames` consecutive misses. Every event has an ID such as `E000123`, the
  time it was emitted, the time the change began (onset), its confidence, the
  frames that support it and any events it cites. Frames that were never
  observed (missing releases, or unlabeled frames under label perception) are
  not treated as absence.
- **Reflex rules** fire on events, for example a tip close to the durotomy or
  the needle leaving the view while the needle driver is still in. They never
  call a model.
- **The procedure tracker** maps visible instruments to coarse steps from a
  procedure spec and flags unexpected transitions. SOSpine has no phase labels;
  the built-in spec is illustrative and unvalidated.
- **The Qwen agent** never watches the video. It reads the event log and looks
  at individual past frames only when events cannot answer. Every tool sees the
  world as of the question: later events and frames do not exist.
- **The citation check** keeps only claims that cite events known at question
  time or frames the agent actually inspected. A claim that names an instrument
  must cite an event about it unless it cites an inspected frame. Only verified
  claim text is spoken.
- **The speak gate** applies a confidence floor, per-message cooldowns, a rolling
  per-minute budget (critical alerts are exempt) and a staleness limit. Answers
  to a direct question are always spoken. Suppressed messages are recorded with
  their reason.

## Setup

Run commands from the repository root.

1. `uv sync --frozen`. Add `--extra selection` for the DINO probe.
2. Download SOSpine as described in [Data sources](DATA_SOURCES.md).
3. For the agent, install the original llama.cpp release and the Qwen
   model/projector pair from the [local inference guide](LLAMA_CPP.md). The
   agent uses the original `b10809` release with its Jinja chat template
   (native tool calls), not the patched video build. `--specialist` also needs
   the MedGemma pair; both 27B models fit together in 96 GB of unified memory.

## Replay a case

Label perception, no agent, as fast as possible:

```bash
uv run python -m yasargil.live replay \
  --dataset-root /path/to/datasets/SOSpine --case S2A2 \
  --output-dir outputs/live/S2A2-labels --speed 0
```

With the agent answering ten generated questions at real-time speed:

```bash
uv run python -m yasargil.live replay \
  --dataset-root /path/to/datasets/SOSpine --case S2A2 \
  --output-dir outputs/live/S2A2-agent --agent --auto-questions 10
```

| Option | Meaning |
| --- | --- |
| `--speed` | `1` real time (default), `4` four times faster, `0` as fast as possible |
| `--blocking` | Pause the replay while the agent answers (deterministic). Without it the agent works on a background thread and its answers are spoken when they arrive, so staleness is measured |
| `--questions FILE` | JSON Lines or a JSON array of `{"at_s": 120, "text": "...", "value_hint": "..."}` |
| `--auto-questions N` | Generate N questions with ground truth from the labels |
| `--start-frame`, `--end-frame` | Replay a window of release indices |
| `--perception probe --probe-dir DIR` | Use a trained DINO probe instead of labels |
| `--miss-rate`, `--false-positive-rate`, `--jitter`, `--confidence-noise`, `--seed` | Degrade any perception source with seeded noise |
| `--agent-mode schema` | One grammar-constrained JSON action per turn instead of native tool calls |
| `--max-steps`, `--max-images`, `--keep-images`, `--deadline-s`, `--thinking` | Agent budgets and Qwen thinking |
| `--specialist` | Offer an `ask_specialist` tool backed by MedGemma |
| `--procedure FILE`, `--rules FILE` | Replace the procedure spec or reflex rules |

Each run directory is new; existing runs are never overwritten. A run contains:

| File | Contents |
| --- | --- |
| `run.json` | Frames, perception identity, tracker, procedure, rules, gate, session and agent settings |
| `observations.jsonl` | Per-frame detections |
| `events.jsonl` | The event log |
| `utterances.jsonl` | Every gate decision, spoken or suppressed, with its reason and citations |
| `timing.jsonl` | Per-frame milliseconds for perception, events, reflex and gate, and missed frame deadlines |
| `questions.jsonl`, `answers.jsonl` | Questions as asked and answers as delivered, with latency and staleness |
| `agent/<question>/` | Exact request and response bytes for every agent step, and the verified result |
| `runtime/` | llama-server command, model identity and server log, when the agent ran |
| `summary.json` | Counts, latency percentiles and statuses |
| `report.html` | The run viewer (next to `report/frames/` thumbnails) |

Open `report.html` in a browser to scrub through frames with perception
overlays, timeline lanes, the spoken feed (optionally with suppressed
messages), each question with its verified and rejected claims and agent steps,
and the full event log. Regenerate it with
`uv run python -m yasargil.live report --run outputs/live/S2A2-agent`.

## Ask about a finished run

```bash
uv run python -m yasargil.live ask --run outputs/live/S2A2-labels --at-s 240 \
  --agent "Has the needle left the view since the first suture pass?"
```

The agent sees the run's event log and frames only up to the frame at or before
`--at-s`. Evidence is saved under the run's `ask/` folder.

## Evaluate

```bash
uv run python -m yasargil.live eval \
  --dataset-root /path/to/datasets/SOSpine --cases S8A1 S8A2 S8A3 \
  --output-dir outputs/live-eval/labels-agent --questions-per-case 20 --agent
```

Questions are generated at frame times spread across each case and answered
at those times. Ground truth is what perfect perception would make of the case
under the tracker's own event definitions (the labels replayed through the same
hysteresis). Two answerers are graded:

- **Symbolic**: deterministic code over the run's event log. With label
  perception it must be perfect; with degraded or learned perception, its error
  is perception and event-layer loss.
- **Agent**: the Qwen loop. Its error beyond the symbolic answerer on the same
  run is what the language model loses. A non-answer is wrong even when the
  truth is "not yet".

| Template | Question | Graded value |
| --- | --- | --- |
| `visible_now` | Which instruments are in view right now? | exact set of names |
| `first_seen` | When did the X first come into view? | seconds, within 2 s, or null |
| `entry_count` | How many times has the X come into view so far? | exact integer |
| `near_now` | Is any instrument tip close to the durotomy right now? | true or false |
| `time_in_view` | How long has the X been in view without interruption? | seconds, within 2 s |
| `last_left` | When did the X last leave the view? | seconds, within 2 s, or null |

The evaluation directory holds one folder per case (`run/`, `questions.jsonl`
with truth, `grades.jsonl`, `summary.json`) and an overall `summary.json` with
accuracy by template, the agent's answered and fully cited rates, latency
percentiles and mean tool calls and images. Repeat with `--miss-rate`, a probe,
or `--agent-mode schema` to compare conditions.

## Train the DINO probe

```bash
uv sync --extra selection
uv run --extra selection python -m yasargil.live train-probe \
  --dataset-root /path/to/datasets/SOSpine --output-dir outputs/live-probe/v1
```

The probe runs frozen DINOv2-small once per frame and trains two linear heads:
presence (which instruments and whether the durotomy is in view) and a 1x1
patch head (where tips are, and the durotomy region). The patch grid keeps the
frame's aspect ratio: 1920x1080 frames use a 28x16 grid at 392x224, so
localization is coarse (about 69 px cells at full resolution). Training splits
by surgeon (default train S1–S6, validation S7, test S8); per-class presence
thresholds are chosen on validation. `metrics.json` reports presence F1, tip
error in units of image height and durotomy IoU for each partition. Use the
probe with `--perception probe --probe-dir outputs/live-probe/v1`.

## Rules and procedure specs

Reflex rules are a JSON list. Each rule matches an event type and optionally
its subject or object, may require current state, and has a priority
(`low`, `medium`, `high`, `critical`), an optional cooldown and a message
template:

```json
[{"id": "needle_out_of_view",
  "when": {"type": "instrument_left", "subject": "needle"},
  "require": {"visible_instruments": ["needle driver"]},
  "priority": "critical", "cooldown_ms": 10000,
  "message": "Needle left the view while the needle driver is still in."}]
```

A procedure spec lists steps in priority order, each with one condition
(`requires_any`, `requires_all` or `requires_none`), plus `initial`,
`dwell_frames`, allowed `transitions` and an `expected_order`. A step change
needs `dwell_frames` consecutive matching frames; a disallowed transition, or
reaching a step before its predecessors in `expected_order`, emits
`step_unexpected`.

## Limitations

- SOSpine releases 1-fps stills, so "real time" here is a 1000 ms frame budget
  and motion between samples is unrecoverable.
- Label perception replays ground truth; it is an upper bound, not a model.
- The default procedure spec is illustrative; SOSpine has no phase labels.
- Distances between tips and the durotomy use units of image height; the
  "close" threshold is a tunable heuristic, not a validated safety margin.
- Perception, the agent and the optional specialist share one GPU on a single
  machine. Contention is measured in `timing.jsonl`, not assumed away.
- Citation checks are mechanical. They catch unknown, future and uninspected
  evidence and some subject mismatches, but they cannot prove that a cited event
  semantically supports a claim.

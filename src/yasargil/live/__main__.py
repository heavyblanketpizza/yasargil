"""Live guidance research harness: replay, evaluate, ask, train a probe, view a run.

Research replay of released SOSpine frames only; not a medical device.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager, nullcontext
import json
from pathlib import Path
import sys
import uuid

from . import LiveError
from .agent import AgentConfig, GuidanceAgent
from .events import EventLog
from .evaluate import evaluate_case, summarize
from .frames import FrameSource, case_frames
from .labels import CaseLabels
from .perception import DegradedPerception, Detection, LabelPerception, Observation
from .procedure import ProcedureSpec
from .questions import GroundTruth, make_questions
from .reflex import load_rules
from .report import write_report
from .session import ScheduledQuestion, Session, SessionConfig, load_questions
from .tools import ToolContext
from .tracker import TrackerConfig
from .transport import LlamaServer, specialist_from_transport
from ..llama_cpp import MEDGEMMA_MODEL


def _perception_args(parser):
    group = parser.add_argument_group("perception")
    group.add_argument("--perception", choices=["labels", "probe"], default="labels",
                       help="labels: replay dataset labels (upper bound); probe: a trained DINO probe")
    group.add_argument("--probe-dir", type=Path, help="Directory written by train-probe")
    group.add_argument("--miss-rate", type=float, default=0.0, help="Drop each detection with this probability")
    group.add_argument("--false-positive-rate", type=float, default=0.0, help="Add a spurious instrument per frame")
    group.add_argument("--jitter", type=float, default=0.0, help="Gaussian position noise (normalized units)")
    group.add_argument("--confidence-noise", type=float, default=0.0, help="Gaussian confidence reduction")
    group.add_argument("--seed", type=int, default=0, help="Seed for perception degradation and questions")


def _agent_args(parser):
    group = parser.add_argument_group("agent")
    group.add_argument("--agent", action="store_true", help="Start the local Qwen agent (llama.cpp b10809)")
    group.add_argument("--agent-mode", choices=["native", "schema"], default="native",
                       help="native tool calls, or one grammar-constrained JSON action per turn")
    group.add_argument("--max-steps", type=int, default=6)
    group.add_argument("--max-images", type=int, default=2)
    group.add_argument("--keep-images", type=int, default=2)
    group.add_argument("--deadline-s", type=float, default=90.0)
    group.add_argument("--thinking", action="store_true", help="Enable Qwen thinking (slower)")
    group.add_argument("--context-size", type=int, default=32768)
    group.add_argument("--image-max-tokens", type=int, default=512)
    group.add_argument("--specialist", action="store_true", help="Also start MedGemma for the ask_specialist tool")
    group.add_argument("--project-root", type=Path, default=Path.cwd(), help="Where .runtime/ lives")


def _loop_args(parser):
    parser.add_argument("--procedure", type=Path, help="Procedure spec JSON (default: illustrative durotomy repair)")
    parser.add_argument("--rules", type=Path, help="Reflex rules JSON (default: built-in rules)")
    parser.add_argument("--no-report", action="store_true", help="Skip writing report.html")


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m yasargil.live", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    replay = sub.add_parser("replay", help="Replay one case through perception, events, reflex, tracker, agent and gate")
    replay.add_argument("--dataset-root", type=Path, required=True)
    replay.add_argument("--case", required=True)
    replay.add_argument("--output-dir", type=Path, required=True)
    replay.add_argument("--speed", type=float, default=1.0, help="1 = real time, 4 = four times faster, 0 = as fast as possible")
    replay.add_argument("--blocking", action="store_true", help="Pause the replay while the agent answers")
    replay.add_argument("--questions", type=Path, help="JSON Lines or JSON array of {at_s, text}")
    replay.add_argument("--auto-questions", type=int, default=0, help="Generate this many questions from the labels")
    replay.add_argument("--start-frame", type=int)
    replay.add_argument("--end-frame", type=int)
    _perception_args(replay)
    _agent_args(replay)
    _loop_args(replay)

    evaluate = sub.add_parser("eval", help="Grade generated questions per case: symbolic baseline and optional agent")
    evaluate.add_argument("--dataset-root", type=Path, required=True)
    evaluate.add_argument("--cases", nargs="+", required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--questions-per-case", type=int, default=20)
    _perception_args(evaluate)
    _agent_args(evaluate)
    _loop_args(evaluate)

    ask = sub.add_parser("ask", help="Ask the agent one question about a finished run at a chosen time")
    ask.add_argument("--run", type=Path, required=True)
    ask.add_argument("--at-s", type=float, required=True)
    ask.add_argument("--dataset-root", type=Path, help="Defaults to the dataset root recorded in the run")
    ask.add_argument("question")
    _agent_args(ask)

    probe = sub.add_parser("train-probe", help="Train DINO probe heads on SOSpine labels, split by surgeon")
    probe.add_argument("--dataset-root", type=Path, required=True)
    probe.add_argument("--output-dir", type=Path, required=True)
    probe.add_argument("--train-surgeons", nargs="+", default=["S1", "S2", "S3", "S4", "S5", "S6"])
    probe.add_argument("--val-surgeons", nargs="+", default=["S7"])
    probe.add_argument("--test-surgeons", nargs="+", default=["S8"])
    probe.add_argument("--encoder", type=Path, default=Path(".runtime/models/dinov2-small"))
    probe.add_argument("--frame-stride", type=int, default=1)
    probe.add_argument("--epochs", type=int, default=40)
    probe.add_argument("--lr", type=float, default=0.01)
    probe.add_argument("--batch-size", type=int, default=256)

    report = sub.add_parser("report", help="Write report.html for a finished run")
    report.add_argument("--run", type=Path, required=True)
    report.add_argument("--thumbnail-side", type=int, default=640)
    return parser


def make_perception(args, labels):
    if args.perception == "probe":
        from .probe import ProbePerception
        base = ProbePerception(args.probe_dir)
    else:
        base = LabelPerception(labels)
    if any((args.miss_rate, args.false_positive_rate, args.jitter, args.confidence_noise)):
        try:
            return DegradedPerception(base, args.miss_rate, args.false_positive_rate, args.jitter,
                                      args.confidence_noise, args.seed)
        except ValueError as exc:
            raise LiveError(str(exc)) from exc
    return base


def perception_from_identity(identity, dataset_root, frames):
    kind = identity.get("kind")
    if kind == "labels":
        if dataset_root is None:
            raise LiveError("This run used label perception; pass --dataset-root.")
        return LabelPerception(CaseLabels.load(dataset_root, frames.case_id, frames.size(frames.frames[0])))
    if kind == "degraded":
        return DegradedPerception(perception_from_identity(identity["inner"], dataset_root, frames),
                                  identity["miss_rate"], identity["false_positive_rate"], identity["jitter"],
                                  identity["confidence_noise"], identity["seed"])
    if kind == "dino-probe":
        from .probe import ProbePerception
        return ProbePerception(identity["model_dir"])
    raise LiveError(f"Cannot rebuild perception of kind {kind!r}")


@contextmanager
def agent_runtime(args, log_dir):
    """Start Qwen (and MedGemma when asked) for the block; yields (agent, specialist)."""
    config = AgentConfig(mode=args.agent_mode, max_steps=args.max_steps, max_images=args.max_images,
                         keep_images=args.keep_images, deadline_s=args.deadline_s, thinking=args.thinking)
    with ExitStack() as stack:
        qwen = stack.enter_context(LlamaServer(args.project_root, Path(log_dir) / "qwen", context_size=args.context_size,
                                               image_max_tokens=args.image_max_tokens, thinking=args.thinking))
        specialist = None
        if args.specialist:
            medgemma = stack.enter_context(LlamaServer(args.project_root, Path(log_dir) / "medgemma",
                                                       model=MEDGEMMA_MODEL, context_size=8192))
            specialist = specialist_from_transport(medgemma.transport(), medgemma.alias)
        yield GuidanceAgent(qwen.transport(), config), specialist


def _runtime(args, log_dir):
    return agent_runtime(args, log_dir) if args.agent else nullcontext((None, None))


def _fresh(directory):
    if directory.exists() and any(directory.iterdir()):
        raise LiveError(f"Refusing to reuse a nonempty output directory: {directory}")


def _loop_options(args):
    procedure = ProcedureSpec.load(args.procedure) if args.procedure else None
    rules = load_rules(args.rules) if args.rules else None
    return procedure, rules


def cmd_replay(args):
    frames = case_frames(args.dataset_root, args.case)
    size = frames.size(frames.frames[0])
    labels = CaseLabels.load(args.dataset_root, args.case, size) if (args.perception == "labels" or args.auto_questions) else None
    perception = make_perception(args, labels)
    procedure, rules = _loop_options(args)
    questions, truth = [], []
    if args.questions:
        questions = load_questions(args.questions)
    if args.auto_questions:
        truth = make_questions(GroundTruth(frames, labels, TrackerConfig(aspect=size[0] / size[1])),
                               args.auto_questions, args.seed)
        offset = len(questions)
        questions += [ScheduledQuestion(f"Q{offset + i + 1:03d}", q.at_ms, q.text, q.value_hint) for i, q in enumerate(truth)]
    output = args.output_dir
    _fresh(output)
    with _runtime(args, output / "runtime") as (agent, specialist):
        summary = Session(frames, perception, output, agent=agent, questions=questions, procedure=procedure, rules=rules,
                          config=SessionConfig(speed=args.speed, blocking_questions=args.blocking,
                                               start_index=args.start_frame, end_index=args.end_frame),
                          specialist=specialist,
                          run_info={"dataset_root": str(Path(args.dataset_root).resolve()), "command": "replay"}).run()
    if truth:
        with (output / "questions-truth.jsonl").open("w", encoding="utf-8") as stream:
            for offset_question, question in zip(questions[len(questions) - len(truth):], truth):
                stream.write(json.dumps({**question.to_json(), "question_id": offset_question.question_id}) + "\n")
    if not args.no_report:
        write_report(output)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_eval(args):
    output = args.output_dir
    _fresh(output)
    output.mkdir(parents=True, exist_ok=True)
    procedure, rules = _loop_options(args)
    summaries, grades = {}, []
    with _runtime(args, output / "runtime") as (agent, specialist):
        for case in args.cases:
            summaries[case] = evaluate_case(
                args.dataset_root, case, output / case, lambda labels: make_perception(args, labels),
                questions=args.questions_per_case, seed=args.seed, agent=agent, specialist=specialist,
                session_config=SessionConfig(speed=0, blocking_questions=True), procedure=procedure, rules=rules,
                run_info={"dataset_root": str(Path(args.dataset_root).resolve()), "command": "eval"})
            grades += [json.loads(line) for line in (output / case / "grades.jsonl").read_text().splitlines()]
            if not args.no_report:
                write_report(output / case / "run")
    summary = {"cases": list(args.cases), "perception": next(iter(summaries.values()))["perception"],
               "agent": vars(args)["agent"], "overall": summarize(grades),
               "by_case": {case: {name: value for name, value in item.items() if name in ("symbolic", "agent")}
                           for case, item in summaries.items()}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_ask(args):
    run = args.run
    manifest_path = run / "run.json"
    if not manifest_path.is_file():
        raise LiveError(f"Not a replay run directory (no run.json): {run}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = FrameSource(manifest["frames"]["directory"], manifest["case_id"], manifest["frames"].get("fps", 1.0))
    dataset_root = args.dataset_root or manifest.get("dataset_root")
    perception = perception_from_identity(manifest["perception"], dataset_root, frames)
    log = EventLog.load(run / "events.jsonl")
    frame = frames.at_or_before(round(args.at_s * 1000))
    if frame is None or not manifest["frames"]["first"] <= frame.index <= manifest["frames"]["last"]:
        raise LiveError(f"No replayed frame at or before t={args.at_s}s in this run.")
    latest = None
    for line in (run / "observations.jsonl").read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if value["frame_index"] == frame.index:
            latest = Observation(value["frame_index"], value["t_ms"],
                                 tuple(Detection.from_json(d) for d in value["detections"]),
                                 value["annotated"], value["producer"])
    question_id = f"t{frame.t_ms}ms-{uuid.uuid4().hex[:8]}"
    with agent_runtime(args, run / "ask" / f"runtime-{question_id}") as (agent, specialist):
        agent.evidence_dir = run / "ask"
        ctx = ToolContext(log, frames, perception, frame.t_ms, frame.index, latest=latest, specialist=specialist)
        result = agent.answer(args.question, ctx, question_id)
    print(json.dumps({"question": args.question, "at_ms": frame.t_ms, "frame_index": frame.index,
                      "status": result.status, "spoken_text": result.spoken_text,
                      "supported": result.verdict.supported, "value": (result.final or {}).get("value"),
                      "elapsed_ms": round(result.elapsed_ms, 1), "tool_calls": result.tool_calls,
                      "evidence": str(run / "ask" / question_id)}, indent=2))
    return 0


def cmd_train_probe(args):
    from .probe import train_probe
    metrics = train_probe(args.dataset_root, args.output_dir, train=args.train_surgeons, val=args.val_surgeons,
                          test=args.test_surgeons, model_path=args.encoder, frame_stride=args.frame_stride,
                          epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, seed=0,
                          progress=lambda message: print(message, file=sys.stderr, flush=True))
    print(json.dumps({key: metrics[key] for key in ("split", "thresholds", "val", "test")}, indent=2))
    return 0


def cmd_report(args):
    print(write_report(args.run, args.thumbnail_side))
    return 0


COMMANDS = {"replay": cmd_replay, "eval": cmd_eval, "ask": cmd_ask, "train-probe": cmd_train_probe, "report": cmd_report}


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "perception", None) == "probe" and not args.probe_dir:
        print("error: --perception probe needs --probe-dir (written by train-probe)", file=sys.stderr)
        return 2
    try:
        return COMMANDS[args.command](args)
    except LiveError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted; partial outputs and summary.json were kept", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())

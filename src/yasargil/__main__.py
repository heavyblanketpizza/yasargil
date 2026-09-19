"""Local source import, archive validation, human review packets and SFT exports."""
from __future__ import annotations

import argparse
import json
import signal
import sys
from contextlib import contextmanager
from pathlib import Path

from .contract import ContractError, export_records, read_records, require, validate_corpus, write_new_json
from .llama_cpp import MEDGEMMA_MODEL, QWEN_MODEL, LlamaCppClient, LlamaCppError


@contextmanager
def cooperative_stop():
    """First signal waits for a call checkpoint; a second forces interruption."""
    pending = [False]

    def stop(signum, frame):
        if pending[0]:
            raise KeyboardInterrupt
        pending[0] = True
        print("Pause requested: finishing and saving the current model call. "
              "Press Ctrl+C again to interrupt it immediately.", file=sys.stderr, flush=True)

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for sig in previous:
            signal.signal(sig, stop)
        yield lambda: pending[0]
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    from .smart_selection import add_selection_parser
    add_selection_parser(sub)
    from .selection_batch import add_selection_batch_parser
    add_selection_batch_parser(sub)
    from .gap_experiment import add_gap_parser
    add_gap_parser(sub)
    from .frame_annotation import add_annotation_parser
    add_annotation_parser(sub)
    from .medgemma_review import add_review_parser
    add_review_parser(sub)
    from .dataset_inspector import add_inspector_parser
    add_inspector_parser(sub)
    imp = sub.add_parser("import-sospine", help="Import a bounded source-derived draft, without inference")
    imp.add_argument("--dataset-root", type=Path, required=True)
    imp.add_argument("--case-id", required=True)
    imp.add_argument("--frame-indices", type=int, nargs="+", required=True)
    imp.add_argument("--output", type=Path, required=True)
    models = sub.add_parser("models", help="Inspect local llama.cpp model files and runtime identities")
    enhance = sub.add_parser("enhance-sospine", help="Qwen discovery and MedGemma review with retained evidence")
    enhance.add_argument("--dataset-root", type=Path, required=True)
    enhance.add_argument("--case-id", required=True)
    enhance.add_argument("--start-index", type=int, required=True)
    enhance.add_argument("--cutoff-index", type=int, required=True)
    enhance.add_argument("--output-dir", type=Path)
    enhance.add_argument("--resume", action="store_true", help="Resume this exact saved configuration")
    enhance.add_argument("--dry-run", action="store_true", help="Print a selection plan without writes or model calls")
    enhance.add_argument("--initial-frames", type=int, default=8)
    enhance.add_argument("--search-frames", type=int, default=4)
    enhance.add_argument("--max-frames", type=int, default=24)
    enhance.add_argument("--max-rounds", type=int, default=1)
    enhance.add_argument("--num-ctx", type=int, default=65536)
    enhance.add_argument("--num-predict", type=int, default=4096)
    enhance.add_argument("--seed", type=int, default=42)
    batch = sub.add_parser("enhance-sospine-batch", help="Process full released sequences with durable call/window checkpoints")
    batch.add_argument("--dataset-root", type=Path, required=True)
    batch.add_argument("--case-ids", nargs="+", required=True)
    batch.add_argument("--output-dir", type=Path)
    batch.add_argument("--dry-run", action="store_true")
    batch.add_argument("--window-size", type=int, default=12)
    batch.add_argument("--initial-frames", type=int, default=4)
    batch.add_argument("--search-frames", type=int, default=4)
    batch.add_argument("--max-frames", type=int, default=8)
    batch.add_argument("--max-rounds", type=int, default=1)
    batch.add_argument("--num-ctx", type=int, default=65536)
    batch.add_argument("--num-predict", type=int, default=4096)
    batch.add_argument("--seed", type=int, default=42)
    resume = sub.add_parser("resume-enhancement", help="Resume a saved batch/window using its original settings")
    resume.add_argument("--output-dir", type=Path, required=True)
    for name in ("enhancement-status", "pause-enhancement"):
        command = sub.add_parser(name)
        command.add_argument("--output-dir", type=Path, required=True)
    for command in (models, enhance, batch):
        command.add_argument("--qwen-model", default=QWEN_MODEL)
        command.add_argument("--medgemma-model", default=MEDGEMMA_MODEL)
    for command in (models, enhance, batch, resume):
        command.add_argument("--project-root", type=Path, default=Path.cwd(), help="Root containing .runtime")
        command.add_argument("--timeout", type=float, default=600, help="Seconds per llama.cpp request")
    for name in ("validate", "review-packet", "export"):
        command = sub.add_parser(name)
        command.add_argument("records", type=Path)
        command.add_argument("--dataset-root", type=Path, required=name != "validate")
        command.add_argument("--artifact-root", type=Path)
        if name == "validate":
            command.add_argument("--training", action="store_true", help="Also require review and export eligibility")
        else:
            command.add_argument("--output", type=Path, required=True)
        if name == "review-packet":
            command.add_argument("--include-outcomes", action="store_true", help="Unblind outcomes/generator in a separate retrospective section")
        if name == "export":
            command.add_argument("--preview", action="store_true")
            command.add_argument("--partition", choices=["train", "validation", "test", "unassigned"])
            command.add_argument("--allow-retrospective", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "select-sospine-batch":
            from .selection_batch import selection_batch_cli
            with cooperative_stop() as should_stop:
                selection_batch_cli(args, should_stop=should_stop)
            return
        if args.command in {"selection-batch-status", "pause-selection-batch"}:
            from .selection_batch import selection_batch_status, pause_selection_batch
            action = selection_batch_status if args.command == "selection-batch-status" else pause_selection_batch
            print(json.dumps(action(args.output_dir), indent=2))
            return
        if args.command == "inspect-dataset":
            from .dataset_inspector import inspector_cli
            inspector_cli(args)
            return
        if args.command == "review-frame-annotations":
            from .medgemma_review import review_cli
            with cooperative_stop() as should_stop:
                review_cli(args, should_stop=should_stop)
            return
        if args.command in {"frame-review-status", "pause-frame-review"}:
            from .medgemma_review import review_status, request_review_pause
            action = review_status if args.command == "frame-review-status" else request_review_pause
            print(json.dumps(action(args.output_dir), indent=2))
            return
        if args.command == "annotate-video-frames":
            from .frame_annotation import annotation_cli
            with cooperative_stop() as should_stop:
                annotation_cli(args, should_stop=should_stop)
            return
        if args.command in {"annotation-status", "pause-annotation"}:
            from .frame_annotation import annotation_status, request_annotation_pause
            action = annotation_status if args.command == "annotation-status" else request_annotation_pause
            print(json.dumps(action(args.output_dir), indent=2))
            return
        if args.command == "experiment-frame-gaps":
            from .gap_experiment import experiment_cli
            with cooperative_stop() as should_stop:
                experiment_cli(args, should_stop=should_stop)
            return
        if args.command in {"gap-experiment-status", "pause-gap-experiment"}:
            from .gap_experiment import experiment_status, request_experiment_pause
            action = experiment_status if args.command == "gap-experiment-status" else request_experiment_pause
            print(json.dumps(action(args.output_dir), indent=2))
            return
        if args.command == "select-video-frames":
            from .smart_selection import selection_cli
            selection_cli(args)
            return
        if args.command in {"enhancement-status", "pause-enhancement"}:
            from .batch import enhancement_status, request_pause
            action = enhancement_status if args.command == "enhancement-status" else request_pause
            print(json.dumps(action(args.output_dir), indent=2))
            return
        if args.command == "resume-enhancement":
            from .batch import BatchConfig, _json, enhance_batch
            from .enhancement import EnhancementConfig, enhance_sospine
            require(not args.output_dir.is_symlink(), "Cannot resume a symlinked directory")
            if (args.output_dir / "batch.json").is_file():
                saved = _json(args.output_dir / "batch.json")["plan"]
                config = BatchConfig(**saved["config"])
                run = enhance_batch
                dataset_root = saved["dataset_root"]
            else:
                saved = _json(args.output_dir / "plan.json")
                session = _json(args.output_dir / "session.json")
                config = EnhancementConfig(**saved["config"])
                run = enhance_sospine
                dataset_root = session["dataset_root"]
            with cooperative_stop() as paused:
                result = run(dataset_root, args.output_dir, config, resume=True,
                             client=LlamaCppClient(args.project_root, timeout=args.timeout), pause_requested=paused,
                             progress=lambda message: print(message, flush=True))
            print(json.dumps(result, indent=2))
            return
        if args.command == "enhance-sospine-batch":
            from .batch import BatchConfig, enhance_batch, plan_batch
            config = BatchConfig(**{key: getattr(args, key) for key in (
                "case_ids", "window_size", "initial_frames", "search_frames", "max_frames",
                "max_rounds", "qwen_model", "medgemma_model", "num_ctx", "num_predict", "seed")})
            if args.dry_run:
                print(json.dumps(plan_batch(args.dataset_root, config), indent=2))
            else:
                require(args.output_dir is not None, "--output-dir is required unless --dry-run is used")
                with cooperative_stop() as paused:
                    result = enhance_batch(args.dataset_root, args.output_dir, config,
                        client=LlamaCppClient(args.project_root, timeout=args.timeout), pause_requested=paused,
                        progress=lambda message: print(message, flush=True))
                print(json.dumps(result, indent=2))
            return
        if args.command == "models":
            client = LlamaCppClient(args.project_root, timeout=args.timeout)
            for name in (args.qwen_model, args.medgemma_model):
                info = client.model_info(name)
                print(json.dumps(info))
            return
        if args.command == "enhance-sospine":
            from .enhancement import EnhancementConfig, enhance_sospine, plan_enhancement
            config = EnhancementConfig(**{key: getattr(args, key) for key in (
                "case_id", "start_index", "cutoff_index", "initial_frames", "search_frames", "max_frames",
                "max_rounds", "qwen_model", "medgemma_model", "num_ctx", "num_predict", "seed")})
            if args.dry_run:
                print(json.dumps(plan_enhancement(args.dataset_root, config), indent=2))
            else:
                require(args.output_dir is not None, "--output-dir is required unless --dry-run is used")
                with cooperative_stop() as paused:
                    result = enhance_sospine(args.dataset_root, args.output_dir, config,
                                         resume=args.resume, pause_requested=paused,
                                         client=LlamaCppClient(args.project_root, timeout=args.timeout),
                                         progress=lambda message: print(message, flush=True))
                print(json.dumps(result, indent=2))
            return
        if args.command == "import-sospine":
            from .sospine import import_case
            require(not args.output.resolve().is_relative_to(args.dataset_root.resolve()), "Output must be outside the source dataset")
            require(not args.output.exists(), "Output already exists; choose a new archive path")
            record = import_case(args.dataset_root, args.case_id, args.frame_indices)
            write_new_json(args.output, record)
            print(f"Wrote source-derived draft: {args.output}. Review pending; no MedGemma inference.")
            return
        records = read_records(args.records)
        roots = {"dataset_root": args.dataset_root, "artifact_root": args.artifact_root}
        if args.command == "validate":
            validate_corpus(records, **roots, training=args.training)
            print(f"Validated {len(records)} archive(s); source bytes checked: {args.dataset_root is not None}. Review status unchanged.")
        else:
            require(not args.output.resolve().is_relative_to(args.dataset_root.resolve()), "Output must be outside the source dataset")
            if args.command == "review-packet":
                from .review import write_review_packet
                require(len(records) == 1, "Create one review packet per archive record")
                result = write_review_packet(records[0], args.output, **roots, include_outcomes=args.include_outcomes)
                print(json.dumps(result, default=str))
            else:
                receipt = export_records(records, args.output, **roots, preview=args.preview,
                                         partition=args.partition, allow_retrospective=args.allow_retrospective)
                print(f"Wrote {receipt['row_count']} row(s): {receipt['purpose']}; {args.output}")
    except KeyboardInterrupt:
        print("yasargil: interrupted; retained run artifacts are available in the output directory", file=sys.stderr)
        raise SystemExit(130) from None
    except (ContractError, LlamaCppError, OSError, ValueError, KeyError) as exc:
        print(f"yasargil: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()

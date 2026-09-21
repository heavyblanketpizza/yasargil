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
    from .medgemma_annotation import add_annotation_parser as add_medgemma_parser
    add_medgemma_parser(sub)
    from .dataset_inspector import add_inspector_parser
    add_inspector_parser(sub)
    imp = sub.add_parser("import-sospine", help="Import a bounded source-derived draft, without inference")
    imp.add_argument("--dataset-root", type=Path, required=True)
    imp.add_argument("--case-id", required=True)
    imp.add_argument("--frame-indices", type=int, nargs="+", required=True)
    imp.add_argument("--output", type=Path, required=True)
    models = sub.add_parser("models", help="Inspect local llama.cpp model files and runtime identities")
    models.add_argument("--qwen-model", default=QWEN_MODEL)
    models.add_argument("--medgemma-model", default=MEDGEMMA_MODEL)
    models.add_argument("--project-root", type=Path, default=Path.cwd())
    models.add_argument("--timeout", type=float, default=600)
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
        if args.command == "annotate-selected-frames":
            from .medgemma_annotation import annotation_cli as medgemma_cli
            with cooperative_stop() as should_stop:
                medgemma_cli(args, should_stop=should_stop)
            return
        if args.command in {"medgemma-annotation-status", "pause-medgemma-annotation"}:
            from .medgemma_annotation import annotation_status, request_annotation_pause
            action = annotation_status if args.command == "medgemma-annotation-status" else request_annotation_pause
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
        if args.command == "models":
            client = LlamaCppClient(args.project_root, timeout=args.timeout)
            for name in (args.qwen_model, args.medgemma_model):
                info = client.model_info(name)
                print(json.dumps(info))
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

"""Two-case annotation coordination without loading a model or changing media."""

import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from yasargil.annotation_integrity import MANIFEST_NAME
from yasargil.annotation_pair import CASES, run_pair
from yasargil.contract import ContractError, require, sha256_file
from yasargil.selection_batch import PROCEDURE_CONTEXT
from yasargil.smart_selection import _write


def read(path):
    return json.loads(Path(path).read_bytes())


class FakeAnnotationRunner:
    """Persist completed drafts so a controller restart need not infer again."""

    def __init__(self, test, *, fail_once=(), after_case=None, context_conflict=()):
        self.test = test
        self.fail_once = set(fail_once)
        self.after_case = after_case
        self.context_conflict = set(context_conflict)
        self.calls = []
        self.inferences = []
        self.active = False

    def __call__(self, selection_run, output_dir, config, *, resume, should_stop, progress):
        directory = Path(output_dir)
        case_id = directory.name
        self.test.assertIn(case_id, CASES)
        self.test.assertFalse(self.active)
        self.active = True
        try:
            self.calls.append({"case_id": case_id, "selection_run": str(selection_run),
                               "output_dir": str(directory), "config": asdict(config), "resume": resume})
            self.test.assertFalse(should_stop())
            self.test.assertEqual(read(self.test.output / "state.json")["active_case"], case_id)
            if resume:
                self.test.assertTrue((directory / "run.json").is_file())
                if (directory / "summary.json").is_file():
                    return read(directory / "summary.json")
            else:
                self.test.assertFalse(directory.exists(), "Partial preparation must be preserved elsewhere")
                directory.mkdir()
                _write(directory / "run.json", {"selection_run": str(selection_run), "config": asdict(config)})
            if case_id in self.fail_once:
                self.fail_once.remove(case_id)
                raise RuntimeError("Synthetic annotation failure")
            self.inferences.append(case_id)
            status = "context_conflict" if case_id in self.context_conflict else "completed"
            summary = {"status": status, "selected_frame_count": 2,
                       "source": {"expected_video_frames": self.test.frame_counts[case_id]}}
            _write(directory / "annotations.json", {"annotations": [
                {"frame_id": "f000000", "visible_observation": "Unchanged synthetic annotation."},
                {"frame_id": "f000003", "visible_observation": "Another unchanged synthetic annotation."}],
                "training_eligible": False})
            _write(directory / "summary.json", summary)
            (directory / "report.html").write_text("<!doctype html><title>Fixture annotation</title>")
            if self.after_case:
                self.after_case(case_id)
            return summary
        finally:
            self.active = False


class FakeIntegritySealer:
    """Exercise controller seal calls with byte-sensitive immutable test receipts."""

    def __init__(self):
        self.calls = []

    def __call__(self, output_dir):
        directory = Path(output_dir)
        self.calls.append(directory.name)
        hashes = {name: sha256_file(directory / name) for name in ("run.json", "annotations.json", "summary.json")}
        destination = directory / MANIFEST_NAME
        if destination.exists():
            require(read(destination)["sha256"] == hashes, "Sealed annotation evidence changed")
        else:
            _write(destination, {"sha256": hashes})
        return read(destination)


class AnnotationPairTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.dataset = root / "dataset"
        self.dataset.mkdir()
        self.batch = root / "selection-batch"
        self.batch.mkdir()
        self.output = root / "annotation-pair"
        self.frame_counts = {"S2A2": 4, "S1A2": 6}
        remaining = ["S1A3", "S2A3", "S3A1", "S3A2", "S3A3", "S4A1", "S4A2", "S4A3",
                     "S5A1", "S5A2", "S5A3", "S6A1", "S6A2", "S6A3", "S7A1", "S7A2",
                     "S7A3", "S8A1", "S8A2", "S8A3", "Clip0", "Clip1"]
        self.all_cases = [*CASES, *remaining]
        queue = {"dataset_root": str(self.dataset), "jobs": [
            {"case_id": case_id, "position": index, "frame_count": self.frame_counts.get(case_id, 8)}
            for index, case_id in enumerate(self.all_cases, start=1)]}
        _write(self.batch / "queue.json", queue)
        self.ready = {"writer_active": False, "jobs": [
            {"case_id": case_id, "status": "completed" if case_id in CASES else "pending"}
            for case_id in self.all_cases]}
        self.runner = FakeAnnotationRunner(self)
        self.sealer = FakeIntegritySealer()

    def run_controller(self, **kwargs):
        return run_pair(self.batch, self.output, annotation_runner=self.runner,
                        integrity_sealer=self.sealer, poll_seconds=0, progress=lambda _: None, **kwargs)

    def ready_call(self, **kwargs):
        with patch("yasargil.annotation_pair.selection_batch_status", return_value=copy.deepcopy(self.ready)):
            return self.run_controller(**kwargs)

    def test_waits_for_both_completed_and_writer_inactive_then_annotates_only_two(self):
        still_running = copy.deepcopy(self.ready)
        still_running["writer_active"] = True
        still_running["jobs"][1]["status"] = "running"
        both_saved_but_writer_active = copy.deepcopy(self.ready)
        both_saved_but_writer_active["writer_active"] = True
        not_started = copy.deepcopy(self.ready)
        not_started["jobs"][0]["status"] = "pending"
        not_started["jobs"][1]["status"] = "pending"
        def no_annotation_while_waiting(_):
            self.assertEqual(self.runner.calls, [])
            self.assertTrue(all(not (self.output / case).exists() for case in self.all_cases))
        with patch("yasargil.annotation_pair.selection_batch_status", side_effect=[
                not_started, still_running, both_saved_but_writer_active, self.ready]) as status, \
                patch("yasargil.annotation_pair.time.sleep", side_effect=no_annotation_while_waiting) as sleep:
            result = self.run_controller()
        self.assertEqual(status.call_count, 4)
        self.assertEqual(sleep.call_count, 3)
        self.assertEqual(result["status"], "completed")
        self.assertEqual([call["case_id"] for call in self.runner.calls], list(CASES))
        self.assertEqual(self.sealer.calls, list(CASES))
        self.assertTrue(all(not (self.output / case).exists() for case in self.all_cases[2:]))
        for call, job in zip(self.runner.calls, result["jobs"]):
            self.assertFalse(call["resume"])
            self.assertEqual(call["config"]["context_size"], 262144)
            self.assertEqual(call["config"]["request_timeout_seconds"], 21600)
            self.assertEqual(call["config"]["image_max_tokens"], 256)
            self.assertEqual(call["config"]["procedure_context"], PROCEDURE_CONTEXT)
            self.assertTrue(Path(job["integrity_path"]).is_file())

    def test_failure_of_first_annotation_does_not_prevent_second_and_resume_retries_only_first(self):
        self.runner = FakeAnnotationRunner(self, fail_once={CASES[0]})
        first = self.ready_call()
        self.assertEqual(first["status"], "completed_with_issues")
        self.assertEqual([job["status"] for job in first["jobs"]], ["failed", "completed"])
        self.assertIn("Synthetic annotation failure", first["jobs"][0]["error"])
        self.assertEqual([call["case_id"] for call in self.runner.calls], list(CASES))
        second = self.ready_call(resume=True)
        self.assertEqual(second["status"], "completed")
        self.assertEqual([call["case_id"] for call in self.runner.calls], [*CASES, CASES[0]])
        self.assertTrue(self.runner.calls[-1]["resume"])
        self.assertEqual(sorted(self.runner.inferences), sorted(CASES))

    def test_pause_before_any_call_and_explicit_resume(self):
        with patch("yasargil.annotation_pair.selection_batch_status") as status:
            paused = self.run_controller(should_stop=lambda: True)
        status.assert_not_called()
        self.assertEqual(paused["status"], "paused")
        self.assertEqual(self.runner.calls, [])
        self.assertEqual(self.sealer.calls, [])
        self.assertEqual(self.ready_call(resume=True)["status"], "completed")

    def test_successful_resume_checks_seals_without_running_annotations_again(self):
        self.ready_call()
        before = {case: (self.output / case / "annotations.json").read_bytes() for case in CASES}
        calls = copy.deepcopy(self.runner.calls)
        self.assertEqual(self.ready_call(resume=True)["status"], "completed")
        self.assertEqual(self.runner.calls, calls)
        self.assertEqual(self.sealer.calls, [*CASES, *CASES])
        self.assertEqual(before, {case: (self.output / case / "annotations.json").read_bytes() for case in CASES})

    def test_partial_preparation_is_archived_and_keeps_original_files(self):
        self.run_controller(prepare_only=True)
        partial = self.output / CASES[0]
        partial.mkdir()
        (partial / "partial-source.bin").write_bytes(b"retain this interrupted source preparation")
        self.assertEqual(self.ready_call(resume=True)["status"], "completed")
        archived = list((self.output / "interrupted-preparations").glob(f"{CASES[0]}-*/partial-source.bin"))
        self.assertEqual(len(archived), 1)
        self.assertEqual(archived[0].read_bytes(), b"retain this interrupted source preparation")
        self.assertFalse(self.runner.calls[0]["resume"])

    def test_queue_plan_and_scope_tampering_rejected_before_annotation(self):
        self.run_controller(prepare_only=True)
        for path, mutate, expected in (
            (self.batch / "queue.json", lambda data: data["jobs"].reverse(), "queue changed"),
            (self.output / "run.json", lambda data: data["config"].update(context_size=131072), "settings changed"),
            (self.output / "state.json", lambda data: data["jobs"].append({"case_id": "S1A3", "status": "pending"}), "scope changed"),
        ):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                data = json.loads(original)
                mutate(data)
                _write(path, data)
                with self.assertRaisesRegex(ContractError, expected):
                    self.ready_call(resume=True)
                path.write_bytes(original)
                self.assertEqual(self.runner.calls, [])

    def test_changed_completed_annotation_fails_integrity_without_any_new_inference(self):
        self.ready_call()
        calls = copy.deepcopy(self.runner.calls)
        path = self.output / CASES[0] / "annotations.json"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ContractError, "Sealed annotation evidence changed"):
            self.ready_call(resume=True)
        self.assertEqual(self.runner.calls, calls)
        self.assertEqual(read(self.output / "state.json")["status"], "failed")

    def test_unready_selection_or_attempt_outside_pair_prevents_annotations(self):
        self.run_controller(prepare_only=True)
        for mutation in ("failed", "needs_review", "third_case_started", "controller_failed"):
            with self.subTest(condition=mutation):
                status = copy.deepcopy(self.ready)
                controller = self.batch / "first-two-controller.json"
                if mutation in {"failed", "needs_review"}:
                    status["jobs"][0]["status"] = mutation
                elif mutation == "third_case_started":
                    status["jobs"][2]["status"] = "running"
                else:
                    _write(controller, {"status": "failed"})
                with patch("yasargil.annotation_pair.selection_batch_status", return_value=status):
                    with self.assertRaises(ContractError):
                        self.run_controller(resume=True)
                self.assertEqual(self.runner.calls, [])
                self.assertEqual(self.sealer.calls, [])
                controller.unlink(missing_ok=True)

    def test_context_conflict_is_sealed_and_recorded_as_completed_with_issues(self):
        self.runner = FakeAnnotationRunner(self, context_conflict={CASES[0]})
        result = self.ready_call()
        self.assertEqual(result["status"], "completed_with_issues")
        self.assertEqual([job["status"] for job in result["jobs"]], ["context_conflict", "completed"])
        self.assertEqual(self.sealer.calls, list(CASES))


if __name__ == "__main__":
    unittest.main()

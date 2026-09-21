"""Selection-based pair coordination never requires Qwen annotation drafts."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from yasargil.contract import ContractError, require, sha256_file
from yasargil.medgemma_pair import CASES, PROTOCOL, run_pair


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True))


class FakeAnnotationRunner:
    """Model-free runner retaining accepted output and verifying it on resume."""
    def __init__(self, test, *, fail_once=(), pause_once=(), interrupt_once=(), unresolved=()):
        self.test = test
        self.fail_once, self.pause_once = set(fail_once), set(pause_once)
        self.interrupt_once, self.unresolved = set(interrupt_once), set(unresolved)
        self.calls, self.inferences, self.expected_hashes = [], [], {}

    def __call__(self, selection_run, output_dir, config, *, resume, client, should_stop, progress):
        directory = Path(output_dir)
        case = directory.name
        self.test.assertIn(case, CASES)
        self.test.assertEqual(Path(selection_run), self.test.batch / "runs" / f"{CASES.index(case) + 1:02d}-{case}")
        self.test.assertFalse((Path(selection_run) / "annotations.json").exists())
        self.test.assertFalse(should_stop())
        self.calls.append({"case_id": case, "resume": resume, "config": asdict(config),
                           "selection_run": str(selection_run), "timeout": client.timeout})
        if resume:
            require(read(directory / "run.json")["config"] == asdict(config), "Resume settings differ")
            if case in self.expected_hashes:
                require(sha256_file(directory / "annotations.json") == self.expected_hashes[case],
                        "Accepted independent annotation bytes changed")
        else:
            self.test.assertFalse(directory.exists(), "Partial preparation must be archived before retry")
            directory.mkdir()
            write(directory / "run.json", {"selection_run": str(selection_run), "config": asdict(config)})
        if case in self.fail_once:
            self.fail_once.remove(case)
            raise RuntimeError("Synthetic annotation failure")
        if case in self.pause_once:
            self.pause_once.remove(case)
            return {"status": "paused"}
        if case in self.interrupt_once:
            self.interrupt_once.remove(case)
            raise KeyboardInterrupt("Synthetic annotation interruption")
        if case not in self.expected_hashes:
            write(directory / "annotations.json", {"annotations": [
                {"target_frame_id": "f0", "claims": []}, {"target_frame_id": "f1", "claims": []}]})
            self.expected_hashes[case] = sha256_file(directory / "annotations.json")
            self.inferences.append(case)
        return {"status": "completed_with_unresolved_questions" if case in self.unresolved else "completed",
                "annotated_frame_count": 2, "selected_frame_count": 2,
                "source_frame_count": 101 + CASES.index(case),
                "unresolved_frame_count": int(case in self.unresolved)}


class MedGemmaPairTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.batch = self.root / "selection-batch"
        self.batch.mkdir()
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        self.output = self.root / "annotation-pair"
        jobs = [{"case_id": case, "position": index, "frame_count": 100 + index}
                for index, case in enumerate((*CASES, "S1A3"), start=1)]
        write(self.batch / "queue.json", {"dataset_root": str(self.dataset), "jobs": jobs})
        self.ready = {"status": "paused", "writer_active": False,
                      "jobs": [{"case_id": case, "status": "completed" if case in CASES else "pending"}
                               for case in (*CASES, "S1A3")]}
        self.runner = FakeAnnotationRunner(self)
        patcher = patch("yasargil.medgemma_pair.selection_batch_status", return_value=self.ready)
        self.status = patcher.start()
        self.addCleanup(patcher.stop)

    def run_controller(self, **kwargs):
        return run_pair(self.batch, self.output, annotation_runner=self.runner, poll_seconds=0,
                        progress=lambda _: None, **kwargs)

    def test_prepare_uses_selection_queue_without_annotation_inputs_or_inference(self):
        result = self.run_controller(prepare_only=True)
        self.assertEqual(result["status"], "prepared")
        self.assertEqual(self.runner.calls, [])
        self.status.assert_not_called()
        plan = read(self.output / "run.json")
        self.assertEqual(plan["schema_version"], PROTOCOL)
        self.assertEqual(plan["selection_batch"], str(self.batch))
        self.assertEqual([row["case_id"] for row in plan["cases"]], list(CASES))
        self.assertEqual([row["expected_video_frames"] for row in plan["cases"]], [101, 102])
        self.assertEqual(plan["config"]["before_frames"], 2)
        self.assertEqual(plan["config"]["after_frames"], 2)
        self.assertTrue(plan["config"]["detail_crops"])
        self.assertFalse(any("annotation_run" in row for row in plan["cases"]))
        self.assertEqual(list(self.batch.rglob("annotations.json")), [])

    def test_waits_for_both_selections_and_inactive_selection_writer_only(self):
        pending = copy.deepcopy(self.ready)
        pending["jobs"][1]["status"] = "running"
        active = copy.deepcopy(self.ready)
        active["writer_active"] = True
        self.status.side_effect = [pending, active, self.ready]
        waits = []
        def waiting(_):
            self.assertEqual(self.runner.calls, [])
            waits.append(True)
        with patch("yasargil.medgemma_pair.time.sleep", side_effect=waiting):
            result = self.run_controller()
        self.assertEqual(len(waits), 2)
        self.assertEqual(result["status"], "completed")
        self.assertEqual([call["case_id"] for call in self.runner.calls], list(CASES))
        self.assertTrue(all(call["timeout"] == 1800 for call in self.runner.calls))
        self.assertTrue(all(not call["resume"] for call in self.runner.calls))
        self.assertEqual({path.name for path in self.output.iterdir() if path.is_dir()}, set(CASES))

    def test_first_failure_continues_and_resume_rechecks_both_cases(self):
        self.runner = FakeAnnotationRunner(self, fail_once={CASES[0]})
        first = self.run_controller()
        self.assertEqual(first["status"], "completed_with_issues")
        self.assertEqual([row["status"] for row in first["jobs"]], ["failed", "completed"])
        self.assertIn("Synthetic annotation failure", first["jobs"][0]["error"])
        saved = (self.output / CASES[1] / "annotations.json").read_bytes()
        self.assertEqual(self.run_controller(resume=True)["status"], "completed")
        self.assertEqual([call["case_id"] for call in self.runner.calls], [*CASES, *CASES])
        self.assertTrue(all(call["resume"] for call in self.runner.calls[2:]))
        self.assertEqual(len(self.runner.inferences), 2)
        self.assertEqual(saved, (self.output / CASES[1] / "annotations.json").read_bytes())

    def test_completed_resume_invokes_runner_verification_without_duplicate_inference(self):
        self.run_controller()
        original = {case: (self.output / case / "annotations.json").read_bytes() for case in CASES}
        self.assertEqual(self.run_controller(resume=True)["status"], "completed")
        self.assertEqual([call["case_id"] for call in self.runner.calls], [*CASES, *CASES])
        self.assertEqual(self.runner.inferences, list(CASES))
        self.assertEqual(original, {case: (self.output / case / "annotations.json").read_bytes() for case in CASES})

    def test_completed_annotation_tamper_is_reported_on_resume(self):
        self.run_controller()
        path = self.output / CASES[0] / "annotations.json"
        path.write_bytes(path.read_bytes() + b"\n")
        result = self.run_controller(resume=True)
        self.assertEqual(result["status"], "completed_with_issues")
        self.assertEqual([row["status"] for row in result["jobs"]], ["failed", "completed"])
        self.assertIn("annotation bytes changed", result["jobs"][0]["error"])
        self.assertEqual(self.runner.inferences, list(CASES))

    def test_stop_prevents_upstream_reads_and_explicit_resume_clears_pause_file(self):
        self.assertEqual(self.run_controller(should_stop=lambda: True)["status"], "paused")
        self.status.assert_not_called()
        self.assertEqual(self.runner.calls, [])
        marker = self.output / ".pause-requested"
        marker.touch()
        self.assertEqual(self.run_controller(resume=True)["status"], "completed")
        self.assertFalse(marker.exists())

    def test_new_pause_request_while_waiting_prevents_inference(self):
        pending = copy.deepcopy(self.ready)
        pending["jobs"][1]["status"] = "running"
        self.status.return_value = pending
        with patch("yasargil.medgemma_pair.time.sleep", side_effect=lambda _: (self.output / ".pause-requested").touch()):
            self.assertEqual(self.run_controller()["status"], "paused")
        self.assertEqual(self.runner.calls, [])

    def test_runner_pause_leaves_second_pending_then_resumes_first(self):
        self.runner = FakeAnnotationRunner(self, pause_once={CASES[0]})
        result = self.run_controller()
        self.assertEqual(result["status"], "paused")
        self.assertEqual([row["status"] for row in result["jobs"]], ["pending", "pending"])
        self.assertEqual(len(self.runner.calls), 1)
        self.assertEqual(self.run_controller(resume=True)["status"], "completed")
        self.assertTrue(self.runner.calls[1]["resume"])

    def test_keyboard_interruption_is_recorded_and_can_resume(self):
        self.runner = FakeAnnotationRunner(self, interrupt_once={CASES[0]})
        with self.assertRaises(KeyboardInterrupt):
            self.run_controller()
        self.assertEqual(read(self.output / "state.json")["status"], "paused")
        self.assertEqual(self.run_controller(resume=True)["status"], "completed")

    def test_unresolved_evidence_is_recorded_without_followups(self):
        self.runner = FakeAnnotationRunner(self, unresolved={CASES[0]})
        result = self.run_controller()
        self.assertEqual(result["status"], "completed_with_issues")
        self.assertEqual(result["jobs"][0]["status"], "completed_with_unresolved_questions")
        self.assertEqual(result["jobs"][0]["unresolved_frame_count"], 1)
        self.assertEqual(result["jobs"][1]["unresolved_frame_count"], 0)
        self.assertEqual(self.runner.inferences, list(CASES))

    def test_missing_target_result_fails_case_and_continues(self):
        runner = self.runner
        def incomplete(*args, **kwargs):
            summary = runner(*args, **kwargs)
            if Path(args[1]).name == CASES[0]:
                summary["annotated_frame_count"] = 1
            return summary
        self.runner = incomplete
        result = self.run_controller()
        self.assertEqual([row["status"] for row in result["jobs"]], ["failed", "completed"])
        self.assertIn("every selected frame", result["jobs"][0]["error"])

    def test_source_count_must_match_the_frozen_selection_queue(self):
        runner = self.runner
        def wrong_source(*args, **kwargs):
            summary = runner(*args, **kwargs)
            if Path(args[1]).name == CASES[0]:
                summary["source_frame_count"] = 999
            return summary
        self.runner = wrong_source
        result = self.run_controller()
        self.assertEqual([row["status"] for row in result["jobs"]], ["failed", "completed"])
        self.assertIn("source", result["jobs"][0]["error"].lower())

    def test_frozen_queue_config_scope_and_protocol_must_match(self):
        self.run_controller(prepare_only=True)
        for path, mutate, expected in (
            (self.batch / "queue.json", lambda value: value["jobs"].reverse(), "queue changed"),
            (self.output / "run.json", lambda value: value["config"].update(num_predict=2000), "configuration changed"),
            (self.output / "state.json", lambda value: value["jobs"].append({"case_id": "S1A3"}), "scope changed"),
        ):
            with self.subTest(path=path):
                original = path.read_bytes()
                value = json.loads(original)
                mutate(value)
                write(path, value)
                with self.assertRaisesRegex(ContractError, expected):
                    self.run_controller(resume=True)
                path.write_bytes(original)
                self.assertEqual(self.runner.calls, [])
        with patch("yasargil.medgemma_pair._protocol_hash", return_value="changed"):
            with self.assertRaisesRegex(ContractError, "configuration changed"):
                self.run_controller(resume=True)
        with self.assertRaisesRegex(ContractError, "Resume selection batch differs"):
            run_pair(self.root / "different", self.output, resume=True, annotation_runner=self.runner)

    def test_historical_qwen_review_pairs_cannot_resume(self):
        self.run_controller(prepare_only=True)
        original = read(self.output / "run.json")
        for changes in ({"schema_version": "sospine-first-two-medgemma-surgery-v1"}, {"runtime": "ollama"}):
            write(self.output / "run.json", {**original, **changes})
            state = read(self.output / "state.json")
            state["plan_sha256"] = sha256_file(self.output / "run.json")
            write(self.output / "state.json", state)
            with self.subTest(changes=changes), self.assertRaisesRegex(ContractError, "Historical Qwen-review"):
                self.run_controller(resume=True)
        self.assertEqual(self.runner.calls, [])

    def test_only_first_two_scope_is_authorized(self):
        invalid = copy.deepcopy(self.ready)
        invalid["jobs"][2]["status"] = "running"
        self.status.return_value = invalid
        with self.assertRaisesRegex(ContractError, "outside the authorized pair"):
            self.run_controller()
        self.assertEqual(self.runner.calls, [])
        self.assertEqual(read(self.output / "state.json")["status"], "failed")

    def test_wrong_initial_selection_order_fails_before_preparation(self):
        queue = read(self.batch / "queue.json")
        queue["jobs"][:2] = reversed(queue["jobs"][:2])
        write(self.batch / "queue.json", queue)
        with self.assertRaisesRegex(ContractError, "authorized S2A2 and S1A2"):
            self.run_controller()
        self.assertFalse(self.output.exists())

    def test_failed_upstream_selection_or_controller_prevents_annotation(self):
        self.run_controller(prepare_only=True)
        for failure in ("failed", "needs_review"):
            invalid = copy.deepcopy(self.ready)
            invalid["jobs"][0]["status"] = failure
            self.status.return_value = invalid
            with self.subTest(failure=failure), self.assertRaisesRegex(ContractError, "completed, ready selections"):
                self.run_controller(resume=True)
        self.status.return_value = self.ready
        write(self.batch / "first-two-controller.json", {"status": "needs_attention"})
        with self.assertRaisesRegex(ContractError, "controller needs attention"):
            self.run_controller(resume=True)
        self.assertEqual(self.runner.calls, [])

    def test_interrupted_preparation_is_preserved_before_retry(self):
        self.run_controller(prepare_only=True)
        partial = self.output / CASES[0]
        partial.mkdir()
        (partial / "evidence.bin").write_bytes(b"keep interrupted evidence preparation")
        self.assertEqual(self.run_controller(resume=True)["status"], "completed")
        archived = list((self.output / "interrupted-preparations").glob(f"{CASES[0]}-*/evidence.bin"))
        self.assertEqual(len(archived), 1)
        self.assertEqual(archived[0].read_bytes(), b"keep interrupted evidence preparation")


if __name__ == "__main__":
    unittest.main()

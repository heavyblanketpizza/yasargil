"""Scoped MedGemma coordination with sealed sources and durable fake requests."""

import copy
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from yasargil import medgemma_pair
from yasargil.contract import ContractError, require, sha256_file
from yasargil.medgemma_pair import CASES, run_pair
from yasargil.smart_selection import _write


def read(path):
    return json.loads(Path(path).read_bytes())


class FakeIntegrityVerifier:
    def __init__(self, test):
        self.test = test
        self.expected = {case: sha256_file(test.parent / case / "annotations.json") for case in CASES}
        self.calls = []

    def __call__(self, directory):
        directory = Path(directory)
        self.calls.append(directory.name)
        self.test.events.append(f"verify:{directory.name}")
        digest = sha256_file(directory / "annotations.json")
        require(digest == self.expected[directory.name], "Sealed Qwen annotation bytes changed")
        return {"annotation_count": 2, "source_annotation_sha256": digest}


class FakeReviewRunner:
    """Create one request per surgery; recovery reuses its accepted response."""

    def __init__(self, test, *, fail_once=(), pause_once=(), deferred=(), after_review=None):
        self.test = test
        self.fail_once = set(fail_once)
        self.pause_once = set(pause_once)
        self.deferred = set(deferred)
        self.after_review = after_review
        self.calls = []
        self.inferences = []
        self.active = False

    def __call__(self, annotation_run, output_dir, config, *, resume, client, should_stop, progress):
        directory = Path(output_dir)
        case = directory.name
        self.test.assertIn(case, CASES)
        self.test.assertFalse(self.active)
        self.active = True
        try:
            self.test.assertEqual(set(self.test.verifier.calls) | set(self.test.intake.calls), set(CASES),
                                  "Both Qwen outputs must be verified before any model request")
            self.test.assertFalse(medgemma_pair._active(self.test.parent / ".pair.lock"))
            self.test.assertFalse(should_stop())
            self.test.events.append(f"review:{case}")
            self.calls.append({"case_id": case, "resume": resume, "config": asdict(config),
                               "annotation_run": str(annotation_run), "timeout": client.timeout})
            if resume:
                require(read(directory / "run.json")["config"] == asdict(config), "Resume settings differ")
            else:
                self.test.assertFalse(directory.exists(), "Partial preparations must not be overwritten")
                directory.mkdir()
                _write(directory / "run.json", {"annotation_run": str(annotation_run), "config": asdict(config)})
            if case in self.fail_once:
                self.fail_once.remove(case)
                raise RuntimeError("Synthetic MedGemma review failure")
            target = directory / "calls" / "surgery"
            request = {"case_id": case, "frame_ids": ["f000000", "f000001"], "config": asdict(config)}
            response = {"reviews": [
                {"target_frame_id": f"f{index:06d}", "status":
                 "needs_more_evidence" if case in self.deferred and index == 0 else "review_complete"}
                for index in range(2)]}
            if target.exists():
                require(read(target / "request.json") == request, "Accepted raw request changed")
                require(read(target / "response.json") == response, "Accepted raw response changed")
            else:
                target.mkdir(parents=True)
                _write(target / "request.json", request)
                _write(target / "response.json", response)
                self.inferences.append(case)
            if case in self.pause_once:
                self.pause_once.remove(case)
                raise KeyboardInterrupt("Synthetic interruption after accepted whole-surgery response")
            reviews = response["reviews"]
            summary = {"status": "completed_with_deferred_evidence" if case in self.deferred else "completed",
                       "reviewed_frame_count": 2, "selected_frame_count": 2,
                       "deferred_frame_count": int(case in self.deferred)}
            _write(directory / "reviews.json", reviews)
            _write(directory / "summary.json", summary)
            (directory / "report.html").write_text("<!doctype html><title>Fake review</title>")
            if self.after_review:
                self.after_review(case)
            return summary
        finally:
            self.active = False


class FakeRejectedIntake:
    """An audited loader accepts only its unchanged synthetic rejected drafts."""

    def __init__(self, test):
        self.test, self.calls, self.invalid_cases = test, [], set()

    def __call__(self, directory):
        directory = Path(directory)
        case = directory.name
        self.calls.append(case)
        self.test.events.append(f"audit:{case}")
        require(case not in self.invalid_cases, "Failure was not exclusively temporal citations")
        digest = sha256_file(directory / "annotations.json")
        audit = {"schema_version": "qwen-rejected-annotation-intake-v1",
                 "status": "rejected_temporal_citations", "validation_status": "rejected_temporal_citations",
                 "annotation_count": 2, "frame_ids": ["f000000", "f000001"],
                 "artifact_sha256": {"annotations.json": digest}, "raw_response_sha256": digest,
                 "issues": [{"code": "timestamp_out_of_range"}], "training_eligible": False}
        return {}, {}, read(directory / "annotations.json"), audit


class MedGemmaPairTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.parent = root / "annotation-pair"
        self.parent.mkdir()
        self.output = root / "medgemma-pair"
        self.events = []
        _write(self.parent / "run.json", {"cases": [{"case_id": case} for case in CASES]})
        self.ready = {"status": "completed", "jobs": [{"case_id": case, "status": "completed"} for case in CASES]}
        _write(self.parent / "state.json", self.ready)
        (self.parent / ".pair.lock").touch()
        for case in CASES:
            (self.parent / case).mkdir()
            _write(self.parent / case / "annotations.json", {"annotations": [
                {"frame_id": "f000000", "visible_observation": "Original Qwen annotation."},
                {"frame_id": "f000001", "visible_observation": "Second original Qwen annotation."}]})
        self.verifier = FakeIntegrityVerifier(self)
        self.intake = FakeRejectedIntake(self)
        self.runner = FakeReviewRunner(self)

    def run_controller(self, **kwargs):
        return run_pair(self.parent, self.output, review_runner=self.runner,
                        integrity_verifier=self.verifier, poll_seconds=0, progress=lambda _: None, **kwargs)

    def test_waits_for_both_finished_and_actual_qwen_worker_lock_before_model_calls(self):
        pending = copy.deepcopy(self.ready)
        pending["status"] = "running"
        pending["jobs"][1]["status"] = "running"
        _write(self.parent / "state.json", pending)
        pauses = []
        with (self.parent / ".pair.lock").open("a") as parent_lock:
            fcntl.flock(parent_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            def advance(_):
                self.assertEqual(self.runner.calls, [])
                self.assertEqual(self.verifier.calls, [])
                pauses.append(True)
                if len(pauses) == 1:
                    _write(self.parent / "state.json", self.ready)
                else:
                    fcntl.flock(parent_lock, fcntl.LOCK_UN)
            with patch("yasargil.medgemma_pair.time.sleep", side_effect=advance):
                result = self.run_controller()
        self.assertEqual(len(pauses), 2)
        self.assertEqual(result["status"], "completed")
        self.assertEqual([call["case_id"] for call in self.runner.calls], list(CASES))
        self.assertEqual(self.events, [f"verify:{CASES[0]}", f"verify:{CASES[1]}",
                                      f"verify:{CASES[0]}", f"review:{CASES[0]}", f"verify:{CASES[0]}",
                                      f"verify:{CASES[1]}", f"review:{CASES[1]}", f"verify:{CASES[1]}"])
        self.assertEqual({p.name for p in self.output.iterdir() if p.is_dir()}, set(CASES))
        for call in self.runner.calls:
            self.assertFalse(call["resume"])
            self.assertFalse({"max_context_frames", "before_frames", "after_frames"} & set(call["config"]))
            self.assertEqual(call["config"]["num_ctx"], 65536)
            self.assertEqual(call["config"]["num_predict"], 16384)
            self.assertEqual(call["timeout"], 21600)
        self.assertEqual(self.runner.inferences, list(CASES))
        self.assertEqual(read(self.output / "run.json")["schema_version"],
                         "sospine-first-two-medgemma-surgery-v1")

    def test_first_failure_does_not_prevent_second_case_and_resume_reuses_accepted_requests(self):
        self.runner = FakeReviewRunner(self, fail_once={CASES[0]})
        first = self.run_controller()
        self.assertEqual(first["status"], "completed_with_issues")
        self.assertEqual([job["status"] for job in first["jobs"]], ["failed", "completed"])
        self.assertIn("Synthetic MedGemma review failure", first["jobs"][0]["error"])
        self.assertEqual([call["case_id"] for call in self.runner.calls], list(CASES))
        existing = {p: p.read_bytes() for p in (self.output / CASES[1] / "calls").rglob("*.json")}
        second = self.run_controller(resume=True)
        self.assertEqual(second["status"], "completed")
        self.assertEqual(len(self.runner.inferences), 2)
        self.assertEqual(len(set(self.runner.inferences)), 2)
        self.assertEqual(existing, {p: p.read_bytes() for p in existing})

    def test_interrupted_accepted_surgery_response_resumes_without_duplication_or_replacement(self):
        self.runner = FakeReviewRunner(self, pause_once={CASES[0]})
        with self.assertRaises(KeyboardInterrupt):
            self.run_controller()
        first = read(self.output / "state.json")
        self.assertEqual(first["status"], "paused")
        self.assertEqual([job["status"] for job in first["jobs"]], ["running", "pending"])
        self.assertEqual(self.runner.inferences, [CASES[0]])
        accepted = self.output / CASES[0] / "calls" / "surgery"
        original = {p: p.read_bytes() for p in accepted.iterdir()}
        second = self.run_controller(resume=True)
        self.assertEqual(second["status"], "completed")
        self.assertEqual(self.runner.inferences, list(CASES))
        self.assertEqual(original, {p: p.read_bytes() for p in original})
        self.assertTrue(self.runner.calls[1]["resume"])

    def test_pause_before_source_verification_or_inference_then_resume(self):
        paused = self.run_controller(should_stop=lambda: True)
        self.assertEqual(paused["status"], "paused")
        self.assertEqual(self.events, [])
        self.assertEqual(self.run_controller(resume=True)["status"], "completed")

    def test_changed_second_source_seal_prevents_all_model_calls(self):
        source = self.parent / CASES[1] / "annotations.json"
        source.write_bytes(source.read_bytes() + b"\n")
        with self.assertRaisesRegex(ContractError, "Sealed Qwen annotation bytes changed"):
            self.run_controller()
        self.assertEqual(self.runner.calls, [])
        self.assertEqual(self.verifier.calls, list(CASES))

    def test_source_integrity_is_rechecked_after_each_completed_review(self):
        def change_after_first(case):
            if case == CASES[0]:
                source = self.parent / case / "annotations.json"
                source.write_bytes(source.read_bytes() + b"\n")
        self.runner = FakeReviewRunner(self, after_review=change_after_first)
        result = self.run_controller()
        self.assertEqual(result["status"], "completed_with_issues")
        self.assertEqual([job["status"] for job in result["jobs"]], ["failed", "completed"])
        self.assertIn("Sealed Qwen annotation bytes changed", result["jobs"][0]["error"])
        self.assertEqual(self.verifier.calls, [*CASES, CASES[0], CASES[0], CASES[1], CASES[1]])

    def test_frozen_parent_queue_config_scope_and_prompt_are_checked_before_model_calls(self):
        self.run_controller(prepare_only=True)
        for path, mutate, expected in (
            (self.parent / "run.json", lambda value: value["cases"].reverse(), "Parent annotation plan changed"),
            (self.output / "run.json", lambda value: value["config"].update(num_predict=8192), "configuration changed"),
            (self.output / "state.json", lambda value: value["jobs"].append({"case_id": "S1A3", "status": "pending"}), "scope changed"),
        ):
            with self.subTest(path=path):
                original = path.read_bytes()
                value = json.loads(original)
                mutate(value)
                _write(path, value)
                with self.assertRaisesRegex(ContractError, expected):
                    self.run_controller(resume=True)
                path.write_bytes(original)
                self.assertEqual(self.runner.calls, [])
        with patch("yasargil.medgemma_pair._protocol_hash", return_value="changed-review-prompt"):
            with self.assertRaisesRegex(ContractError, "review prompt changed"):
                self.run_controller(resume=True)
        self.assertEqual(self.runner.calls, [])

    def test_deferred_evidence_is_reported_without_followup_requests(self):
        self.runner = FakeReviewRunner(self, deferred={CASES[0]})
        result = self.run_controller()
        self.assertEqual(result["status"], "completed_with_issues")
        self.assertEqual(result["jobs"][0]["status"], "completed_with_deferred_evidence")
        self.assertEqual(result["jobs"][0]["deferred_frames"], 1)
        self.assertEqual(result["jobs"][1]["deferred_frames"], 0)
        self.assertEqual(self.runner.inferences, list(CASES))

    def test_joint_response_missing_a_target_fails_without_a_per_frame_fallback(self):
        runner = self.runner
        def incomplete(*args, **kwargs):
            summary = runner(*args, **kwargs)
            if Path(args[1]).name == CASES[0]:
                summary["reviewed_frame_count"] = 1
            return summary
        self.runner = incomplete
        result = self.run_controller()
        self.assertEqual(result["status"], "completed_with_issues")
        self.assertEqual([job["status"] for job in result["jobs"]], ["failed", "completed"])
        self.assertIn("every annotated key frame", result["jobs"][0]["error"])
        self.assertEqual([call["case_id"] for call in runner.calls], list(CASES))
        self.assertEqual(runner.inferences, list(CASES))

    def test_completed_surgery_review_resume_verifies_without_another_inference(self):
        self.run_controller()
        original = {p: p.read_bytes() for case in CASES
                    for p in (self.output / case / "calls").rglob("*.json")}
        self.assertEqual(self.run_controller(resume=True)["status"], "completed")
        self.assertEqual(self.runner.inferences, list(CASES))
        self.assertEqual(original, {p: p.read_bytes() for p in original})

    def test_completed_review_bundle_tamper_is_not_silently_reconstructed(self):
        self.run_controller()
        calls = copy.deepcopy(self.runner.calls)
        path = self.output / CASES[0] / "reviews.json"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ContractError, "review bundle changed"):
            self.run_controller(resume=True)
        self.assertEqual(self.runner.calls, calls)

    def test_legacy_per_frame_pair_is_not_resumed_as_a_surgery_pair(self):
        self.run_controller(prepare_only=True)
        plan = read(self.output / "run.json")
        plan["schema_version"] = "sospine-first-two-medgemma-v1"
        _write(self.output / "run.json", plan)
        state = read(self.output / "state.json")
        state["plan_sha256"] = sha256_file(self.output / "run.json")
        _write(self.output / "state.json", state)
        with self.assertRaisesRegex(ContractError, "configuration changed"):
            self.run_controller(resume=True)
        self.assertEqual(self.runner.calls, [])
        self.assertEqual(self.verifier.calls, [])

    def test_pre_migration_pair_cannot_resume_with_llama_cpp(self):
        self.run_controller(prepare_only=True)
        plan = read(self.output / "run.json")
        plan.pop("runtime")
        _write(self.output / "run.json", plan)
        state = read(self.output / "state.json")
        state["plan_sha256"] = sha256_file(self.output / "run.json")
        _write(self.output / "state.json", state)
        with self.assertRaisesRegex(ContractError, "predates the llama.cpp migration"):
            self.run_controller(resume=True)
        self.assertEqual(self.runner.calls, [])
        self.assertEqual(self.verifier.calls, [])

    def test_partial_review_preparation_is_preserved_before_retry(self):
        self.run_controller(prepare_only=True)
        partial = self.output / CASES[0]
        partial.mkdir()
        (partial / "frozen-packet.bin").write_bytes(b"keep interrupted review preparation")
        self.assertEqual(self.run_controller(resume=True)["status"], "completed")
        archived = list((self.output / "interrupted-preparations").glob(f"{CASES[0]}-*/frozen-packet.bin"))
        self.assertEqual(len(archived), 1)
        self.assertEqual(archived[0].read_bytes(), b"keep interrupted review preparation")

    def test_failed_upstream_annotation_prevents_medgemma_start(self):
        failed = copy.deepcopy(self.ready)
        failed["jobs"][1]["status"] = "failed"
        _write(self.parent / "state.json", failed)
        with self.assertRaisesRegex(ContractError, "Qwen annotation needs attention"):
            self.run_controller()
        self.assertEqual(self.events, [])

    def failed_parent(self):
        failed = {"status": "failed", "jobs": [{"case_id": case, "status": "failed"} for case in CASES]}
        _write(self.parent / "state.json", failed)
        return failed

    def test_explicit_override_accepts_both_audited_failures_and_pins_their_exact_intake(self):
        self.failed_parent()
        original = {path: path.read_bytes() for path in self.parent.rglob("*.json")}
        with patch("yasargil.qwen_draft_intake.load_rejected_annotation", side_effect=self.intake):
            result = self.run_controller(allow_rejected_temporal_citations=True)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.verifier.calls, [], "Rejected drafts must not acquire completed annotation seals")
        self.assertEqual(self.intake.calls, [*CASES, CASES[0], CASES[0], CASES[1], CASES[1]])
        self.assertEqual(self.runner.inferences, list(CASES))
        self.assertTrue(all(call["config"]["allow_rejected_temporal_citations"] for call in self.runner.calls))
        evidence = read(self.output / "qwen-evidence.json")
        self.assertEqual(set(evidence), set(CASES))
        for case, value in evidence.items():
            self.assertEqual(value["kind"], "rejected_temporal_citations")
            self.assertEqual(value["annotation_count"], 2)
            self.assertEqual(value["audit"]["artifact_sha256"]["annotations.json"],
                             sha256_file(self.parent / case / "annotations.json"))
        self.assertEqual(result["qwen_evidence_sha256"], sha256_file(self.output / "qwen-evidence.json"))
        self.assertEqual(original, {path: path.read_bytes() for path in original})

    def test_override_still_waits_for_both_terminal_jobs_and_inactive_parent_lock(self):
        failed = self.failed_parent()
        pending = copy.deepcopy(failed)
        pending["jobs"][1]["status"] = "running"
        _write(self.parent / "state.json", pending)
        waits = []
        with (self.parent / ".pair.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            def advance(_):
                self.assertEqual(self.runner.calls, [])
                self.assertEqual(self.intake.calls, [])
                waits.append(True)
                if len(waits) == 1:
                    _write(self.parent / "state.json", failed)
                else:
                    fcntl.flock(lock, fcntl.LOCK_UN)
            with patch("yasargil.medgemma_pair.time.sleep", side_effect=advance), \
                    patch("yasargil.qwen_draft_intake.load_rejected_annotation", side_effect=self.intake):
                result = self.run_controller(allow_rejected_temporal_citations=True)
        self.assertEqual(len(waits), 2)
        self.assertEqual(result["status"], "completed")

    def test_override_does_not_bypass_completed_seals_or_accept_other_failures(self):
        failed = self.failed_parent()
        failed["jobs"][0]["status"] = "context_conflict"
        _write(self.parent / "state.json", failed)
        self.intake.invalid_cases.add(CASES[1])
        with patch("yasargil.qwen_draft_intake.load_rejected_annotation", side_effect=self.intake):
            with self.assertRaisesRegex(ContractError, "not exclusively temporal"):
                self.run_controller(allow_rejected_temporal_citations=True)
        self.assertEqual(self.verifier.calls, [CASES[0]])
        self.assertEqual(self.intake.calls, [CASES[1]])
        self.assertEqual(self.runner.calls, [])

    def test_rejected_intake_is_checked_again_after_review_and_on_resume(self):
        self.failed_parent()
        def change_after_first(case):
            if case == CASES[0]:
                path = self.parent / case / "annotations.json"
                path.write_bytes(path.read_bytes() + b"\n")
        self.runner = FakeReviewRunner(self, after_review=change_after_first)
        with patch("yasargil.qwen_draft_intake.load_rejected_annotation", side_effect=self.intake):
            result = self.run_controller(allow_rejected_temporal_citations=True)
            self.assertEqual([job["status"] for job in result["jobs"]], ["failed", "completed"])
            self.assertIn("changed during MedGemma", result["jobs"][0]["error"])
            calls = copy.deepcopy(self.runner.calls)
            with self.assertRaisesRegex(ContractError, "intake evidence changed"):
                self.run_controller(resume=True)
        self.assertEqual(self.runner.calls, calls)

    def test_frozen_override_inherits_on_resume_and_rejects_explicit_changes(self):
        self.failed_parent()
        self.run_controller(prepare_only=True, allow_rejected_temporal_citations=True)
        with self.assertRaisesRegex(ContractError, "override differs"):
            self.run_controller(resume=True, allow_rejected_temporal_citations=False)
        self.assertEqual(self.runner.calls, [])
        with patch("yasargil.qwen_draft_intake.load_rejected_annotation", side_effect=self.intake):
            self.assertEqual(self.run_controller(resume=True)["status"], "completed")
            self.assertEqual(self.run_controller(resume=True)["status"], "completed")
        self.assertEqual(self.runner.inferences, list(CASES))

    def test_pinned_rejected_audit_bytes_cannot_change_on_resume(self):
        self.failed_parent()
        with patch("yasargil.qwen_draft_intake.load_rejected_annotation", side_effect=self.intake):
            self.run_controller(allow_rejected_temporal_citations=True)
            path = self.output / "qwen-evidence.json"
            path.write_bytes(path.read_bytes() + b"\n")
            calls = copy.deepcopy(self.runner.calls)
            with self.assertRaisesRegex(ContractError, "intake evidence bytes changed"):
                self.run_controller(resume=True)
        self.assertEqual(self.runner.calls, calls)


if __name__ == "__main__":
    unittest.main()

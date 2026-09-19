"""Durable batch execution with real image inventories and a fake review runner."""

import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from yasargil import selection_batch
from yasargil.contract import ContractError, canonical_hash
from yasargil.selection_batch import (
    PROCEDURE_CONTEXT, pause_selection_batch, run_selection_batch, selection_batch_status,
)
from yasargil.smart_selection import SelectionConfig, _write


def read(path):
    return json.loads(Path(path).read_text())


class SimulatedProcessDeath(BaseException):
    pass


class FakeSelectionRunner:
    """Stand in for the independently tested per-case source/review pipeline.

    Accepted outputs survive a process death and resume without another fake
    inference. Each invocation observes the actual persisted batch status.
    """

    def __init__(self, test, *, fail_once=(), after_case=None, alter_result=None):
        self.test = test
        self.fail_once = set(fail_once)
        self.after_case = after_case
        self.alter_result = alter_result
        self.calls = []
        self.inferences = []
        self.active = False

    def __call__(self, input_path, output_dir, config, *, released_fps, resume, progress):
        case_id = Path(input_path).name
        self.test.assertFalse(self.active, "Per-case workers must run serially")
        self.active = True
        try:
            status = selection_batch_status(self.test.output)
            self.test.assertTrue(status["writer_active"])
            self.test.assertEqual(status["active_case"], case_id)
            self.test.assertEqual(status["counts"]["running"], 1)
            self.test.assertEqual(config.procedure_context, PROCEDURE_CONTEXT)
            self.test.assertEqual(config.review_mode, "keep_drop")
            self.test.assertEqual(config.max_retrieval_rounds, 0)
            self.test.assertEqual(released_fps, 1)
            self.calls.append({"case_id": case_id, "resume": resume,
                               "context": config.procedure_context, "config": asdict(config)})
            output = Path(output_dir)
            if resume:
                self.test.assertTrue((output / "run.json").is_file())
                self.test.assertEqual(read(output / "run.json")["config"], asdict(config))
                if (output / "selection.json").is_file():
                    return read(output / "selection.json")
            else:
                self.test.assertFalse(output.exists(), "Never overwrite a partial preparation")
                output.mkdir(parents=True)
                _write(output / "run.json", {
                    "schema_version": "smart-frame-selection-run-v1",
                    "input_path": str(Path(input_path).resolve()), "config": asdict(config),
                    "config_sha256": canonical_hash(asdict(config)), "released_fps": released_fps,
                    "native_video_fps": 0, "temporal_exposure": "retrospective_full_video",
                })
            if case_id in self.fail_once:
                self.fail_once.remove(case_id)
                raise RuntimeError("Synthetic first-attempt inference failure")
            self.inferences.append(case_id)
            images = sorted(Path(input_path).glob("*.png"))
            candidates = [images[0], images[-1]][:config.candidate_budget]
            frames = [{
                "frame_id": f"f{images.index(image):06d}",
                "frame_index": images.index(image),
                "timestamp_ms": images.index(image) * 1000,
                "timestamp_basis": "reconstructed_nominal",
                "image_path": str(image.resolve()),
                "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "source_path": str(Path(input_path).resolve()),
                "model_decision": "keep", "effective_decision": "keep",
                "model_reason": "Synthetic fixture decision", "protected_temporal_anchor": True,
            } for image in candidates]
            result = {
                "schema_version": "smart-frame-selection-v1", "status": "completed",
                "source_manifest": str(output / "source" / "source.json"),
                "source_kind": "released_image_sequence", "expected_video_frames": len(images),
                "timestamp_basis": "reconstructed_nominal",
                "selector_exposure": "retrospective_full_video", "native_video_required_every_round": True,
                "completed_rounds_full_video_verified": True,
                "clinical_validation": "not_performed", "training_eligible": False,
                "scene_summary": "Synthetic fixture review", "context_check": "consistent",
                "frames": frames, "selected_frame_ids": [frame["frame_id"] for frame in frames],
                "rounds": [{"verification": {"full_source_video_verified": True}}],
                "unresolved_searches": [],
            }
            if self.alter_result:
                self.alter_result(case_id, result)
            _write(output / "selection.json", result)
            (output / "selection.html").write_text("<!doctype html><title>Fixture selection</title>")
            if self.after_case:
                self.after_case(case_id, output)
            return result
        finally:
            self.active = False


class SelectionBatchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        self.dataset = self.workspace / "dataset"
        self.output = self.workspace / "batch"
        frames = self.dataset / "frames"
        frames.mkdir(parents=True)
        # Four known outcomes permit two complete alternations; Clip0 is last.
        # Deliberately different repair times do not influence the default order.
        rows = [("S2A1", "N", 12321), ("S1A1", "Y", 11),
                ("S2A2", "N", 7), ("S1A2", "Y", 54321), ("Clip0", "", "")]
        self.case_order = ["S2A1", "S1A1", "S2A2", "S1A2", "Clip0"]
        with (self.dataset / "sospine_outcomes.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Trial ID", "Leak At 40mmHg", "Time for repair"])
            writer.writerows(rows)
        for case_id, _, _ in rows:
            directory = frames / case_id
            directory.mkdir()
            for index in range(1, 5):
                Image.new("RGB", (8, 8), (index * 40, 0, 255 - index * 40)).save(
                    directory / f"{case_id}_frame_{index:08d}.png")
        self.config = SelectionConfig(candidate_budget=2, max_candidates=4,
                                      context_size=4096, max_tokens=512,
                                      procedure_context=PROCEDURE_CONTEXT)

    def prepare(self):
        return run_selection_batch(self.dataset, self.output, self.config,
                                   prepare_only=True, progress=lambda _: None)

    def start(self, runner, **kwargs):
        return run_selection_batch(self.dataset, self.output, self.config,
                                   selection_runner=runner, progress=lambda _: None, **kwargs)

    def resume(self, runner, **kwargs):
        return run_selection_batch(None, self.output, resume=True,
                                   selection_runner=runner, progress=lambda _: None, **kwargs)

    def test_sequential_outcome_order_is_frozen_and_does_not_enter_model_context(self):
        runner = FakeSelectionRunner(self)
        result = self.start(runner)
        self.assertEqual(result["status"], "completed")
        self.assertEqual([call["case_id"] for call in runner.calls], self.case_order)
        self.assertTrue(all(not call["resume"] for call in runner.calls))
        self.assertEqual(result["counts"]["completed"], 5)
        self.assertEqual(len(set(call["context"] for call in runner.calls)), 1)
        configurations = json.dumps([call["config"] for call in runner.calls])
        for value in ("12321", "54321", "Leak At 40mmHg", "repair_time_seconds", "metadata_locator"):
            self.assertNotIn(value, configurations)
        queue = read(self.output / "queue.json")
        self.assertEqual([job["leak_result"] for job in queue["jobs"]], ["N", "Y", "N", "Y", None])
        self.assertEqual({job["duration_ms"] for job in queue["jobs"]}, {4000})
        self.assertTrue((self.output / "report.html").is_file())
        self.assertFalse(selection_batch_status(self.output)["writer_active"])
        unchanged = [call.copy() for call in runner.calls]
        self.assertEqual(self.resume(runner)["status"], "completed")
        self.assertEqual(runner.calls, unchanged, "Completed selections must not be inferred again")

    def test_failure_continues_then_explicit_resume_retries_only_failed_case(self):
        runner = FakeSelectionRunner(self, fail_once={"S1A1"})
        first = self.start(runner)
        self.assertEqual(first["status"], "completed_with_issues")
        self.assertEqual(first["counts"]["failed"], 1)
        self.assertEqual(first["counts"]["completed"], 4)
        self.assertEqual([call["case_id"] for call in runner.calls], self.case_order)
        self.assertIn("Synthetic first-attempt", first["jobs"][1]["error"])
        second = self.resume(runner)
        self.assertEqual(second["status"], "completed")
        self.assertEqual(runner.calls[-1]["case_id"], "S1A1")
        self.assertTrue(runner.calls[-1]["resume"])
        self.assertEqual(len(runner.calls), 6)
        self.assertEqual(sorted(runner.inferences), sorted(self.case_order))

    def test_cooperative_pause_before_first_case_and_resume(self):
        runner = FakeSelectionRunner(self)
        stopped = self.start(runner, should_stop=lambda: True)
        self.assertEqual(stopped["status"], "paused")
        self.assertEqual(stopped["counts"]["pending"], 5)
        self.assertEqual(runner.calls, [])
        result = self.resume(runner)
        self.assertEqual(result["status"], "completed")
        self.assertEqual([call["case_id"] for call in runner.calls], self.case_order)

    def test_pause_during_case_finishes_that_case_and_leaves_next_pending(self):
        def pause_after_first(case_id, directory):
            if case_id == self.case_order[0]:
                status = pause_selection_batch(self.output)
                self.assertTrue(status["pause_requested"])
                self.assertTrue(status["writer_active"])
        runner = FakeSelectionRunner(self, after_case=pause_after_first)
        stopped = self.start(runner)
        self.assertEqual(stopped["status"], "paused")
        self.assertEqual(stopped["counts"]["completed"], 1)
        self.assertEqual(stopped["counts"]["pending"], 4)
        self.assertIsNone(stopped["active_case"])
        result = self.resume(runner)
        self.assertEqual(result["status"], "completed")
        self.assertEqual([call["case_id"] for call in runner.calls], self.case_order)
        self.assertFalse(selection_batch_status(self.output)["pause_requested"])

    def test_accepted_case_survives_process_death_before_batch_checkpoint(self):
        runner = FakeSelectionRunner(self)
        original_save = selection_batch._save
        def crash_before_checkpoint(output, plan, state):
            if state["jobs"][0]["status"] == "completed":
                raise SimulatedProcessDeath("Crash after case files were saved")
            return original_save(output, plan, state)
        with patch.object(selection_batch, "_save", side_effect=crash_before_checkpoint):
            with self.assertRaises(SimulatedProcessDeath):
                self.start(runner)
        self.assertEqual(read(self.output / "state.json")["jobs"][0]["status"], "running")
        self.assertTrue((self.output / "runs" / "01-S2A1" / "selection.json").is_file())
        result = self.resume(runner)
        self.assertEqual(result["status"], "completed")
        self.assertEqual([call["case_id"] for call in runner.calls], [self.case_order[0], *self.case_order])
        self.assertTrue(runner.calls[1]["resume"])
        self.assertEqual(runner.inferences, self.case_order)

    def test_interrupted_partial_preparation_is_archived_without_overwriting_files(self):
        self.prepare()
        partial = self.output / "runs" / "01-S2A1"
        partial.mkdir(parents=True)
        sentinel = partial / "source-partial.bin"
        sentinel.write_bytes(b"irreplaceable interrupted preparation evidence")
        runner = FakeSelectionRunner(self)
        self.assertEqual(self.resume(runner)["status"], "completed")
        preserved = list((self.output / "interrupted-preparations").glob("01-S2A1-*/source-partial.bin"))
        self.assertEqual(len(preserved), 1)
        self.assertEqual(preserved[0].read_bytes(), b"irreplaceable interrupted preparation evidence")
        self.assertFalse(runner.calls[0]["resume"])

    def test_changed_queue_plan_and_job_order_are_rejected_before_any_runner_call(self):
        self.prepare()
        runner = FakeSelectionRunner(self)
        for filename, mutate, expected in (
            ("queue.json", lambda value: value["jobs"].reverse(), "queue changed"),
            ("run.json", lambda value: value["config"].update(image_max_tokens=512), "settings changed"),
            ("state.json", lambda value: value["jobs"].reverse(), "job order changed"),
            ("state.json", lambda value: value["jobs"][0].update(status="silently_skipped"), "Unknown batch job status"),
        ):
            with self.subTest(filename=filename):
                path = self.output / filename
                original = path.read_bytes()
                changed = json.loads(original)
                mutate(changed)
                _write(path, changed)
                with self.assertRaisesRegex(ContractError, expected):
                    self.resume(runner)
                self.assertEqual(runner.calls, [])
                path.write_bytes(original)

    def test_changed_dataset_metadata_or_inventory_rejects_frozen_queue(self):
        self.prepare()
        runner = FakeSelectionRunner(self)
        for path in (self.dataset / "sospine_outcomes.csv",
                     self.dataset / "frames" / "S2A1" / "S2A1_frame_00000001.png"):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b"\n")
                with self.assertRaisesRegex(ValueError, "differs"):
                    self.resume(runner)
                self.assertEqual(runner.calls, [])
                path.write_bytes(original)

    def test_completed_selection_hash_rejects_changed_draft_on_resume(self):
        runner = FakeSelectionRunner(self)
        self.start(runner)
        calls = len(runner.calls)
        output = self.output / "runs" / "01-S2A1" / "selection.json"
        changed = read(output)
        changed["selected_frame_ids"] = []
        _write(output, changed)
        with self.assertRaisesRegex(ContractError, "Saved selection changed"):
            self.resume(runner)
        self.assertEqual(len(runner.calls), calls)

    def test_incomplete_coverage_or_added_candidates_fail_without_stopping_later_cases(self):
        def alter(case_id, result):
            if case_id == "S2A1":
                result["completed_rounds_full_video_verified"] = False
            elif case_id == "S1A1":
                result["unresolved_searches"] = [{"start_ms": 0, "end_ms": 1000}]
            elif case_id == "S2A2":
                result["frames"].append({"frame_id": "unrequested_extra_frame"})
        runner = FakeSelectionRunner(self, alter_result=alter)
        result = self.start(runner)
        self.assertEqual(result["counts"]["failed"], 3)
        self.assertEqual(result["counts"]["completed"], 2)
        self.assertEqual([call["case_id"] for call in runner.calls], self.case_order)
        self.assertIn("entire released video", result["jobs"][0]["error"])
        self.assertIn("additional-frame requests", result["jobs"][1]["error"])
        self.assertIn("sampled candidate count", result["jobs"][2]["error"])

    def test_per_case_config_tamper_is_not_overwritten_and_other_cases_continue(self):
        self.prepare()
        directory = self.output / "runs" / "01-S2A1"
        directory.mkdir(parents=True)
        stale = {"config": asdict(replace(self.config, image_max_tokens=512))}
        _write(directory / "run.json", stale)
        before = (directory / "run.json").read_bytes()
        runner = FakeSelectionRunner(self)
        result = self.resume(runner)
        self.assertEqual(result["counts"]["failed"], 1)
        self.assertEqual(result["counts"]["completed"], 4)
        self.assertEqual((directory / "run.json").read_bytes(), before)
        self.assertEqual([call["case_id"] for call in runner.calls], self.case_order[1:])

    def test_changed_prompt_policy_does_not_mix_review_versions_in_one_batch(self):
        self.prepare()
        runner = FakeSelectionRunner(self)
        for constant in ("REVIEW_POLICY_SHA256", "TIMING_POLICY_SHA256"):
            with self.subTest(policy=constant):
                with patch.object(selection_batch, constant, "0" * 64):
                    with self.assertRaisesRegex(ContractError, "prompt policy changed"):
                        self.resume(runner)
                self.assertEqual(runner.calls, [])

    def test_returned_selection_must_match_saved_artifact_before_certifying_case(self):
        def change_saved_artifact(case_id, directory):
            if case_id == self.case_order[0]:
                saved = read(directory / "selection.json")
                saved["scene_summary"] = "Unexpected concurrent change after runner return value was built"
                _write(directory / "selection.json", saved)
        runner = FakeSelectionRunner(self, after_case=change_saved_artifact)
        result = self.start(runner)
        self.assertEqual(result["counts"]["failed"], 1)
        self.assertEqual(result["counts"]["completed"], 4)
        self.assertIn("differs from its saved artifact", result["jobs"][0]["error"])
        self.assertNotIn("selection_sha256", result["jobs"][0])

    def test_source_inventory_is_rechecked_between_cases_after_initial_queue_verification(self):
        def change_next_source(case_id, directory):
            if case_id == self.case_order[0]:
                image = self.dataset / "frames" / "S1A1" / "S1A1_frame_00000001.png"
                image.write_bytes(image.read_bytes() + b"unexpected source change")
        runner = FakeSelectionRunner(self, after_case=change_next_source)
        result = self.start(runner)
        self.assertEqual(result["counts"]["failed"], 1)
        self.assertEqual(result["counts"]["completed"], 4)
        self.assertIn("Source inventory changed", result["jobs"][1]["error"])
        self.assertEqual([call["case_id"] for call in runner.calls],
                         [case for case in self.case_order if case != "S1A1"])


if __name__ == "__main__":
    unittest.main()

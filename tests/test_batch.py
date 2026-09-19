"""Crash/pause/resume checks use tiny fabricated sequences and scripted models."""
import contextlib
import io
import json
import signal
import unittest
from dataclasses import replace
from unittest.mock import patch

import test_contract
from test_enhancement import ScriptedClient
from yasargil.__main__ import cooperative_stop, main
from yasargil.batch import BatchConfig, enhance_batch, enhancement_status, plan_batch, request_pause
from yasargil.checkpoint import atomic_json, directory_lock
from yasargil.contract import ContractError, validate_record
from yasargil.enhancement import enhance_sospine


class BatchTests(unittest.TestCase):
    def setUp(self):
        fixture = test_contract.ContractTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root, self.base = fixture.root, fixture.base
        self.config = BatchConfig(("S1A2",), window_size=2, initial_frames=2,
                                  search_frames=1, max_frames=3, max_rounds=0)
        self.dest = self.base / "batch"

    def run_batch(self, client, **kwargs):
        return enhance_batch(self.root, self.dest, self.config, client=client, **kwargs)

    def test_full_sequence_includes_short_tail_and_reuses_completed_windows(self):
        client = ScriptedClient(search=False)
        plan = plan_batch(self.root, self.config)
        self.assertEqual(plan["total_frames"], 3)
        self.assertEqual([(w["start_index"], w["cutoff_index"]) for w in plan["windows"]], [(1, 2), (3, 3)])
        result = self.run_batch(client)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(client.requests), 6)
        second = ScriptedClient(search=False)
        resumed = self.run_batch(second, resume=True)
        self.assertEqual(resumed["completed_windows"], 2)
        self.assertEqual(second.requests, [])
        for path in (self.dest / "windows").iterdir():
            record = json.loads((path / "archive.json").read_text())
            validate_record(record, dataset_root=self.root, artifact_root=path)
            self.assertEqual(record["reviews"], [])
        status = enhancement_status(self.dest)
        self.assertEqual(status["completed_windows"], 2)
        self.assertFalse(status["writer_active"])

    def test_pause_during_first_window_preserves_successful_call_and_resume_finishes(self):
        first = ScriptedClient(search=False)
        result = self.run_batch(first, pause_requested=lambda: len(first.requests) >= 1)
        self.assertEqual(result["status"], "paused")
        self.assertEqual(result["completed_windows"], 0)
        second = ScriptedClient(search=False)
        result = self.run_batch(second, resume=True)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(second.requests), 5)
        self.assertEqual(json.loads(second.requests[0]["messages"][1]["content"])["stage"], "independent_observe")

    def test_external_pause_marker_and_saved_settings_cli_resume(self):
        client = ScriptedClient(search=False)
        original_chat = client.chat_raw

        def chat(request):
            raw = original_chat(request)
            request_pause(self.dest)
            return raw

        client.chat_raw = chat
        self.assertEqual(self.run_batch(client)["status"], "paused")
        self.assertTrue(enhancement_status(self.dest)["pause_requested"])
        second = ScriptedClient(search=False)
        with patch("yasargil.__main__.OllamaClient", return_value=second), contextlib.redirect_stdout(io.StringIO()):
            main(["resume-enhancement", "--output-dir", str(self.dest)])
        self.assertEqual(len(second.requests), 5)
        self.assertFalse((self.dest / "PAUSE").exists())
        self.assertEqual(enhancement_status(self.dest)["status"], "completed")

    def test_failed_window_retry_retains_prior_calls_and_failure(self):
        first = ScriptedClient(search=False, fail_stage="review")
        with self.assertRaises(ContractError):
            self.run_batch(first)
        self.assertEqual(enhancement_status(self.dest)["status"], "failed")
        window = self.dest / "windows/S1A2-00000001-00000002"
        failure = (window / "calls/run-003-review/response.json").read_bytes()
        second = ScriptedClient(search=False)
        self.run_batch(second, resume=True)
        self.assertEqual(len(second.requests), 4)
        self.assertEqual((window / "calls/run-003-review/response.json").read_bytes(), failure)

    def test_source_model_and_configuration_changes_fail_before_inference(self):
        self.run_batch(ScriptedClient(search=False), pause_requested=lambda: True)
        changed = ScriptedClient(search=False)
        old_info = changed.model_info

        def model_info(name):
            result = old_info(name)
            result["runtime_version"] = "changed-runtime"
            return result

        changed.model_info = model_info
        with self.assertRaisesRegex(ContractError, "runtime changed"):
            self.run_batch(changed, resume=True)
        self.assertFalse(changed.requests)
        with self.assertRaisesRegex(ContractError, "settings"):
            enhance_batch(self.root, self.dest, replace(self.config, seed=43), resume=True,
                          client=ScriptedClient(search=False))
        (self.root / "documentation/readme.txt").write_text("Changed source")
        with self.assertRaisesRegex(ContractError, "source bytes changed"):
            self.run_batch(ScriptedClient(search=False), resume=True)

    def test_new_or_removed_frames_in_unprocessed_window_reject_resume(self):
        self.run_batch(ScriptedClient(search=False), pause_requested=lambda: True)
        (self.root / "frames/S1A2/S1A2_frame_00000003.jpeg").unlink()
        with self.assertRaisesRegex(ContractError, "inventory changed"):
            self.run_batch(ScriptedClient(search=False), resume=True)

    def test_future_source_changes_during_run_fail_before_next_window(self):
        client = ScriptedClient(search=False)
        original = client.chat_raw

        def chat(request):
            raw = original(request)
            if len(client.requests) == 3:
                (self.root / "frames/S1A2/S1A2_frame_00000003.jpeg").write_bytes(b"changed future evidence")
            return raw

        client.chat_raw = chat
        with self.assertRaisesRegex(ContractError, "source bytes changed before window"):
            self.run_batch(client)
        self.assertEqual(len(client.requests), 3)
        self.assertEqual(enhancement_status(self.dest)["completed_windows"], 1)

    def test_prompt_change_between_windows_rejects_resume(self):
        self.run_batch(ScriptedClient(search=False), pause_requested=lambda: True)
        with patch("yasargil.batch.enhancement_protocol_fingerprint", return_value={"changed": True}):
            with self.assertRaisesRegex(ContractError, "protocol changed"):
                self.run_batch(ScriptedClient(search=False), resume=True)

    def test_single_writer_and_status_without_source_or_models(self):
        self.run_batch(ScriptedClient(search=False), pause_requested=lambda: True)
        with directory_lock(self.dest):
            self.assertTrue(enhancement_status(self.dest)["writer_active"])
            with self.assertRaises(ContractError):
                self.run_batch(ScriptedClient(search=False), resume=True)
        self.root.rename(self.base / "unmounted")
        self.assertEqual(enhancement_status(self.dest)["status"], "paused")

    def test_plan_rejects_gaps_and_duplicates_and_dry_run_has_no_writes(self):
        with self.assertRaises(ContractError):
            plan_batch(self.root, replace(self.config, case_ids=("S1A2", "S1A2")))
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            main(["enhance-sospine-batch", "--dataset-root", str(self.root), "--case-ids", "S1A2",
                  "--window-size", "2", "--dry-run", "--output-dir", str(self.dest)])
        self.assertEqual(json.loads(stdout.getvalue())["total_windows"], 2)
        self.assertFalse(self.dest.exists())
        (self.root / "frames/S1A2/S1A2_frame_00000002.jpeg").unlink()
        with self.assertRaisesRegex(ContractError, "contiguous"):
            plan_batch(self.root, self.config)

    def test_first_signal_pauses_and_second_interrupts_restoring_handlers(self):
        previous = signal.getsignal(signal.SIGINT)
        with contextlib.redirect_stderr(io.StringIO()), cooperative_stop() as paused:
            handler = signal.getsignal(signal.SIGINT)
            self.assertFalse(paused())
            handler(signal.SIGINT, None)
            self.assertTrue(paused())
            with self.assertRaises(KeyboardInterrupt):
                handler(signal.SIGINT, None)
        self.assertEqual(signal.getsignal(signal.SIGINT), previous)

    def test_single_window_saved_settings_resume_and_stale_status(self):
        config = self.config.window_config("S1A2", 1, 3)
        first = ScriptedClient(search=False)
        result = enhance_sospine(self.root, self.dest, config, client=first,
                                pause_requested=lambda: len(first.requests) >= 1)
        self.assertEqual(result["status"], "paused")
        checkpoint = json.loads((self.dest / "checkpoint.json").read_text())
        for status in ("initializing", "running", "finalizing"):
            atomic_json(self.dest / "checkpoint.json", {**checkpoint, "status": status}, overwrite=True)
            self.assertEqual(enhancement_status(self.dest)["status"], "interrupted")
        second = ScriptedClient(search=False)
        with patch("yasargil.__main__.OllamaClient", return_value=second), contextlib.redirect_stdout(io.StringIO()):
            main(["resume-enhancement", "--output-dir", str(self.dest)])
        self.assertEqual(len(second.requests), 2)
        self.assertEqual(enhancement_status(self.dest)["status"], "completed")


if __name__ == "__main__":
    unittest.main()

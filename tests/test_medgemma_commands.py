"""The annotation CLI cannot route MedGemma back through Qwen draft review."""
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import signal
import unittest
from unittest.mock import patch

from yasargil.__main__ import cooperative_stop, main


class MedGemmaCommandTests(unittest.TestCase):
    def test_first_signal_pauses_and_second_interrupts_restoring_handlers(self):
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        with redirect_stderr(io.StringIO()), cooperative_stop() as paused:
            handler = signal.getsignal(signal.SIGINT)
            self.assertFalse(paused())
            handler(signal.SIGINT, None)
            self.assertTrue(paused())
            with self.assertRaises(KeyboardInterrupt):
                handler(signal.SIGTERM, None)
        for sig, original in previous.items():
            self.assertEqual(signal.getsignal(sig), original)

    def test_help_exposes_independent_annotation_and_removes_old_runners(self):
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as raised:
            main(["--help"])
        self.assertEqual(raised.exception.code, 0)
        text = output.getvalue()
        for command in ("annotate-selected-frames", "medgemma-annotation-status", "pause-medgemma-annotation"):
            self.assertIn(command, text)
        for command in ("review-frame-annotations", "frame-review-status", "pause-frame-review",
                        "enhance-sospine", "enhance-sospine-batch", "resume-enhancement",
                        "enhancement-status", "pause-enhancement", "annotate-video-frames"):
            self.assertNotIn(command, text)
            with self.subTest(command=command), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as rejected:
                main([command])
            self.assertEqual(rejected.exception.code, 2)

    def test_prepare_dispatches_selection_directly_without_model_inference(self):
        with patch("yasargil.medgemma_annotation.run_annotation", return_value={"status": "prepared"}) as run, \
                patch("yasargil.medgemma_annotation.LlamaCppClient") as client, redirect_stdout(io.StringIO()):
            main(["annotate-selected-frames", "--selection-run", "/fixture/selection",
                  "--output-dir", "/fixture/annotation", "--prepare-only"])
        args, options = run.call_args
        self.assertEqual(args[:2], (Path("/fixture/selection"), Path("/fixture/annotation")))
        self.assertEqual((args[2].num_ctx, args[2].num_predict), (32768, 4096))
        self.assertEqual((args[2].before_frames, args[2].after_frames, args[2].detail_crops), (2, 2, True))
        self.assertTrue(options["prepare_only"])
        self.assertFalse(options["resume"])
        client.return_value.chat_raw.assert_not_called()

    def test_annotation_rejects_old_draft_and_source_label_inputs(self):
        for option in ("--annotation-run", "--dataset-root", "--allow-rejected-temporal-citations"):
            with self.subTest(option=option), patch("yasargil.medgemma_annotation.run_annotation") as run, \
                    redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as rejected:
                main(["annotate-selected-frames", "--output-dir", "/fixture/annotation", option, "/fixture/old"])
            self.assertEqual(rejected.exception.code, 2)
            run.assert_not_called()

    def test_target_only_baseline_can_disable_neighbors_and_crops(self):
        with patch("yasargil.medgemma_annotation.run_annotation", return_value={"status": "prepared"}) as run, \
                patch("yasargil.medgemma_annotation.LlamaCppClient"), redirect_stdout(io.StringIO()):
            main(["annotate-selected-frames", "--selection-run", "/fixture/selection",
                  "--output-dir", "/fixture/annotation", "--prepare-only",
                  "--before-frames", "0", "--after-frames", "0", "--no-detail-crops"])
        config = run.call_args.args[2]
        self.assertEqual((config.before_frames, config.after_frames, config.detail_crops), (0, 0, False))


if __name__ == "__main__":
    unittest.main()

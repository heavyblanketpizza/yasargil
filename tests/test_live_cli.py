"""The ``python -m yasargil.live`` command line, with the model runtime replaced by a script."""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from live_fixtures import make_case, scripted_timeline
from test_live_session import answer_with_driver
from yasargil.live import __main__ as cli
from yasargil.live.agent import AgentConfig, GuidanceAgent
from yasargil.live.transport import ScriptedTransport

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_repo_hygiene as hygiene  # noqa: E402


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli.main([str(a) for a in argv])
    return code, out.getvalue(), err.getvalue()


def scripted_runtime(responses):
    @contextmanager
    def runtime(args, log_dir):
        log_dir.mkdir(parents=True, exist_ok=True)
        yield GuidanceAgent(ScriptedTransport(list(responses)), AgentConfig()), None
    return runtime


class CliTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        make_case(self.root, "S1A1", scripted_timeline())

    def test_replay_without_agent_writes_run_and_report(self):
        code, out, err = run("replay", "--dataset-root", self.root, "--case", "S1A1", "--output-dir",
                             self.root / "run", "--speed", "0")
        self.assertEqual(code, 0, err)
        self.assertTrue((self.root / "run" / "report.html").is_file())
        summary = json.loads(out)
        self.assertEqual(summary["frames"], 12)

    def test_replay_with_degraded_perception_and_window(self):
        code, out, err = run("replay", "--dataset-root", self.root, "--case", "S1A1", "--output-dir",
                             self.root / "run", "--speed", "0", "--miss-rate", "0.3", "--seed", "4",
                             "--start-frame", "2", "--end-frame", "9", "--no-report")
        self.assertEqual(code, 0, err)
        manifest = json.loads((self.root / "run" / "run.json").read_text())
        self.assertEqual(manifest["perception"]["kind"], "degraded")
        self.assertEqual(manifest["perception"]["miss_rate"], 0.3)
        self.assertEqual(json.loads(out)["frames"], 8)
        self.assertFalse((self.root / "run" / "report.html").exists())

    def test_replay_with_question_file_and_agent(self):
        questions = self.root / "questions.jsonl"
        questions.write_text(json.dumps({"at_s": 6, "text": "Which instruments are in view?"}) + "\n")
        with patch.object(cli, "agent_runtime", scripted_runtime([answer_with_driver])):
            code, out, err = run("replay", "--dataset-root", self.root, "--case", "S1A1", "--output-dir",
                                 self.root / "run", "--speed", "0", "--questions", questions, "--agent", "--blocking")
        self.assertEqual(code, 0, err)
        answers = [json.loads(l) for l in (self.root / "run" / "answers.jsonl").read_text().splitlines()]
        self.assertEqual(answers[0]["question_id"], "Q001")
        self.assertEqual(answers[0]["spoken_text"], "The needle driver is in view.")
        self.assertTrue((self.root / "run" / "runtime").is_dir())

    def test_auto_questions_without_agent_are_recorded_unanswered(self):
        code, out, err = run("replay", "--dataset-root", self.root, "--case", "S1A1", "--output-dir",
                             self.root / "run", "--speed", "0", "--auto-questions", "3", "--no-report")
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)["questions"]["by_status"], {"agent_unavailable": 3})

    def test_eval_aggregates_cases(self):
        make_case(self.root, "S2A1", scripted_timeline(), append=True)
        code, out, err = run("eval", "--dataset-root", self.root, "--cases", "S1A1", "S2A1", "--output-dir",
                             self.root / "eval", "--questions-per-case", "6", "--seed", "1")
        self.assertEqual(code, 0, err)
        summary = json.loads((self.root / "eval" / "summary.json").read_text())
        self.assertEqual(summary["cases"], ["S1A1", "S2A1"])
        self.assertEqual(summary["overall"]["symbolic"]["graded"], 12)
        self.assertEqual(summary["overall"]["symbolic"]["accuracy"], 1.0)
        self.assertTrue((self.root / "eval" / "S2A1" / "run" / "report.html").is_file())
        self.assertEqual(json.loads(out)["overall"]["symbolic"]["accuracy"], 1.0)

    def test_eval_with_agent(self):
        with patch.object(cli, "agent_runtime", scripted_runtime([answer_with_driver] * 4)):
            code, out, err = run("eval", "--dataset-root", self.root, "--cases", "S1A1", "--output-dir",
                                 self.root / "eval", "--questions-per-case", "4", "--agent", "--no-report")
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)["overall"]["agent"]["graded"], 4)

    def test_ask_against_a_finished_run(self):
        run("replay", "--dataset-root", self.root, "--case", "S1A1", "--output-dir", self.root / "run",
            "--speed", "0", "--no-report")
        with patch.object(cli, "agent_runtime", scripted_runtime([answer_with_driver])):
            code, out, err = run("ask", "--run", self.root / "run", "--dataset-root", self.root, "--at-s", "6",
                                 "Which instruments are in view?")
        self.assertEqual(code, 0, err)
        result = json.loads(out)
        self.assertEqual(result["spoken_text"], "The needle driver is in view.")
        self.assertEqual(result["frame_index"], 7)
        self.assertEqual(len(list((self.root / "run" / "ask").glob("*/result.json"))), 1)

    def test_report_command_regenerates(self):
        run("replay", "--dataset-root", self.root, "--case", "S1A1", "--output-dir", self.root / "run",
            "--speed", "0", "--no-report")
        code, out, err = run("report", "--run", self.root / "run")
        self.assertEqual(code, 0, err)
        self.assertTrue((self.root / "run" / "report.html").is_file())

    def test_user_errors_exit_with_a_message(self):
        code, out, err = run("replay", "--dataset-root", self.root / "missing", "--case", "S1A1",
                             "--output-dir", self.root / "run", "--speed", "0")
        self.assertEqual(code, 2)
        self.assertIn("error:", err)
        code, out, err = run("replay", "--dataset-root", self.root, "--case", "S1A1", "--output-dir",
                             self.root / "run", "--perception", "probe")
        self.assertEqual(code, 2)
        self.assertIn("--probe-dir", err)

    def test_module_entry_point(self):
        result = subprocess.run([sys.executable, "-B", "-m", "yasargil.live", "--help"], capture_output=True,
                                text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0)
        for command in ("replay", "eval", "ask", "train-probe", "report"):
            self.assertIn(command, result.stdout)


class DocumentationTests(unittest.TestCase):
    def test_live_guide_is_public_and_clean(self):
        guide = "docs/LIVE_GUIDANCE.md"
        self.assertIn(guide, hygiene.PUBLIC_MARKDOWN)
        self.assertEqual(list(hygiene.findings(guide, (ROOT / guide).read_bytes())), [])
        ignored = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "--no-index", "-q", guide])
        self.assertEqual(ignored.returncode, 1)


if __name__ == "__main__":
    unittest.main()

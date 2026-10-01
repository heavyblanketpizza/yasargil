"""The self-contained run viewer."""
import json
from pathlib import Path
import re
import tempfile
import unittest

from live_fixtures import make_case, scripted_timeline
from test_live_session import answer_with_driver
from yasargil.live import LiveError
from yasargil.live.agent import AgentConfig, GuidanceAgent
from yasargil.live.frames import case_frames
from yasargil.live.labels import CaseLabels
from yasargil.live.perception import LabelPerception
from yasargil.live.report import write_report
from yasargil.live.session import ScheduledQuestion, Session
from yasargil.live.transport import ScriptedTransport


def embedded(html):
    match = re.search(r'<script id="run-data" type="application/json">(.*?)</script>', html, re.S)
    return json.loads(match.group(1))


class ReportTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        make_case(self.root, "S1A1", scripted_timeline(), size=(96, 54))
        self.frames = case_frames(self.root, "S1A1")
        self.perception = LabelPerception(CaseLabels.load(self.root, "S1A1", (96, 54)))

    def test_report_embeds_run_data_and_thumbnails(self):
        agent = GuidanceAgent(ScriptedTransport([answer_with_driver]), AgentConfig())
        hostile = "What is in view?</script><script>alert(1)</script>"
        Session(self.frames, self.perception, self.root / "run", agent=agent,
                questions=[ScheduledQuestion("Q1", 6000, hostile)]).run()
        path = write_report(self.root / "run", thumbnail_side=48)
        html = path.read_text()
        self.assertEqual(path, self.root / "run" / "report.html")
        self.assertEqual(html.count("</script>"), 2)
        data = embedded(html)
        self.assertEqual(data["run"]["case_id"], "S1A1")
        self.assertEqual(len(data["frames"]), 12)
        self.assertTrue(any(e["type"] == "tip_near_structure" for e in data["events"]))
        self.assertTrue(any(u["text"].startswith("Needle left") for u in data["utterances"]))
        self.assertEqual(data["answers"][0]["text"], hostile)
        self.assertEqual(data["agent"]["Q1"]["status"], "answered")
        thumbnail = self.root / "run" / data["frames"][0]["thumbnail"]
        self.assertTrue(thumbnail.is_file())
        from PIL import Image
        with Image.open(thumbnail) as image:
            self.assertEqual(max(image.size), 48)
        self.assertIn("prefers-color-scheme: dark", html)

    def test_report_without_agent_files(self):
        Session(self.frames, self.perception, self.root / "run").run()
        data = embedded(write_report(self.root / "run").read_text())
        self.assertEqual(data["answers"], [])
        self.assertEqual(data["agent"], {})

    def test_missing_run_is_an_error(self):
        with self.assertRaises(LiveError):
            write_report(self.root / "nope")


if __name__ == "__main__":
    unittest.main()

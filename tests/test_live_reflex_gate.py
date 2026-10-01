"""Reflex rules and the speak gate."""
import json
from pathlib import Path
import tempfile
import unittest

from yasargil.live import LiveError
from yasargil.live.events import EventLog
from yasargil.live.gate import Candidate, GateConfig, SpeakGate
from yasargil.live.procedure import DEFAULT_RULES
from yasargil.live.reflex import ReflexEngine, load_rules


class ReflexTests(unittest.TestCase):
    def setUp(self):
        self.log = EventLog()
        self.engine = ReflexEngine(DEFAULT_RULES, self.log)
        self.driver = self.log.append(0, 1, "instrument_entered", "needle driver")
        self.needle = self.log.append(0, 1, "instrument_entered", "needle")

    def test_near_structure_rule_fires_and_records_an_alert_event(self):
        near = self.log.append(1000, 2, "tip_near_structure", "needle driver", object="durotomy", confidence=0.8)
        candidates = self.engine.on_events([near], self.log.state_at(1000), 1000)
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate.text, "Needle driver tip is close to the durotomy.")
        self.assertEqual(candidate.priority, "high")
        self.assertEqual(candidate.key, "rule:tip_near_durotomy:needle driver")
        alert = self.log.all()[-1]
        self.assertEqual((alert.type, alert.subject, alert.cites), ("alert", "tip_near_durotomy", (near.event_id,)))
        self.assertEqual(candidate.cites, (alert.event_id, near.event_id))
        self.assertEqual(candidate.confidence, 0.8)
        self.assertEqual(candidate.cooldown_ms, 15000)

    def test_required_state_gates_the_rule(self):
        left = self.log.append(2000, 3, "instrument_left", "needle")
        fired = self.engine.on_events([left], self.log.state_at(2000), 2000)
        self.assertEqual([c.priority for c in fired], ["critical"])
        self.assertIn(self.driver.event_id, fired[0].cites)
        self.log.append(3000, 4, "instrument_left", "needle driver")
        self.log.append(3000, 4, "instrument_entered", "needle")
        left_again = self.log.append(4000, 5, "instrument_left", "needle")
        self.assertEqual(self.engine.on_events([left_again], self.log.state_at(4000), 4000), [])

    def test_message_fields_from_event_data(self):
        changed = self.log.append(1000, 2, "step_changed", "suturing")
        unexpected = self.log.append(1000, 2, "step_unexpected", "suturing", cites=(changed.event_id,),
                                     data={"reason": "suturing began before exposure was observed"})
        fired = self.engine.on_events([unexpected], self.log.state_at(1000), 1000)
        self.assertEqual(fired[0].text, "Unexpected step: suturing began before exposure was observed.")

    def test_invalid_rules_are_rejected_and_files_load(self):
        with self.assertRaises(LiveError):
            ReflexEngine([{"id": "x", "when": {}, "priority": "high", "message": "m"}])
        with self.assertRaises(LiveError):
            ReflexEngine([{"id": "x", "when": {"type": "alert"}, "priority": "urgent", "message": "m"}])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rules.json"
            path.write_text(json.dumps(DEFAULT_RULES))
            self.assertEqual(len(load_rules(path)), len(DEFAULT_RULES))


def alert(key="k", priority="high", t_ms=0, confidence=1.0, cooldown_ms=None):
    return Candidate("reflex", priority, key, f"text {key}", ("E000001",), confidence, t_ms, None, cooldown_ms)


class GateTests(unittest.TestCase):
    def test_cooldown_suppresses_repeats_of_the_same_key(self):
        gate = SpeakGate(GateConfig(default_cooldown_ms=10000))
        self.assertTrue(gate.offer(alert(), 0).spoken)
        self.assertEqual(gate.offer(alert(t_ms=5000), 5000).reason, "cooldown")
        self.assertTrue(gate.offer(alert(t_ms=10000), 10000).spoken)
        self.assertTrue(gate.offer(alert("other", t_ms=5000), 5000).spoken)

    def test_candidate_cooldown_overrides_default(self):
        gate = SpeakGate(GateConfig(default_cooldown_ms=10000))
        gate.offer(alert(cooldown_ms=1000), 0)
        self.assertTrue(gate.offer(alert(t_ms=1500, cooldown_ms=1000), 1500).spoken)

    def test_rate_limit_with_critical_bypass(self):
        gate = SpeakGate(GateConfig(max_per_minute=2))
        self.assertTrue(gate.offer(alert("a"), 0).spoken)
        self.assertTrue(gate.offer(alert("b"), 0).spoken)
        self.assertEqual(gate.offer(alert("c"), 0).reason, "rate_limited")
        self.assertTrue(gate.offer(alert("d", "critical"), 0).spoken)
        self.assertTrue(gate.offer(alert("e", t_ms=60001), 60001).spoken)

    def test_stale_and_low_confidence_candidates_are_suppressed(self):
        gate = SpeakGate(GateConfig(max_age_ms=3000, min_confidence=0.6))
        self.assertEqual(gate.offer(alert("a", t_ms=0), 4000).reason, "stale")
        self.assertEqual(gate.offer(alert("b", t_ms=4000, confidence=0.5), 4000).reason, "low_confidence")

    def test_answers_are_always_spoken_and_everything_is_logged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "utterances.jsonl"
            gate = SpeakGate(GateConfig(max_per_minute=1, max_age_ms=1), path)
            gate.offer(alert("a"), 0)
            answer = Candidate("agent", "medium", "answer:Q1", "Two instruments.", ("E000001",), 0.2, 0, "Q1")
            decision = gate.offer(answer, 99999)
            self.assertTrue(decision.spoken)
            self.assertEqual(decision.reason, "answer")
            gate.offer(alert("b"), 0)
            lines = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([(l["key"], l["spoken"], l["reason"]) for l in lines],
                             [("a", True, "spoken"), ("answer:Q1", True, "answer"), ("b", False, "rate_limited")])
            self.assertEqual(lines[1]["question_id"], "Q1")
            self.assertEqual(len(gate.records), 3)


if __name__ == "__main__":
    unittest.main()

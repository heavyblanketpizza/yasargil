"""Mechanical claim verification: what may be spoken.

Each claim in a final answer must cite events that exist and were known at the
question time, frames that are not in the future and were actually inspected
(viewed, perceived, or listed as evidence by a cited event), and/or tool results
the agent actually received. A claim naming an instrument or structure must cite
something about it: an event about it, an inspected frame as visual evidence, a
state snapshot, or an event query filtered to it (or unfiltered). Results make
absence citable: "the grasper has not entered" cites the query that found no
entry. Only verified claim text is spoken.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

from .labels import ANATOMY, INSTRUMENTS

NO_ANSWER = "I can't confirm that from what has been observed so far."
_LABELS = sorted(INSTRUMENTS + ANATOMY, key=len, reverse=True)


@dataclass(frozen=True)
class Verdict:
    spoken_text: str
    verified: list
    rejected: list

    @property
    def supported(self):
        return bool(self.verified) and not self.rejected

    def to_json(self):
        return {"spoken_text": self.spoken_text, "supported": self.supported,
                "verified": self.verified, "rejected": self.rejected}


def mentioned_labels(text):
    """Instrument and structure names in a claim; 'needle driver' does not also count as 'needle'."""
    remaining = text.lower()
    found = set()
    for label in _LABELS:
        pattern = re.compile(r"\b" + re.escape(label) + r"\b")
        if pattern.search(remaining):
            found.add(label)
            remaining = pattern.sub(" ", remaining)
    return found


def check_claims(final, log, now_ms, current_index, viewed_frames, results=None):
    results = results or {}
    verified, rejected = [], []
    for claim in final.get("claims") or []:
        reasons = []
        events = []
        for event_id in claim.get("event_ids") or []:
            event = log.get(event_id)
            if event is None:
                reasons.append(f"unknown event {event_id}")
            elif event.t_ms > now_ms:
                reasons.append(f"future event {event_id}")
            else:
                events.append(event)
        evidence = set(viewed_frames)
        for event in events:
            evidence.update(event.evidence_frames)
        frames = []
        for index in claim.get("frame_indices") or []:
            if index > current_index:
                reasons.append(f"future frame {index}")
            elif index not in evidence:
                reasons.append(f"frame {index} was not inspected")
            else:
                frames.append(index)
        cited_results = []
        for result_id in claim.get("result_ids") or []:
            if result_id in results:
                cited_results.append(results[result_id])
            else:
                reasons.append(f"unknown result {result_id}")
        if not claim.get("event_ids") and not claim.get("frame_indices") and not claim.get("result_ids"):
            reasons.append("no citation")
        covers_all = any(r.get("tool") == "current_state" or (r.get("tool") == "query_events" and not r.get("subject"))
                         for r in cited_results)
        if not frames and not covers_all:
            about = {name for event in events for name in (event.subject, event.object) if name}
            about.update(r["subject"] for r in cited_results if r.get("tool") == "query_events" and r.get("subject"))
            for label in sorted(mentioned_labels(claim.get("text", ""))):
                if label not in about:
                    reasons.append(f"cites no event about {label}")
        (rejected if reasons else verified).append({**claim, "reasons": reasons} if reasons else dict(claim))
    if verified:
        spoken = " ".join(claim["text"].strip() for claim in verified)
        if rejected:
            spoken += " Other parts could not be verified."
    else:
        spoken = NO_ANSWER
        if final.get("unresolved"):
            spoken += " " + final["unresolved"].strip()
    return Verdict(spoken, verified, rejected)

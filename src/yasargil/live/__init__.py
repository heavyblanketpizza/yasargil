"""Replayable real-time guidance research harness.

Frames are perceived at the replay clock, perception becomes an append-only event
log, and three loops read that log at different speeds: reflex rules, a procedure
tracker and a tool-calling Qwen agent. A speak gate decides what reaches the
surgeon. Research replay only; not a medical device.
"""
from __future__ import annotations


class LiveError(RuntimeError):
    """A user-facing configuration, input or runtime failure in the live harness."""

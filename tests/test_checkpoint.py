"""Failure, publication, and contention boundaries for local checkpoints."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from yasargil.checkpoint import (LOCK_FILENAME, atomic_bytes, atomic_json,
                                directory_is_locked, directory_lock, durable_mkdir)
from yasargil.contract import ContractError


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.target = self.directory / "state.json"

    def assert_no_temporaries(self):
        self.assertEqual(list(self.directory.glob(".checkpoint-*.tmp")), [])

    def test_json_round_trip_keeps_unicode(self):
        value = {"state": "검토 필요 — naïve", "steps": [1, None, True]}
        self.assertEqual(atomic_json(self.target, value), self.target)
        self.assertEqual(json.loads(self.target.read_text(encoding="utf-8")), value)
        self.assertIn("검토".encode(), self.target.read_bytes())
        self.assertTrue(self.target.read_bytes().endswith(b"\n"))
        self.assert_no_temporaries()

    def test_durable_mkdir_flushes_nested_directory_entries(self):
        target = self.directory / "calls" / "run-001" / "attempts" / "0002"
        flushed = []
        real_fsync = os.fsync

        def observe(descriptor):
            info = os.fstat(descriptor)
            flushed.append((info.st_dev, info.st_ino))
            real_fsync(descriptor)

        with patch("yasargil.checkpoint.os.fsync", side_effect=observe):
            self.assertEqual(durable_mkdir(target), target)
        for directory in (self.directory, self.directory / "calls", self.directory / "calls/run-001",
                          self.directory / "calls/run-001/attempts", target):
            info = directory.stat()
            self.assertIn((info.st_dev, info.st_ino), flushed)

    def test_durable_mkdir_flushes_existing_directory_and_parent(self):
        target = self.directory / "existing"
        target.mkdir()
        sentinel = target / "user-file"
        sentinel.write_bytes(b"preserve")
        flushed = []
        real_fsync = os.fsync

        def observe(descriptor):
            info = os.fstat(descriptor)
            flushed.append((info.st_dev, info.st_ino))
            real_fsync(descriptor)

        with patch("yasargil.checkpoint.os.fsync", side_effect=observe):
            durable_mkdir(target)
        for directory in (self.directory, target):
            info = directory.stat()
            self.assertIn((info.st_dev, info.st_ino), flushed)
        self.assertEqual(sentinel.read_bytes(), b"preserve")

    def test_durable_mkdir_accepts_ancestor_alias_but_rejects_target_alias(self):
        parent = self.directory / "real"
        parent.mkdir()
        alias = self.directory / "alias"
        alias.symlink_to(parent, target_is_directory=True)
        durable_mkdir(alias / "child")
        self.assertTrue((parent / "child").is_dir())
        with self.assertRaisesRegex(ContractError, "symlink"):
            durable_mkdir(alias)
        self.target.write_bytes(b"user file")
        with self.assertRaises(ContractError):
            durable_mkdir(self.target)
        self.assertEqual(self.target.read_bytes(), b"user file")

    def test_default_never_overwrites_even_identical_bytes(self):
        self.target.write_bytes(b"user-owned bytes")
        for data in (b"new checkpoint", b"user-owned bytes"):
            with self.subTest(data=data), self.assertRaisesRegex(ContractError, "already exists"):
                atomic_bytes(self.target, data)
        self.assertEqual(self.target.read_bytes(), b"user-owned bytes")
        self.assert_no_temporaries()

    def test_racing_creation_cannot_be_overwritten(self):
        real_link = os.link

        def competing_creation(*args, **kwargs):
            self.target.write_bytes(b"racing writer")
            return real_link(*args, **kwargs)

        with patch("yasargil.checkpoint.os.link", side_effect=competing_creation):
            with self.assertRaisesRegex(ContractError, "already exists"):
                atomic_bytes(self.target, b"our checkpoint")
        self.assertEqual(self.target.read_bytes(), b"racing writer")
        self.assert_no_temporaries()

    def test_explicit_overwrite_replaces_complete_checkpoint(self):
        self.target.write_bytes(b"old")
        atomic_bytes(self.target, b"complete new state", overwrite=True)
        self.assertEqual(self.target.read_bytes(), b"complete new state")
        self.assert_no_temporaries()

    def test_failed_file_flush_preserves_previous_checkpoint(self):
        self.target.write_bytes(b"old checkpoint")
        with patch("yasargil.checkpoint.os.fsync", side_effect=OSError("storage failure")):
            with self.assertRaisesRegex(OSError, "storage failure"):
                atomic_bytes(self.target, b"new checkpoint", overwrite=True)
        self.assertEqual(self.target.read_bytes(), b"old checkpoint")
        self.assert_no_temporaries()

    def test_failed_publication_leaves_no_partial_target(self):
        with patch("yasargil.checkpoint.os.link", side_effect=OSError("publication failure")):
            with self.assertRaisesRegex(OSError, "publication failure"):
                atomic_bytes(self.target, b"some new bytes")
        self.assertFalse(self.target.exists())
        self.assert_no_temporaries()

    def test_failed_replacement_preserves_old_bytes(self):
        self.target.write_bytes(b"old checkpoint")
        with patch("yasargil.checkpoint.os.replace", side_effect=OSError("replace failure")):
            with self.assertRaisesRegex(OSError, "replace failure"):
                atomic_bytes(self.target, b"some new bytes", overwrite=True)
        self.assertEqual(self.target.read_bytes(), b"old checkpoint")
        self.assert_no_temporaries()

    def test_invalid_json_never_changes_checkpoint(self):
        self.target.write_bytes(b"old checkpoint")
        for value in ({"score": float("nan")}, {"unserializable": object()}):
            with self.subTest(value=value), self.assertRaises(ContractError):
                atomic_json(self.target, value, overwrite=True)
        self.assertEqual(self.target.read_bytes(), b"old checkpoint")
        self.assert_no_temporaries()

    def test_symlink_targets_and_nonfiles_are_rejected(self):
        original = self.directory / "original"
        original.write_bytes(b"user data")
        self.target.symlink_to(original)
        for overwrite in (False, True):
            with self.subTest(overwrite=overwrite), self.assertRaisesRegex(ContractError, "symlink"):
                atomic_bytes(self.target, b"replacement", overwrite=overwrite)
        self.assertEqual(original.read_bytes(), b"user data")
        self.target.unlink()
        self.target.mkdir()
        with self.assertRaises(ContractError):
            atomic_bytes(self.target, b"replacement", overwrite=True)
        self.assertTrue(self.target.is_dir())

    def test_traversal_and_symlink_parent_are_rejected(self):
        child = self.directory / "child"
        child.mkdir()
        with self.assertRaisesRegex(ContractError, "Invalid checkpoint"):
            atomic_bytes(child / ".." / "state.json", b"no")
        alias = self.directory / "alias"
        alias.symlink_to(child, target_is_directory=True)
        with self.assertRaises(ContractError):
            atomic_bytes(alias / "state.json", b"no")
        self.assertFalse((child / "state.json").exists())

    def test_same_process_contention_and_exception_release_keep_inode(self):
        with self.assertRaisesRegex(RuntimeError, "aborted"):
            with directory_lock(self.directory):
                inode = (self.directory / LOCK_FILENAME).stat().st_ino
                with self.assertRaisesRegex(ContractError, "active writer"):
                    with directory_lock(self.directory):
                        self.fail("second writer acquired lock")
                raise RuntimeError("aborted")
        self.assertEqual((self.directory / LOCK_FILENAME).stat().st_ino, inode)
        with directory_lock(self.directory):
            self.assertEqual((self.directory / LOCK_FILENAME).stat().st_ino, inode)

    def test_lock_status_does_not_create_file_and_tracks_writer(self):
        self.assertFalse(directory_is_locked(self.directory))
        self.assertFalse((self.directory / LOCK_FILENAME).exists())
        with directory_lock(self.directory):
            self.assertTrue(directory_is_locked(self.directory))
        self.assertFalse(directory_is_locked(self.directory))
        self.assertTrue((self.directory / LOCK_FILENAME).exists())

    def test_other_process_sees_contention_and_release(self):
        code = """from pathlib import Path
import sys
from yasargil.checkpoint import directory_is_locked, directory_lock
from yasargil.contract import ContractError
directory = Path(sys.argv[1])
print(directory_is_locked(directory), flush=True)
try:
    with directory_lock(directory):
        print('acquired', flush=True)
except ContractError:
    print('contended', flush=True)
"""

        def probe():
            return subprocess.run([sys.executable, "-c", code, str(self.directory)],
                                  capture_output=True, text=True, check=True, timeout=15).stdout.splitlines()

        with directory_lock(self.directory):
            self.assertEqual(probe(), ["True", "contended"])
        self.assertEqual(probe(), ["False", "acquired"])

    def test_lock_rejects_symlink_and_reserved_checkpoint_name(self):
        original = self.directory / "original"
        original.write_bytes(b"keep")
        (self.directory / LOCK_FILENAME).symlink_to(original)
        for action in (lambda: directory_is_locked(self.directory),
                       lambda: atomic_bytes(self.directory / LOCK_FILENAME, b"bad", overwrite=True)):
            with self.assertRaises(ContractError):
                action()
        with self.assertRaises(ContractError):
            with directory_lock(self.directory):
                self.fail("symlink lock accepted")
        self.assertEqual(original.read_bytes(), b"keep")

    def test_fifo_lock_is_rejected_without_waiting_for_a_writer(self):
        os.mkfifo(self.directory / LOCK_FILENAME)
        with self.assertRaisesRegex(ContractError, "regular file"):
            directory_is_locked(self.directory)
        with self.assertRaisesRegex(ContractError, "regular file"):
            with directory_lock(self.directory):
                self.fail("FIFO lock accepted")


if __name__ == "__main__":
    unittest.main()

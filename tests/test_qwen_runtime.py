from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from yasargil.qwen_runtime import (BASE_COMMIT, PATCH_RELATIVE_PATH, RUNTIME_REVISION,
                                   SOURCE_ARCHIVE_SHA256, file_sha256, verified_qwen_runtime)


class QwenRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / ".runtime/llama.cpp" / RUNTIME_REVISION
        (self.directory / "bin").mkdir(parents=True)
        self.binary = self.directory / "bin/llama-server"
        self.binary.write_bytes(b"server")
        self.library = self.directory / "bin/libmtmd.dylib"
        self.library.write_bytes(b"patched video library")
        self.patch = self.root / PATCH_RELATIVE_PATH
        self.patch.parent.mkdir(parents=True)
        self.patch.write_bytes(b"reviewed patch")
        self.receipt = {"revision": RUNTIME_REVISION, "base_commit": BASE_COMMIT,
                        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
                        "patch_sha256": file_sha256(self.patch),
                        "artifacts": {"bin/llama-server": file_sha256(self.binary),
                                      "bin/libmtmd.dylib": file_sha256(self.library)}}
        self.save(self.receipt)

    def save(self, receipt):
        (self.directory / "build-receipt.json").write_text(json.dumps(receipt))

    def test_matching_build_selected(self):
        self.assertEqual(verified_qwen_runtime(self.root), (self.binary, self.receipt))

    def test_changed_library_or_patch_rejected(self):
        for path in (self.binary, self.library, self.patch):
            original = path.read_bytes()
            path.write_bytes(original + b"modified")
            with self.subTest(path=path.name), self.assertRaises(ValueError):
                verified_qwen_runtime(self.root)
            path.write_bytes(original)

    def test_missing_stale_or_escaping_receipts_rejected(self):
        for key, value in (("revision", "b10809"), ("base_commit", "unknown"),
                           ("artifacts", {}), ("artifacts", {"bin/llama-server": "bad", "../other": "bad"})):
            changed = deepcopy(self.receipt)
            changed[key] = value
            self.save(changed)
            with self.subTest(key=key), self.assertRaises(ValueError):
                verified_qwen_runtime(self.root)
        (self.directory / "build-receipt.json").unlink()
        with self.assertRaises(ValueError):
            verified_qwen_runtime(self.root)

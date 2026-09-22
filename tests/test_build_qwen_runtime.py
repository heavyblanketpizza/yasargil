"""The pinned build must reject unrelated edits in a reused source checkout."""
import importlib.util
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest

from yasargil.qwen_runtime import file_sha256


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_qwen_runtime.py"
SPEC = importlib.util.spec_from_file_location("build_qwen_runtime", SCRIPT)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


@unittest.skipUnless(shutil.which("patch"), "the native build requires patch")
class VerifiedSourceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        original = self.root / "original/llama.cpp-pinned"
        original.mkdir(parents=True)
        (original / "changed.cpp").write_text("before\n")
        (original / "unrelated.cpp").write_text("untouched\n")
        self.archive = self.root / "source.tar.gz"
        with tarfile.open(self.archive, "w:gz") as archive:
            archive.add(original, arcname=original.name)
        self.archive_digest = file_sha256(self.archive)
        self.source_dir = self.root / original.name
        self.patch = self.root / "source.patch"
        self.patch.write_text("--- a/changed.cpp\n+++ b/changed.cpp\n@@ -1 +1 @@\n-before\n+after\n")

    def prepare(self):
        return builder.prepare_verified_source(self.archive, self.source_dir, self.patch,
                                               self.archive_digest)

    def test_new_and_matching_reused_sources_are_verified_without_replacement(self):
        expected = self.prepare()
        self.assertEqual((self.source_dir / "changed.cpp").read_text(), "after\n")
        before_stat = (self.source_dir.stat(), (self.source_dir / "changed.cpp").stat())
        self.assertEqual(self.prepare(), expected)
        after_stat = (self.source_dir.stat(), (self.source_dir / "changed.cpp").stat())
        self.assertEqual([(s.st_ino, s.st_mtime_ns) for s in before_stat],
                         [(s.st_ino, s.st_mtime_ns) for s in after_stat])

    def test_unrelated_edit_is_rejected_and_preserved(self):
        self.prepare()
        unrelated = self.source_dir / "unrelated.cpp"
        unrelated.write_text("unreviewed local edit\n")
        with self.assertRaisesRegex(ValueError, "changed: unrelated.cpp"):
            self.prepare()
        self.assertEqual(unrelated.read_text(), "unreviewed local edit\n")
        self.assertEqual((self.source_dir / "changed.cpp").read_text(), "after\n")

    def test_extra_and_missing_files_are_rejected(self):
        self.prepare()
        (self.source_dir / "extra.cpp").write_text("not in the archive\n")
        (self.source_dir / "unrelated.cpp").unlink()
        with self.assertRaisesRegex(ValueError, "missing: unrelated.cpp; extra: extra.cpp"):
            self.prepare()
        self.assertTrue((self.source_dir / "extra.cpp").is_file())

    def test_source_links_and_wrong_archive_are_rejected(self):
        self.prepare()
        unrelated = self.source_dir / "unrelated.cpp"
        unrelated.unlink()
        unrelated.symlink_to(self.root / "original/llama.cpp-pinned/unrelated.cpp")
        with self.assertRaisesRegex(ValueError, "non-regular source entry"):
            self.prepare()
        self.archive.write_bytes(self.archive.read_bytes() + b"changed archive")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()

"""The README artwork exception must not admit source or dataset media."""

import importlib.util
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_repo_hygiene", ROOT / "scripts" / "check_repo_hygiene.py"
)
hygiene = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hygiene)

BANNER = "docs/assets/yasargil-banner.webp"
EXCLUDED_MEDIA = (
    "docs/assets/source.webp",
    "docs/assets/yasargil-banner.png",
    "docs/other/yasargil-banner.webp",
    "nested/docs/assets/yasargil-banner.webp",
    "data/yasargil-banner.webp",
    "outputs/yasargil-banner.webp",
    "source.jpeg",
    "surgery.mp4",
)


class RepositoryHygieneTests(unittest.TestCase):
    def test_only_exact_banner_path_is_allowed_media(self):
        self.assertEqual(list(hygiene.findings(BANNER, b"RIFF\0WEBP")), [])
        for name in EXCLUDED_MEDIA:
            with self.subTest(name=name):
                self.assertIn(
                    (1, "source-media-or-archive"),
                    list(hygiene.findings(name, b"synthetic\0media")),
                )

    def test_banner_exception_keeps_content_checks(self):
        synthetic_path = b"/" + b"Users/" + b"private-user/file"
        self.assertIn(
            (1, "machine-specific-path"),
            list(hygiene.findings(BANNER, synthetic_path)),
        )

    def test_gitignore_keeps_other_media_excluded(self):
        result = subprocess.run(
            ["git", "-C", str(ROOT), "check-ignore", "--no-index", "--stdin"],
            input="\n".join((BANNER, *EXCLUDED_MEDIA)) + "\n",
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(set(result.stdout.splitlines()), set(EXCLUDED_MEDIA))


if __name__ == "__main__":
    unittest.main()

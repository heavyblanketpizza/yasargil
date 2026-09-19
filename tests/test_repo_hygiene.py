"""Public repository guards reject local assets without excluding source files."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
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

LEAK_PATHS = {
    "vendored-font": (
        "src/static/Meslo.ttf", "assets/other.OTF", "assets/other.woff",
        "assets/other.WOFF2", "assets/other.eot",
    ),
    "dataset-or-generated-data": (
        "weights.bin", "snapshots/model.CKPT", "network.onnx",
        "cache.pkl", "cache.pickle", "estimator.joblib", "records.csv",
        "annotations.jsonl", "reviews.sqlite",
    ),
    "source-media-or-archive": (
        "slides.pptx", "scan.dcm", "volume.nii", "recording.wav",
        "archive.zip",
    ),
    "credential-file": (
        ".netrc", "tools/.npmrc", "id_ed25519", "id_ed25519.pub",
        "nested/.ssh/config", ".env.local", "service.pem",
    ),
    "local-development-file": (
        "nested/.cache/state", "frontend/node_modules/dependency/index.js",
        "nested/.venv/lib/module.py", "nested/build/generated.js",
    ),
    "internal-notes": (
        "nested/.idea/workspace.xml", "nested/.vscode/settings.json",
    ),
    "artifact-not-in-public-allowlist": (
        "review.json", "nested/report.JSON5", "settings.jsonc",
        "report.html", "nested/report.HTM", "analysis.ipynb",
        "nested/schemas/enhancement-record.schema.json",
        "schemas/Enhancement-record.schema.json",
        "nested/src/yasargil/review_ui/index.html",
        "src/yasargil/review_ui/Index.html",
    ),
}
PUBLIC_SOURCE_FILES = (
    "src/yasargil/example.py", "tests/test_example.py", "pyproject.toml",
    "uv.lock", "src/yasargil/review_ui/app.js",
    "src/yasargil/review_ui/styles.css",
    "schemas/enhancement-record.schema.json",
    "src/yasargil/review_ui/index.html",
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

    def test_asset_and_local_file_paths_are_rejected(self):
        for category, names in LEAK_PATHS.items():
            for name in names:
                with self.subTest(name=name):
                    self.assertIn(
                        (1, category), list(hygiene.findings(name, b"synthetic"))
                    )

    def test_source_and_exact_artifact_exceptions_remain_publishable(self):
        for name in PUBLIC_SOURCE_FILES:
            with self.subTest(name=name):
                self.assertEqual(list(hygiene.findings(name, b"synthetic")), [])

    def test_artifact_exceptions_keep_content_checks(self):
        synthetic_secret = b"gh" + b"p_" + b"A" * 30
        for name in (
            "schemas/enhancement-record.schema.json",
            "src/yasargil/review_ui/index.html",
        ):
            with self.subTest(name=name):
                self.assertIn(
                    (1, "secret-or-private-key"),
                    list(hygiene.findings(name, synthetic_secret)),
                )

    def test_renamed_binary_assets_cannot_bypass_filename_checks(self):
        for name, content in (
            ("assets/image.dat", b"synthetic\0binary"),
            ("assets/font.dat", b"\xff\xfe"),
            ("opaque.bin", b"synthetic\0binary"),
        ):
            with self.subTest(name=name):
                self.assertIn(
                    (1, "binary-not-in-public-allowlist"),
                    list(hygiene.findings(name, content)),
                )

    def test_gitignore_matches_path_guards_and_preserves_public_files(self):
        excluded = {name for names in LEAK_PATHS.values() for name in names}
        candidates = sorted(excluded | set(PUBLIC_SOURCE_FILES) | {BANNER})
        # The checker enforces exact path case regardless of local Git settings.
        result = subprocess.run(
            [
                "git", "-c", "core.ignorecase=false", "-C", str(ROOT),
                "check-ignore", "--no-index", "--stdin",
            ],
            input="\n".join(candidates) + "\n",
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(set(result.stdout.splitlines()), excluded)


class RepositoryHygieneGitTests(unittest.TestCase):
    """Exercise actual Git state without modifying the development repository."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.git("init", "--quiet")
        self.git("config", "core.excludesFile", "/dev/null")
        (self.root / ".gitignore").write_bytes((ROOT / ".gitignore").read_bytes())

    def git(self, *args):
        return subprocess.run(
            ["git", "-c", "core.ignorecase=false", "-C", str(self.root), *args],
            capture_output=True,
            text=True,
            check=True,
        )

    def check(self, *args):
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts/check_repo_hygiene.py"), *args],
            cwd=self.root,
            capture_output=True,
            text=True,
        )

    def test_forced_add_assets_are_still_rejected(self):
        files = {
            "assets/font.WOFF2": "vendored-font",
            "snapshot.ckpt": "dataset-or-generated-data",
            "report.json": "artifact-not-in-public-allowlist",
        }
        for name in files:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic", encoding="utf-8")
        ignored = self.git("check-ignore", "--", *files)
        self.assertEqual(set(ignored.stdout.splitlines()), set(files))
        self.git("add", "--force", "--", *files)

        for args in ((), ("--staged",)):
            with self.subTest(args=args):
                result = self.check(*args)
                self.assertEqual(result.returncode, 1, result.stderr)
                for name, category in files.items():
                    self.assertIn(f"{name}:1: {category}", result.stdout)

    def test_staged_secret_is_found_after_worktree_is_sanitized(self):
        synthetic_secret = "gh" + "p_" + "A" * 30
        path = self.root / "settings.py"
        path.write_text(f'TOKEN = "{synthetic_secret}"\n', encoding="utf-8")
        self.git("add", "--", path.name)
        path.write_text('TOKEN = ""\n', encoding="utf-8")

        worktree = self.check()
        self.assertEqual(worktree.returncode, 0, worktree.stderr)
        staged = self.check("--staged")
        self.assertEqual(staged.returncode, 1, staged.stderr)
        self.assertIn("settings.py:1: secret-or-private-key", staged.stdout)
        self.assertFalse(synthetic_secret in staged.stdout + staged.stderr)


if __name__ == "__main__":
    unittest.main()

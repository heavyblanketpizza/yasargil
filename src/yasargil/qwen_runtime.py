"""Identity and integrity checks for the locally patched Qwen runtime."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


RUNTIME_REVISION = "b10809-qwen-reference-v1"
BASE_COMMIT = "5266f24da75dc449bd56cbed7addb9c8e4a6a73e"
SOURCE_ARCHIVE_SHA256 = "2de0d87eda4696e9f6bbd771d4c623267f4e95856cce6f99793f91522f993e43"
PATCH_RELATIVE_PATH = Path("scripts/runtime_patches/qwen-reference-v1.patch")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_qwen_runtime(project_root: Path) -> tuple[Path, dict]:
    """Reject an absent, outdated or modified build rather than silently falling back."""
    directory = project_root / ".runtime/llama.cpp" / RUNTIME_REVISION
    receipt_path = directory / "build-receipt.json"
    try:
        receipt = json.loads(receipt_path.read_text())
        if (receipt["revision"] != RUNTIME_REVISION or receipt["base_commit"] != BASE_COMMIT
                or receipt["source_archive_sha256"] != SOURCE_ARCHIVE_SHA256
                or receipt["patch_sha256"] != file_sha256(project_root / PATCH_RELATIVE_PATH)):
            raise ValueError("Qwen runtime build does not match the current pinned patch.")
        artifacts = receipt["artifacts"]
        if not isinstance(artifacts, dict) or "bin/llama-server" not in artifacts:
            raise ValueError("Qwen runtime receipt omits the server.")
        for relative, expected_hash in artifacts.items():
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
                raise ValueError("Invalid runtime artifact path.")
            if file_sha256(directory / path) != expected_hash:
                raise ValueError(f"Qwen runtime artifact changed: {relative}")
        return directory / "bin/llama-server", receipt
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(
            f"Verified Qwen reference runtime is unavailable: {error} "
            "Run .venv/bin/python scripts/build_qwen_runtime.py; see docs/LLAMA_CPP.md."
        ) from error

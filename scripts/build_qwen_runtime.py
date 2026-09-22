#!/usr/bin/env python3
"""Build the pinned Qwen patch separately from the untouched release runtime."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from yasargil.qwen_runtime import (
    BASE_COMMIT, PATCH_RELATIVE_PATH, RUNTIME_REVISION, SOURCE_ARCHIVE_SHA256, file_sha256,
)


def source_inventory(directory: Path) -> dict[str, str]:
    """Hash every source file, rejecting links and other non-archive entries."""
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"Source path is not a regular directory: {directory}")
    inventory = {}
    for path in sorted(directory.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISREG(mode):
            inventory[path.relative_to(directory).as_posix()] = file_sha256(path)
        elif not stat.S_ISDIR(mode):
            raise ValueError(f"Unexpected non-regular source entry: {path}")
    return inventory


def prepare_verified_source(
    archive: Path, source_dir: Path, patch: Path, expected_archive_sha256: str,
) -> dict[str, str]:
    """Reuse a source tree only when it exactly matches freshly patched sources.

    Out-of-source CMake writes build-info.cpp into the build directory, so no
    generated source-file exclusions are necessary for this pinned revision.
    Existing sources are never patched, overwritten, or removed here.
    """
    if file_sha256(archive) != expected_archive_sha256:
        raise ValueError("Source archive checksum mismatch.")
    with tempfile.TemporaryDirectory(prefix="qwen-source-verify-", dir=source_dir.parent) as temporary:
        staging = Path(temporary)
        with tarfile.open(archive) as source:
            source.extractall(staging, filter="data")
        expected_dir = staging / source_dir.name
        subprocess.run(["patch", "--batch", "--forward", "-p1", "-i", str(patch.resolve())],
                       cwd=expected_dir, check=True)
        expected = source_inventory(expected_dir)
        if source_dir.exists() or source_dir.is_symlink():
            actual = source_inventory(source_dir)
            missing = sorted(expected.keys() - actual.keys())
            extra = sorted(actual.keys() - expected.keys())
            changed = sorted(name for name in expected.keys() & actual.keys()
                             if expected[name] != actual[name])
            if missing or extra or changed:
                differences = "; ".join(f"{kind}: {', '.join(names[:8])}"
                                        for kind, names in (("missing", missing), ("extra", extra),
                                                            ("changed", changed)) if names)
                raise ValueError(
                    f"Existing source tree differs from the pinned archive and patch ({differences}). "
                    f"Preserve or move {source_dir} before rebuilding; it has not been changed."
                )
        else:
            expected_dir.rename(source_dir)
        return expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmake", help="CMake executable (also checks the project-local build tool).")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    local_cmake = ROOT / ".runtime/build-deps/cmake/data/bin/cmake"
    cmake = args.cmake or shutil.which("cmake") or (str(local_cmake) if local_cmake.is_file() else None)
    if not cmake:
        parser.error("Install CMake, or use: uv pip install --target .runtime/build-deps cmake==3.31.6")
    sources = ROOT / ".runtime/sources"
    sources.mkdir(parents=True, exist_ok=True)
    archive = sources / f"llama-{BASE_COMMIT}.tar.gz"
    if not archive.exists():
        temporary = archive.with_suffix(".download")
        try:
            with urlopen(f"https://codeload.github.com/ggml-org/llama.cpp/tar.gz/{BASE_COMMIT}", timeout=120) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output)
            if file_sha256(temporary) != SOURCE_ARCHIVE_SHA256:
                raise ValueError("Downloaded source archive does not match the pinned digest.")
            temporary.rename(archive)
        finally:
            temporary.unlink(missing_ok=True)
    source_dir = sources / f"llama.cpp-{BASE_COMMIT}"
    patch = ROOT / PATCH_RELATIVE_PATH
    prepare_verified_source(archive, source_dir, patch, SOURCE_ARCHIVE_SHA256)
    destination = ROOT / ".runtime/llama.cpp" / RUNTIME_REVISION
    destination.mkdir(parents=True, exist_ok=True)
    # A failed rebuild must never leave a receipt advertising stale binaries.
    (destination / "build-receipt.json").unlink(missing_ok=True)
    configure = [str(cmake), "-S", str(source_dir), "-B", str(destination),
                 "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_BUILD_NUMBER=10809",
                 f"-DLLAMA_BUILD_COMMIT={BASE_COMMIT[:9]}-qwenref1",
                 "-DLLAMA_BUILD_TESTS=ON", "-DLLAMA_BUILD_EXAMPLES=OFF",
                 "-DLLAMA_BUILD_TOOLS=ON", "-DLLAMA_BUILD_SERVER=ON", "-DLLAMA_OPENSSL=OFF",
                 "-DGGML_METAL=ON", "-DGGML_METAL_EMBED_LIBRARY=ON", "-DMTMD_VIDEO=ON"]
    build = [str(cmake), "--build", str(destination), "--config", "Release", "--target", "llama-server", "--parallel", str(args.jobs)]
    subprocess.run(configure, check=True)
    subprocess.run(build, check=True)
    artifacts = {}
    for path in sorted((destination / "bin").iterdir()):
        if path.is_file() and (path.name == "llama-server" or ".dylib" in path.name or ".so" in path.name or path.suffix in (".dll", ".metallib")):
            artifacts[path.relative_to(destination).as_posix()] = file_sha256(path)
    version = subprocess.check_output([str(destination / "bin/llama-server"), "--version"], stderr=subprocess.STDOUT, text=True)
    receipt = {"revision": RUNTIME_REVISION, "base_commit": BASE_COMMIT,
               "source_archive_sha256": SOURCE_ARCHIVE_SHA256, "patch_sha256": file_sha256(patch),
               "built_at": datetime.now(timezone.utc).isoformat(), "configure_command": configure,
               "build_command": build, "binary_version": version, "artifacts": artifacts}
    temporary_receipt = destination / "build-receipt.json.tmp"
    temporary_receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary_receipt.replace(destination / "build-receipt.json")
    print(f"Built {RUNTIME_REVISION}: {destination / 'bin/llama-server'}")


if __name__ == "__main__":
    main()

"""Durable local checkpoints and an advisory single-writer directory lock.

Callers choose and validate their output root. These helpers require existing
directories, reject explicit traversal and final-component symlinks, and pin the
opened directory while publishing files. They do not establish an output root.
"""
from __future__ import annotations

import errno
import fcntl
import json
import os
import secrets
import stat
import threading
from contextlib import contextmanager, suppress
from pathlib import Path

from .contract import ContractError, require

LOCK_FILENAME = ".writer.lock"
_held_locks = set()
_held_guard = threading.Lock()


def _open_directory(directory):
    directory = Path(directory)
    require(".." not in directory.parts, f"Directory traversal is not allowed: {directory}")
    try:
        return os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        if exc.errno in {errno.ENOENT, errno.ENOTDIR, errno.ELOOP}:
            raise ContractError(f"Expected an existing directory without a symlink target: {directory}") from exc
        raise


def durable_mkdir(path):
    """Create directories and fsync each directory and its containing entry.

    Existing paths are flushed too, so a resume can finish an interrupted
    directory creation. The supplied target cannot be a symlink; symlinks in
    existing ancestors (such as /tmp on macOS) are resolved before creation.
    """
    path = Path(path)
    require(".." not in path.parts, f"Directory traversal is not allowed: {path}")
    require(not path.is_symlink(), f"Directory target must not be a symlink: {path}")
    target = path.resolve()

    def ensure(directory):
        if not directory.exists():
            ensure(directory.parent)
        parent_fd = _open_directory(directory.parent)
        directory_fd = None
        try:
            if directory.name:
                try:
                    os.mkdir(directory.name, dir_fd=parent_fd)
                except FileExistsError:
                    pass
            directory_fd = _open_directory(directory)
            os.fsync(directory_fd)
            os.fsync(parent_fd)
        finally:
            if directory_fd is not None:
                os.close(directory_fd)
            os.close(parent_fd)

    ensure(target)
    return path


def _check_target(directory_fd, name, *, overwrite):
    try:
        target = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    require(stat.S_ISREG(target.st_mode), f"Checkpoint target must be a regular file, not a symlink: {name}")
    require(overwrite, f"Checkpoint already exists: {name}")


def atomic_bytes(path, data, *, overwrite=False):
    """Publish complete bytes and fsync the file and parent directory.

    Existing files are protected by a race-safe hard-link publication unless
    overwrite=True is explicitly supplied for a mutable checkpoint. A failure
    before publication preserves the old target. If the final directory fsync
    fails, the complete new target may already be visible; durability is then
    unconfirmed. Temporary files are removed on ordinary exceptions.
    """
    path = Path(path)
    require(".." not in path.parts and path.name not in {"", ".", "..", LOCK_FILENAME},
            f"Invalid checkpoint path: {path}")
    require(isinstance(data, (bytes, bytearray, memoryview)), "Checkpoint data must be bytes")
    directory_fd = _open_directory(path.parent)
    temporary = None
    try:
        _check_target(directory_fd, path.name, overwrite=overwrite)
        # All operations use the pinned directory, including creation and cleanup.
        while True:
            candidate = f".checkpoint-{secrets.token_hex(16)}.tmp"
            try:
                file_fd = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                  0o600, dir_fd=directory_fd)
            except FileExistsError:
                continue
            temporary = candidate
            break
        with os.fdopen(file_fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        _check_target(directory_fd, path.name, overwrite=overwrite)
        if overwrite:
            os.replace(temporary, path.name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
            temporary = None
        else:
            try:
                os.link(temporary, path.name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd,
                        follow_symlinks=False)
            except FileExistsError as exc:
                raise ContractError(f"Checkpoint already exists: {path}") from exc
            os.unlink(temporary, dir_fd=directory_fd)
            temporary = None
        os.fsync(directory_fd)
    finally:
        try:
            if temporary is not None:
                with suppress(FileNotFoundError):
                    os.unlink(temporary, dir_fd=directory_fd)
        finally:
            os.close(directory_fd)
    return path


def atomic_json(path, value, *, overwrite=False):
    """Atomically write finite UTF-8 JSON, preserving Unicode and a final newline."""
    try:
        data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ContractError("Checkpoint value must be finite UTF-8 JSON") from exc
    return atomic_bytes(path, data, overwrite=overwrite)


def _open_lock(directory_fd, *, create):
    try:
        lock_fd = os.open(LOCK_FILENAME, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK |
                          (os.O_CREAT if create else 0),
                          0o600, dir_fd=directory_fd)
    except FileNotFoundError:
        if not create:
            return None
        raise
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ContractError("Writer lock must not be a symlink") from exc
        raise
    try:
        info = os.fstat(lock_fd)
        require(stat.S_ISREG(info.st_mode), "Writer lock must be a regular file")
    except BaseException:
        os.close(lock_fd)
        raise
    return lock_fd


def _lock_key(lock_fd):
    info = os.fstat(lock_fd)
    return os.getpid(), info.st_dev, info.st_ino


@contextmanager
def directory_lock(directory):
    """Hold an exclusive nonblocking advisory lock; retain its inode on release."""
    directory_fd = _open_directory(directory)
    lock_fd = None
    acquired = False
    key = None
    try:
        lock_fd = _open_lock(directory_fd, create=True)
        key = _lock_key(lock_fd)
        with _held_guard:
            require(key not in _held_locks, f"Directory already has an active writer: {directory}")
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in {errno.EACCES, errno.EAGAIN}:
                    raise ContractError(f"Directory already has an active writer: {directory}") from exc
                raise
            acquired = True
            _held_locks.add(key)
        # A competing lock-file replacement must not silently split the lock.
        info = os.stat(LOCK_FILENAME, dir_fd=directory_fd, follow_symlinks=False)
        require((info.st_dev, info.st_ino) == key[1:] and stat.S_ISREG(info.st_mode),
                "Writer lock changed during acquisition")
        os.fsync(directory_fd)
        yield Path(directory)
    finally:
        try:
            if acquired:
                with _held_guard:
                    try:
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    finally:
                        _held_locks.discard(key)
        finally:
            if lock_fd is not None:
                os.close(lock_fd)
            os.close(directory_fd)


def directory_is_locked(directory):
    """Return a momentary writer-lock state without creating or changing files."""
    directory_fd = _open_directory(directory)
    lock_fd = None
    try:
        lock_fd = _open_lock(directory_fd, create=False)
        if lock_fd is None:
            return False
        with _held_guard:
            if _lock_key(lock_fd) in _held_locks:
                return True
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in {errno.EACCES, errno.EAGAIN}:
                    return True
                raise
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            return False
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        os.close(directory_fd)

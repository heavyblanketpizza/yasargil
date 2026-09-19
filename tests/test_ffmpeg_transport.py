"""Byte transport, argument preservation, and actual strict native decoding."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from yasargil.ffmpeg_transport import (
    RECEIPT_DIRECTORY_ENV, install_transport, replace_cached_input, transport_metadata, transport_receipts,
)
from yasargil.llama_video import VideoRuntimeError, verify_native_video_decode
import test_llama_video as native_fixtures


class FFmpegTransportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        real = self.root / "real"
        real.mkdir()
        script = f"#!{sys.executable}\n" + '''import hashlib,json,pathlib,sys
arguments=sys.argv[1:]
path=pathlib.Path(arguments[arguments.index('-i')+1]) if '-i' in arguments else None
data=path.read_bytes() if path else sys.stdin.buffer.read()
print(json.dumps({'arguments':arguments,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
                  'path':str(path) if path else None,'mode':path.stat().st_mode&0o777 if path else None,
                  'directory_mode':path.parent.stat().st_mode&0o777 if path else None}))
sys.exit(7 if '--fail' in arguments else 0)
'''
        for name in ("ffmpeg", "ffprobe"):
            path = real / name
            path.write_text(script)
            path.chmod(0o755)
        self.metadata = install_transport(self.root / "project with spaces", ffmpeg=real / "ffmpeg", ffprobe=real / "ffprobe")
        self.directory = Path(self.metadata["directory"])

    def execute(self, arguments, data=b"", *, program="ffmpeg", env=None):
        return subprocess.run([str(self.directory / program), *arguments], input=data, capture_output=True, env=env)

    def test_complete_stdin_bytes_are_spooled_without_changing_decode_arguments(self):
        data = bytes(range(256)) * (32 * 1024) + b"final bytes\x00\xff"
        arguments = ["-nostdin", "-read_ahead_limit", "-1", "-i", "cache:pipe:0", "-vf", "fps=1.000000",
                     "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1", "-loglevel", "error"]
        result = self.execute(arguments, data)
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        decoded = json.loads(result.stdout)
        receipt = transport_receipts(result.stderr.decode())[0]
        self.assertEqual(decoded["bytes"], len(data))
        self.assertEqual(decoded["sha256"], hashlib.sha256(data).hexdigest())
        self.assertEqual(receipt["input_sha256"], decoded["sha256"])
        self.assertEqual(receipt["input_bytes"], len(data))
        self.assertEqual(receipt["arguments"], arguments)
        self.assertEqual(decoded["arguments"], replace_cached_input(arguments, decoded["path"]))
        self.assertEqual((decoded["mode"], decoded["directory_mode"]), (0o600, 0o700))
        self.assertFalse(Path(decoded["path"]).exists())

    def test_normal_file_and_ffprobe_calls_are_forwarded_unchanged(self):
        source = self.root / "original bytes.bin"
        source.write_bytes(b"unchanged original source")
        arguments = ["-read_ahead_limit", "123", "-i", str(source), "-vf", "fps=2.000000", "-f", "rawvideo", "pipe:1"]
        result = self.execute(arguments)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(json.loads(result.stdout)["arguments"], arguments)
        self.assertEqual(result.stderr, b"")
        arguments = ["-v", "quiet", "-show_streams", "pipe:0"]
        result = self.execute(arguments, b"entire ffprobe stdin", program="ffprobe")
        self.assertEqual(json.loads(result.stdout)["arguments"], arguments)
        self.assertEqual(json.loads(result.stdout)["sha256"], hashlib.sha256(b"entire ffprobe stdin").hexdigest())
        self.assertEqual(result.stderr, b"")

    def test_only_cache_options_for_the_target_input_are_removed(self):
        arguments = ["-read_ahead_limit", "99", "-i", "first.mp4", "-read_ahead_limit", "-1",
                     "-i", "cache:pipe:0", "-read_ahead_limit", "77", "-i", "third.mp4", "-vf", "fps=1"]
        self.assertEqual(replace_cached_input(arguments, "/tmp/spooled.bin"),
                         ["-read_ahead_limit", "99", "-i", "first.mp4", "-i", "/tmp/spooled.bin",
                          "-read_ahead_limit", "77", "-i", "third.mp4", "-vf", "fps=1"])
        with self.assertRaises(ValueError):
            replace_cached_input(["-i", "cache:pipe:0", "-i", "cache:pipe:0"], "/tmp/x")

    def test_decoder_failure_keeps_the_byte_receipt_and_cleans_temporary_input(self):
        result = self.execute(["-read_ahead_limit", "-1", "-i", "cache:pipe:0", "--fail"], b"all input bytes")
        self.assertEqual(result.returncode, 7)
        receipt = transport_receipts(result.stderr.decode())[0]
        self.assertEqual(receipt["input_sha256"], hashlib.sha256(b"all input bytes").hexdigest())
        self.assertFalse(Path(receipt["temporary_input"]).exists())

    def test_native_receipt_is_saved_even_when_child_stderr_is_not_forwarded(self):
        receipts = self.root / "receipts"
        receipts.mkdir()
        result = self.execute(["-i", "cache:pipe:0"], b"complete video bytes",
                              env={**os.environ, RECEIPT_DIRECTORY_ENV: str(receipts)})
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        records = list(receipts.glob("*.json"))
        self.assertEqual(len(records), 1)
        self.assertEqual(json.loads(records[0].read_text()), transport_receipts(result.stderr.decode())[0])
        self.assertEqual(len(list(receipts.iterdir())), 1)

    def test_changed_real_binary_is_rejected_before_decoding(self):
        Path(self.metadata["real_ffmpeg"]["path"]).write_text("modified executable")
        with self.assertRaisesRegex(ValueError, "component changed"):
            transport_metadata(self.directory)
        result = self.execute(["-i", "cache:pipe:0"], b"original video")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"")

    def test_empty_input_is_rejected_without_decoder_output(self):
        result = self.execute(["-i", "cache:pipe:0"])
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"")
        self.assertIn(b"empty", result.stderr)
        self.assertEqual(transport_receipts(result.stderr.decode()), [])


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "Real FFmpeg required")
class FFmpegTransportDecodeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = native_fixtures.NativeDecodeIntegrationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.metadata = install_transport(self.fixture.root)
        self.directory = Path(self.metadata["directory"])

    def verify(self, video):
        return verify_native_video_decode(video, ffmpeg=str(self.directory / "ffmpeg"),
                                          ffprobe=str(self.directory / "ffprobe"), expected_video_frames=4,
                                          output_dir=self.fixture.root / "verify")

    def test_transport_preserves_every_real_cfr_frame_and_original_video_bytes(self):
        video = self.fixture.make_video()
        original = video.read_bytes()
        result = self.verify(video)
        self.assertTrue(result["accepted"])
        self.assertTrue(result["ordered_rgb_frames_identical"])
        self.assertEqual(result["native_decoded_frames"], 4)
        self.assertEqual(result["ffmpeg_transport"], self.metadata)
        for receipt in result["byte_transport_receipts"].values():
            self.assertEqual(receipt["input_sha256"], hashlib.sha256(original).hexdigest())
            self.assertEqual(receipt["input_bytes"], len(original))
        self.assertEqual(video.read_bytes(), original)

    def test_transport_does_not_weaken_duplicate_or_missing_frame_rejection(self):
        video = self.fixture.make_video(extra=("-vf", "setpts=if(eq(N\\,3)\\,5\\,N)/(4*TB)", "-fps_mode", "vfr"))
        with self.assertRaisesRegex(VideoRuntimeError, "changes the frame sequence"):
            self.verify(video)


if __name__ == "__main__":
    unittest.main()

"""Only current, human-approved MedGemma annotations become loadable SFT rows."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from yasargil.__main__ import main
from yasargil.annotation_export import INSTRUCTION, annotation_text, export_reviewed_annotations
from yasargil.contract import ANNOTATION_EXPORT_VERSION, ContractError, load_export
from yasargil.dataset_inspector import InspectorStore
from yasargil.medgemma_annotation_contract import PROTOCOL_V1, build_annotation
from yasargil.medgemma_annotation_evidence import canonical_frame


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def png(path, color, size=(16, 16)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


class AnnotationExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.outputs = self.root / "outputs"
        self.dataset = self.root / "dataset"
        self.selection = self.outputs / "selection" / "S1A1"
        self.medgemma = self.outputs / "medgemma" / "S1A1"
        self.frames = []
        for index, color in enumerate(("red", "green", "blue", "yellow")):
            image = png(self.selection / "source/frames" / f"frame-{index:08d}.png", color)
            self.frames.append({"frame_id": f"f{index}", "frame_index": index, "release_frame_index": index + 1,
                "timestamp_ms": index * 1000, "timestamp_basis": "reconstructed_nominal",
                "source_path": str(self.dataset / "frames/S1A1" / f"S1A1_frame_{index + 1:08d}.jpeg"),
                "source_sha256": "a" * 64, "image_path": str(image), "image_sha256": digest(image),
                "width": 16, "height": 16, "source_pts": None, "time_base": None})
        video = self.selection / "source/video.mp4"
        video.write_bytes(b"video")
        self.source = {"source_kind": "released_image_sequence", "source_path": str(self.dataset / "frames/S1A1"),
            "source_sha256": "b" * 64, "video_path": str(video), "video_sha256": digest(video),
            "frames": self.frames, "duration_ms": 4000, "expected_video_frames": 4,
            "timestamp_basis": "reconstructed_nominal"}
        selection = {"schema_version": "smart-frame-selection-v1", "status": "completed",
            "video_sha256": self.source["video_sha256"], "selected_frame_ids": ["f0", "f2"],
            "frames": [{**frame, "model_decision": "keep" if frame["frame_id"] in {"f0", "f2"} else "drop",
                        "effective_decision": "keep" if frame["frame_id"] in {"f0", "f2"} else "drop"}
                       for frame in self.frames]}
        write(self.selection / "source/source.json", self.source)
        write(self.selection / "run.json", {"schema_version": "smart-frame-selection-run-v1",
              "created_at": "2026-01-01", "config": {"procedure_context": "Simulated repair"}})
        write(self.selection / "selection.json", selection)
        self.annotations = {}
        rows, evidence_files = [], []
        for index in (0, 2):
            packet = self.packet(index)
            annotation = build_annotation({"target_frame_id": f"f{index}", "visibility": "partial",
                "claims": [{"claim_id": "c1", "category": "instrument", "statement": f"Instrument jaws are visible in f{index}.",
                            "support": "target_visible", "evidence_view_ids": [f"f{index}:detail:1"],
                            "uncertainty": "Subtype uncertain"}],
                "unresolved_questions": [{"question": "Which tissue is visible?", "reason": "Boundary obscured",
                                          "kind": "target_detail"}]}, packet, PROTOCOL_V1)
            self.annotations[f"f{index}"] = annotation
            name = f"evidence/frame-{index:04d}.json"
            write(self.medgemma / name, packet)
            evidence_files.append(name)
            rows.append({"target_frame_id": f"f{index}", "target": packet["target"], "annotation": annotation,
                         "evidence": packet, "call_directory": f"calls/frame-{index:04d}/attempt-0001"})
        write(self.medgemma / "source.json", self.source)
        write(self.medgemma / "selected-frames.json", [self.frames[0], self.frames[2]])
        write(self.medgemma / "selection/selection.json", selection)
        names = ["source.json", "selected-frames.json", "selection/selection.json", *evidence_files]
        write(self.medgemma / "run.json", {"schema_version": "medgemma-frame-annotation-v1", "created_at": "2026-02-01",
            "selection_run": str(self.selection), "source_file": "source.json", "selected_file": "selected-frames.json",
            "frame_ids": ["f0", "f2"], "evidence_files": evidence_files,
            "input_sha256": {name: digest(self.medgemma / name) for name in names}})
        write(self.medgemma / "annotations.json", {"schema_version": "medgemma-frame-annotation-v1",
              "annotations": rows, "human_review_required": True, "training_eligible": False})
        write(self.medgemma / "summary.json", {"status": "completed"})
        self.store = InspectorStore(self.outputs)
        self.record_id = self.store.records()["records"][0]["id"]
        self.release = self.outputs / "exports" / "release-1"

    def packet(self, index):
        target, context = canonical_frame(self.frames[index]), canonical_frame(self.frames[index + 1])
        crop = png(self.medgemma / "assets" / f"frame-{index:04d}" / "detail-1.png", "white", (8, 8))

        def view(frame, role, image=None, size=16):
            return {"view_id": f"{frame['frame_id']}:{'detail:1' if image else 'full'}", "frame_id": frame["frame_id"],
                    "image_path": str(image or frame["image_path"]),
                    "image_sha256": digest(image) if image else frame["image_sha256"],
                    "width": size, "height": size, "bounds": [0, 0, size, size], "role": role}

        return {"schema_version": "medgemma-annotation-evidence-v1", "target_frame_id": target["frame_id"],
                "target": target, "frames": [target, context],
                "views": [view(target, "target"), view(target, "target_detail", crop, 8), view(context, "context_after")],
                "procedure_context": "Documented simulated repair.", "limitations": []}

    def curate(self, frame_id, action, text=None):
        identity = self.store.record(self.record_id)["review_identity"]
        request = {"review_identity": identity, "action": action}
        if text is not None:
            request["annotations"] = {"medgemma": text}
        return self.store.curate(self.record_id, frame_id, request)

    def worksheet(self, statuses, *, name="S1A1-human-review.json", **overrides):
        record = self.store.record(self.record_id)
        curation = {frame["frame_id"]: frame["curation"] for frame in record["frames"]}
        notes = [{"frame_id": frame_id, "status": status, "note": "", "updated_at": "2026-09-27T00:00:00Z",
                  "curation_updated_at": (curation[frame_id] or {}).get("updated_at"), **overrides.get(frame_id, {})}
                 for frame_id, status in statuses.items()]
        path = self.root / name
        write(path, {"schema_version": "yasargil-inspector-review-v1", "worksheet_status": "draft",
                     "record_id": self.record_id, "review_identity": record["review_identity"], "case_id": "S1A1",
                     "notes": notes, "enhancements": [], "training_eligible": False})
        return path

    def export(self, worksheets, output="train.jsonl", partition="train", **kwargs):
        options = {"runs_root": self.outputs, "partition": partition, "reviewer": "Example Reviewer",
                   "reviewer_role": "surgeon", **kwargs}
        return export_reviewed_annotations(worksheets, self.release / output, **options)

    def load(self, output="train.jsonl", partition="train"):
        return load_export(self.release / output, self.dataset, artifact_root=self.outputs,
                           expected_partition=partition, expected_intended_use="retrospective_surgical_review")

    def test_complete_frames_export_through_the_cli_and_load_for_training(self):
        self.curate("f2", "edit", "Human-corrected: the needle driver grips the dura edge.")
        path = self.worksheet({"f0": "reviewed", "f1": "reviewed", "f2": "reviewed"})
        with redirect_stdout(io.StringIO()) as printed:
            main(["export-reviewed-annotations", "--review", str(path), "--runs-root", str(self.outputs),
                  "--partition", "train", "--reviewer", "Example Reviewer", "--reviewer-role", "surgeon",
                  "--output", str(self.release / "train.jsonl")])
        self.assertIn("Wrote 2 reviewed row(s)", printed.getvalue())
        receipt = json.loads((self.release / "train.receipt.json").read_text())
        self.assertEqual(receipt["schema_version"], ANNOTATION_EXPORT_VERSION)
        self.assertEqual([row["record_id"] for row in receipt["records"]], ["S1A1:f0", "S1A1:f2"])
        self.assertEqual(receipt["review"]["reviewer_role"], "surgeon")
        self.assertIs(receipt["review"]["identity_verified"], False)
        self.assertEqual(receipt["review"]["worksheets"][0]["surgeon_id"], "S1")
        rows = self.load()
        self.assertEqual(len(rows), 2)
        user, assistant = rows[0]["messages"]
        self.assertEqual(user["content"][0], {"type": "text", "text": INSTRUCTION})
        images = [block["image"] for block in user["content"] if block["type"] == "image"]
        self.assertEqual([image.size for image in images], [(16, 16), (8, 8), (16, 16)])
        self.assertEqual(assistant["content"][0]["text"], annotation_text(self.annotations["f0"]))
        self.assertTrue(assistant["content"][0]["text"].startswith("Visibility: partial\n\ntarget visible · instrument:"))
        self.assertEqual(rows[1]["messages"][1]["content"][0]["text"],
                         "Human-corrected: the needle driver grips the dura edge.")

    def test_only_current_complete_decisions_become_rows(self):
        self.curate("f2", "delete")
        flagged = self.worksheet({"f0": "flagged", "f2": "reviewed"})
        with self.assertRaisesRegex(ContractError, "No current frames are marked Complete"):
            self.export([flagged])
        recheck = self.worksheet({"f0": "reviewed"}, f0={"needs_recheck": True})
        with self.assertRaisesRegex(ContractError, "No current frames are marked Complete"):
            self.export([recheck])
        stale = self.worksheet({"f0": "reviewed"})
        self.curate("f0", "edit", "Changed after the review was saved.")
        with self.assertRaisesRegex(ContractError, "No current frames are marked Complete"):
            self.export([stale])
        self.assertFalse(self.release.exists())

    def test_review_of_a_different_data_revision_is_refused(self):
        path = self.worksheet({"f0": "reviewed"})
        value = json.loads(path.read_text())
        value["review_identity"] = "0" * 64
        write(path, value)
        with self.assertRaisesRegex(ContractError, "dataset changed"):
            self.export([path])

    def test_reviewer_partition_and_output_are_required(self):
        path = self.worksheet({"f0": "reviewed"})
        for options, message in (({"reviewer": " "}, "Declare who reviewed"),
                                 ({"reviewer_role": "student"}, "Reviewer role"),
                                 ({"partition": "unassigned"}, "explicit train"),
                                 ({"output": "train.json"}, ".jsonl")):
            with self.subTest(options=options), self.assertRaisesRegex(ContractError, message):
                self.export([path], **options)
        self.export([path])
        with self.assertRaisesRegex(ContractError, "already exists"):
            self.export([path])

    def test_release_blocks_repeated_cases_and_cross_partition_surgeons(self):
        path = self.worksheet({"f0": "reviewed"})
        self.export([path])
        with self.assertRaisesRegex(ContractError, "already exported in this release"):
            self.export([path], output="validation.jsonl", partition="validation")
        other = self.outputs / "exports" / "release-2"
        write(other / "validation.receipt.json", {"schema_version": ANNOTATION_EXPORT_VERSION, "partition": "validation",
              "media": [], "review": {"worksheets": [{"case_id": "S1A3", "surgeon_id": "S1"}]}})
        with self.assertRaisesRegex(ContractError, "Cross-partition leakage: surgeon S1 is in validation"):
            export_reviewed_annotations([path], other / "train.jsonl", runs_root=self.outputs, partition="train",
                                        reviewer="Example Reviewer", reviewer_role="surgeon")

    def test_tampered_rows_or_images_fail_to_load(self):
        self.export([self.worksheet({"f0": "reviewed"})])
        output = self.release / "train.jsonl"
        original = output.read_text()
        output.write_text(original.replace("Instrument jaws", "Invented jaws"))
        with self.assertRaisesRegex(ContractError, "Export digest mismatch"):
            self.load()
        output.write_text(original)
        png(self.medgemma / "assets/frame-0000/detail-1.png", "black", (8, 8))
        with self.assertRaisesRegex(ContractError, "Export image hash mismatch"):
            self.load()


if __name__ == "__main__":
    unittest.main()

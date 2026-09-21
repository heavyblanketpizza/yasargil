"""Independent MedGemma inference, isolation, provenance and crash recovery."""
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from test_frame_annotation import SelectionRuntime, read, write
from yasargil.contract import ContractError, sha256_file
from yasargil.llama_cpp import LlamaCppError
from yasargil.medgemma_annotation import AnnotationConfig, run_annotation, request_annotation_pause
from yasargil.smart_selection import SelectionConfig, review_loop
from yasargil.video_source import prepare_video_source


class FakeClient:
    def __init__(self, transform=None):
        self.requests = []
        self.transform = transform
        self.last_response_bytes = None

    def model_info(self, model):
        return {"name": model, "digest": "sha256:" + "a" * 64, "quantization": "Q4_K_M",
                "runtime_version": "test", "capabilities": ["vision"], "runtime": "llama.cpp",
                "model_file": {"sha256": "a" * 64}, "projector_file": {"sha256": "b" * 64},
                "runtime_binary": {"sha256": "c" * 64}, "binary_version": "test"}

    def chat_raw(self, request):
        self.requests.append(copy.deepcopy(request))
        context = json.loads(request['messages'][-1]['content'])
        locator = json.loads(request['messages'][1]['content'][0]['text'])
        annotation = {"target_frame_id": context['target_frame_id'], "visibility": "clear",
            "claims": [{"claim_id": "c1", "category": "tissue_state", "statement": "The field is uniformly colored.",
                        "support": "target_visible", "evidence_view_ids": [locator['view_id']], "uncertainty": ""}],
            "unresolved_questions": []}
        response = {"model": request['model'], "choices": [{"finish_reason": "stop",
                    "message": {"role": "assistant", "content": json.dumps(annotation)}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 100}, "extra": "retained"}
        if self.transform:
            response = self.transform(response)
        self.last_response_bytes = json.dumps(response).encode()
        return self.last_response_bytes


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'requires FFmpeg')
class AnnotationTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.release = self.root / 'dataset/frames/S1A1'
        self.release.mkdir(parents=True)
        for i, color in enumerate(('red', 'green', 'blue', 'yellow'), 1):
            Image.new('RGB', (48, 32), color).save(self.release / f'S1A1_frame_{i:08d}.jpeg')
        self.selection = self.root / 'selection'
        self.selection.mkdir()
        (self.selection / '.run.lock').touch()
        self.source = prepare_video_source(self.release, self.selection / 'source', released_fps=1)
        self.ids = [f['frame_id'] for f in self.source['frames']]
        config = SelectionConfig(candidate_budget=4, max_candidates=8, max_retrieval_rounds=0,
                                 procedure_context='Documented synthetic sequence.')
        initial = {'selected_ids': self.ids, 'protected_ids': [self.ids[0]]}
        write(self.selection / 'run.json', {'schema_version': 'smart-frame-selection-run-v1',
              'input_path': str(self.release), 'config': asdict(config),
              'source_manifest_sha256': sha256_file(self.selection / 'source/source.json')})
        write(self.selection / 'initial-selection.json', initial)
        review_loop(self.source, initial, self.selection, config, SelectionRuntime([self.ids[2]], 4), progress=lambda _: None)
        # These artifacts must never be read by the annotation path.
        write(self.selection / 'annotations.json', {'visible_observation': 'FORBIDDEN_QWEN_DRAFT'})
        (self.root / 'dataset/sospine_tool_tips.csv').write_text('FORBIDDEN_SOURCE_LABEL')
        self.output = self.root / 'medgemma'
        self.config = AnnotationConfig(before_frames=1, after_frames=1, num_ctx=8192, num_predict=1024)

    def run_pass(self, client=None, **kwargs):
        return run_annotation(None if kwargs.get('resume') else self.selection, self.output,
                              self.config, client=client or FakeClient(), progress=lambda _: None, **kwargs)

    def test_direct_annotation_needs_no_qwen_draft_and_covers_targets(self):
        client = FakeClient()
        summary = self.run_pass(client)
        self.assertEqual(summary['status'], 'completed')
        self.assertEqual(summary['annotated_frame_count'], 2)
        self.assertEqual(len(client.requests), 2)
        for req in client.requests:
            text = json.dumps({**req, 'messages': [{**m, 'content': m['content'] if isinstance(m['content'], str)
                   else [p for p in m['content'] if p['type'] == 'text']} for m in req['messages']]})
            for sentinel in ('FORBIDDEN_QWEN_DRAFT', 'FORBIDDEN_SOURCE_LABEL', 'PRIVATE_SELECTOR_SCENE_DESCRIPTION',
                             'PRIVATE_SELECTOR_KEEP_DROP_REASON', 'source_path', 'sha256'):
                self.assertNotIn(sentinel, text)
        doc = read(self.output / 'annotations.json')
        self.assertEqual([r['target_frame_id'] for r in doc['annotations']], [self.ids[0], self.ids[2]])
        self.assertFalse(doc['training_eligible'])
        self.assertTrue((self.output / 'report.html').is_file())
        self.assertEqual(read(self.output / 'source.json'), self.source)

    def test_prepare_and_resume_and_completed_resume_make_no_duplicate_calls(self):
        client = FakeClient()
        self.assertEqual(self.run_pass(client, prepare_only=True)['status'], 'prepared')
        self.assertFalse(client.requests)
        self.run_pass(client, resume=True)
        original = (self.output / 'annotations.json').read_bytes()
        self.run_pass(client, resume=True)
        self.assertEqual(len(client.requests), 2)
        self.assertEqual((self.output / 'annotations.json').read_bytes(), original)

    def test_pause_saves_inflight_frame_then_resumes_remaining(self):
        client = FakeClient()
        original = client.chat_raw
        def pausing(request):
            response = original(request)
            request_annotation_pause(self.output)
            return response
        # Preparation publishes the summary needed by the pause endpoint.
        self.run_pass(client, prepare_only=True)
        client.chat_raw = pausing
        self.assertEqual(self.run_pass(client, resume=True)['status'], 'paused')
        self.assertEqual(len(client.requests), 1)
        client.chat_raw = original
        self.assertEqual(self.run_pass(client, resume=True)['status'], 'completed')
        self.assertEqual(len(client.requests), 2)

    def test_raw_response_recovers_after_interruption_without_regeneration(self):
        client = FakeClient()
        with patch('yasargil.medgemma_annotation._parse', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_pass(client)
        self.assertEqual(len(client.requests), 1)
        self.run_pass(client, resume=True)
        self.assertEqual(len(client.requests), 2)

    def test_published_tampering_never_overwrites_publication_or_calls_model(self):
        client = FakeClient()
        self.run_pass(client)
        path = self.output / 'annotations.json'
        doc = read(path)
        doc['annotations'][0]['annotation']['claims'][0]['statement'] = 'Tampered'
        write(path, doc)
        tampered = path.read_bytes()
        with self.assertRaisesRegex(ContractError, 'differs from accepted'):
            self.run_pass(client, resume=True)
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(path.read_bytes(), tampered)

    def test_frozen_evidence_and_source_mutations_block_inference(self):
        self.run_pass(prepare_only=True)
        path = self.output / 'evidence/frame-0000.json'
        packet = read(path)
        packet['procedure_context'] = 'Changed'
        write(path, packet)
        with self.assertRaisesRegex(ContractError, 'Frozen annotation input'):
            self.run_pass(resume=True)

    def test_modified_crop_blocks_accepted_response_reuse(self):
        client = FakeClient()
        self.run_pass(client)
        packet = read(self.output / 'evidence/frame-0000.json')
        crop = next(view for view in packet['views'] if view['role'] == 'target_detail')
        Path(crop['image_path']).write_bytes(b'changed pixels')
        with self.assertRaisesRegex(ContractError, 'Annotation view changed'):
            self.run_pass(client, resume=True)
        self.assertEqual(len(client.requests), 2)

    def test_accepted_receipt_loss_never_repeats_an_accepted_call(self):
        client = FakeClient()
        self.run_pass(client)
        (self.output / 'calls/frame-0000/attempt-0001/receipt.json').unlink()
        with self.assertRaises((ContractError, OSError)):
            self.run_pass(client, resume=True)
        self.assertEqual(len(client.requests), 2)

    def test_interrupted_preparation_does_not_publish_a_resumable_plan(self):
        from yasargil.medgemma_annotation import atomic_json
        def interrupt(path, value, **kwargs):
            if path.name == 'session.json':
                raise KeyboardInterrupt
            return atomic_json(path, value, **kwargs)
        with patch('yasargil.medgemma_annotation.atomic_json', side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_pass(prepare_only=True)
        self.assertFalse((self.output / 'run.json').exists())

    def test_bad_citation_retains_failure_bytes_then_explicit_resume_retries(self):
        def invalid(response):
            content = json.loads(response['choices'][0]['message']['content'])
            content['claims'][0]['evidence_view_ids'] = ['unknown-view']
            response['choices'][0]['message']['content'] = json.dumps(content)
            return response
        with self.assertRaises(ContractError):
            self.run_pass(FakeClient(invalid))
        attempt = self.output / 'calls/frame-0000/attempt-0001'
        self.assertTrue((attempt / 'response.json').is_file())
        self.assertTrue((attempt / 'failure.json').is_file())
        self.assertFalse((attempt / 'receipt.json').exists())
        self.assertEqual(self.run_pass(resume=True)['status'], 'completed')
        self.assertTrue((self.output / 'calls/frame-0000/attempt-0002/receipt.json').is_file())

    def test_legacy_review_plan_rejected_without_inference(self):
        self.output.mkdir()
        write(self.output / 'run.json', {'schema_version': 'medgemma-surgery-review-v1'})
        with self.assertRaisesRegex(ContractError, 'Historical Qwen-review'):
            self.run_pass(resume=True)

    def test_context_budget_failure_is_not_accepted(self):
        def overflow(response):
            response['usage']['prompt_tokens'] = self.config.num_ctx
            return response
        with self.assertRaisesRegex(ContractError, 'answer capacity'):
            self.run_pass(FakeClient(overflow))

    def test_zero_context_no_crops_ablation(self):
        self.config = replace(self.config, before_frames=0, after_frames=0, detail_crops=False)
        client = FakeClient()
        self.run_pass(client)
        self.assertTrue(all(len(req['messages']) == 3 for req in client.requests))


if __name__ == '__main__':
    unittest.main()

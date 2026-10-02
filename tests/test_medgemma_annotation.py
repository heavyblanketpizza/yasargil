"""Independent MedGemma inference, isolation, provenance and crash recovery."""
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import re
import shutil
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from yasargil.contract import ContractError, sha256_file
from yasargil.llama_cpp import LlamaCppError, LlamaCppIncompleteError
from yasargil.medgemma_annotation import (
    AnnotationConfig, annotation_status, request_annotation_pause, run_annotation,
)
from yasargil.smart_selection import SelectionConfig, review_loop
from yasargil.video_source import prepare_video_source


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value))


def publish_result(messages, schema, max_tokens, round_dir, output, frame_count):
    """Save the immutable transport receipts used by real native inference."""
    request = {"messages": copy.deepcopy(messages), "max_tokens": max_tokens,
               "response_format": {"type": "json_schema", "json_schema": {"schema": schema}}}
    response = {"choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": json.dumps(output)}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}}
    overview = json.loads(messages[1]["content"][0]["text"])
    video_url = next(block["input_video"]["url"] for block in messages[1]["content"]
                     if block["type"] == "input_video")
    round_dir.mkdir(parents=True, exist_ok=True)
    write(round_dir / "request.json", request)
    verification = {
        "accepted": True, "full_source_video_verified": True,
        "context_truncation_observed": False, "finish_reason": "stop",
        "video_sha256": overview["video_sha256"],
        "video_relative_path": video_url.removeprefix("file://"),
        "video_fps_setting": 0, "expected_video_frames": frame_count,
        "decoded_frame_ids": list(range(frame_count)), "decoded_frames": frame_count,
        "request_sha256": sha256_file(round_dir / "request.json"),
    }
    result = {"output": output, "response": response, "verification": verification}
    for name, value in (("response.json", response), ("verification.json", verification),
                        ("output.json", output), ("result.json", result)):
        write(round_dir / name, value)
    return result


class SelectionRuntime:
    def __init__(self, kept_ids, frame_count):
        self.kept_ids, self.frame_count = set(kept_ids), frame_count

    def chat(self, messages, *, schema, max_tokens, round_dir):
        ids = schema["properties"]["decisions"]["required"]
        output = {
            "scene_summary": "PRIVATE_SELECTOR_SCENE_DESCRIPTION",
            "context_check": "consistent", "ready": True, "searches": [],
            "decisions": {key: {"decision": "keep" if key in self.kept_ids else "drop",
                                "reason": "PRIVATE_SELECTOR_KEEP_DROP_REASON"} for key in ids},
        }
        return publish_result(messages, schema, max_tokens, round_dir, output, self.frame_count)


class FakeClient:
    """Answers in the model-facing v2 form; transform(response, request) can alter a reply."""
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
        answer = {"visibility": "clear", "claims": [
            {"statement": "The field is uniformly colored.", "category": "tissue_state",
             "support": "target_visible", "evidence_view_ids": ["T"], "uncertainty": ""}],
            "unresolved_questions": []}
        response = {"model": request['model'], "choices": [{"finish_reason": "stop",
                    "message": {"role": "assistant", "content": json.dumps(answer)}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 100}, "extra": "retained"}
        if self.transform:
            response = self.transform(response, request)
        self.last_response_bytes = json.dumps(response).encode()
        return self.last_response_bytes


def set_answer(response, change):
    answer = json.loads(response['choices'][0]['message']['content'])
    change(answer)
    response['choices'][0]['message']['content'] = json.dumps(answer)
    return response


def counting_loop(answer):
    answer['claims'] = [dict(answer['claims'][0], statement=f"A structure is marked with '{n}'.") for n in range(3)]


def loop_when_greedy(response, request):
    return set_answer(response, counting_loop) if request['temperature'] == 0 else response


def image_labels(request):
    """The citation label announced immediately before each image, in order."""
    parts = request['messages'][1]['content']
    return [re.match(r"View (\w+):", parts[i - 1]['text']).group(1)
            for i, part in enumerate(parts) if part['type'] == 'image_url']


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
        self.config = AnnotationConfig(before_frames=1, after_frames=1, num_ctx=16384, num_predict=8192)

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

    def test_selection_must_be_finished_and_match_last_actual_model_review(self):
        selection_path = self.selection / 'selection.json'
        original = read(selection_path)
        def unfinished(value):
            value['status'] = 'awaiting_review'
        def changed_selected_ids(value):
            value['selected_frame_ids'] = self.ids
        def invented_reason(value):
            value['frames'][0]['model_reason'] = 'Changed after model review.'
        def changed_source_locator(value):
            value['frames'][0]['timestamp_ms'] = 999
        def unsupported_full_video(value):
            value['completed_rounds_full_video_verified'] = False
        for index, change in enumerate((unfinished, changed_selected_ids, invented_reason,
                                        changed_source_locator, unsupported_full_video)):
            with self.subTest(change=change.__name__):
                self.output = self.root / f'invalid-selection-{index}'
                value = copy.deepcopy(original)
                change(value)
                write(selection_path, value)
                client = FakeClient()
                with self.assertRaises(ContractError):
                    self.run_pass(client)
                self.assertEqual(client.requests, [])
                self.assertFalse(self.output.exists())
        write(selection_path, original)

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

    def test_bad_citation_retains_failure_bytes_for_each_rejected_attempt(self):
        def invalid(response, request):
            return set_answer(response, lambda answer: answer['claims'][0].update(evidence_view_ids=['X']))
        self.assertEqual(self.run_pass(FakeClient(invalid))['status'], 'completed_with_failures')
        for number in (1, 2):
            attempt = self.output / f'calls/frame-0000/attempt-{number:04d}'
            self.assertTrue((attempt / 'response.json').is_file())
            self.assertIn('Invalid MedGemma answer', read(attempt / 'failure.json')['message'])
            self.assertFalse((attempt / 'receipt.json').exists())

    def test_legacy_review_plan_rejected_without_inference(self):
        self.output.mkdir()
        write(self.output / 'run.json', {'schema_version': 'medgemma-surgery-review-v1'})
        with self.assertRaisesRegex(ContractError, 'Expected an independent MedGemma annotation run'):
            self.run_pass(resume=True)

    def test_v1_run_cannot_resume_under_v2(self):
        self.output.mkdir()
        write(self.output / 'run.json', {'schema_version': 'medgemma-frame-annotation-v1', 'runtime': 'llama.cpp'})
        client = FakeClient()
        with self.assertRaisesRegex(ContractError, 'new output directory'):
            self.run_pass(client, resume=True)
        self.assertEqual(client.requests, [])

    def test_status_still_reports_v1_runs_but_pause_needs_v2(self):
        self.output.mkdir()
        write(self.output / 'run.json', {'schema_version': 'medgemma-frame-annotation-v1'})
        write(self.output / 'summary.json', {'status': 'failed', 'annotated_frame_count': 7})
        self.assertEqual(annotation_status(self.output)['annotated_frame_count'], 7)
        with self.assertRaisesRegex(ContractError, 'new output directory'):
            request_annotation_pause(self.output)
        self.assertFalse((self.output / '.pause-requested').exists())

    def test_context_budget_failure_stops_the_batch(self):
        def overflow(response, request):
            response['usage']['prompt_tokens'] = self.config.num_ctx
            return response
        client = FakeClient(overflow)
        with self.assertRaisesRegex(ContractError, 'answer capacity'):
            self.run_pass(client)
        self.assertEqual(len(client.requests), 1)

    def test_request_is_one_user_turn_with_labelled_views_and_no_frame_ids(self):
        client = FakeClient()
        self.run_pass(client)
        for request in client.requests:
            self.assertEqual([m['role'] for m in request['messages']], ['system', 'user'])
            text = json.dumps([p for p in request['messages'][1]['content'] if p['type'] == 'text'])
            for frame_id in self.ids:
                self.assertNotIn(frame_id, text)
        self.assertEqual(image_labels(client.requests[0]), ['T', 'T1', 'T2', 'T3', 'T4', 'A1'])
        self.assertEqual(image_labels(client.requests[1]), ['T', 'T1', 'T2', 'T3', 'T4', 'B1', 'A1'])

    def test_stored_annotation_uses_full_view_ids_and_software_claim_ids(self):
        self.run_pass()
        row = read(self.output / 'annotations.json')['annotations'][0]
        self.assertEqual(row['annotation']['target_frame_id'], self.ids[0])
        self.assertEqual(row['annotation']['claims'][0]['claim_id'], 'c1')
        self.assertEqual(row['annotation']['claims'][0]['evidence_view_ids'], [f'{self.ids[0]}:full'])
        self.assertEqual(row['annotation']['schema_version'], 'medgemma-frame-annotation-v2')

    def test_rejected_greedy_answer_gets_one_fallback_with_gemma_sampling(self):
        client = FakeClient(loop_when_greedy)
        self.assertEqual(self.run_pass(client)['status'], 'completed')
        self.assertEqual([r['temperature'] for r in client.requests], [0, 1.0, 0, 1.0])
        fallback = client.requests[1]
        self.assertEqual((fallback['top_k'], fallback['top_p'], fallback['min_p'], fallback['repeat_penalty']),
                         (64, 0.95, 0.0, 1.0))
        self.assertEqual(fallback['messages'], client.requests[0]['messages'])
        rows = read(self.output / 'annotations.json')['annotations']
        self.assertEqual([r['decoding_stage'] for r in rows], ['fallback', 'fallback'])
        self.assertIn('Degenerate', read(self.output / 'calls/frame-0000/attempt-0001/failure.json')['message'])
        self.assertTrue((self.output / 'calls/frame-0000/attempt-0002/receipt.json').is_file())

    def test_target_failing_both_attempts_is_recorded_and_the_batch_continues(self):
        def first_target_loops(response, request):
            return set_answer(response, counting_loop) if len(client.requests) <= 2 else response
        client = FakeClient(first_target_loops)
        summary = self.run_pass(client)
        self.assertEqual(summary['status'], 'completed_with_failures')
        self.assertEqual(summary['annotated_frame_count'], 1)
        self.assertEqual(summary['failed_frame_count'], 1)
        failure = summary['failed_targets'][0]
        self.assertEqual(failure['target_frame_id'], self.ids[0])
        self.assertEqual([a['decoding_stage'] for a in failure['attempts']], ['primary', 'fallback'])
        self.assertEqual([r['target_frame_id'] for r in read(self.output / 'annotations.json')['annotations']],
                         [self.ids[2]])
        self.assertEqual(read(self.output / 'calls/frame-0000/failed.json'), failure)
        self.assertEqual(self.run_pass(client, resume=True)['status'], 'completed_with_failures')
        self.assertEqual(len(client.requests), 3)

    def test_truncated_generation_is_a_rejected_answer(self):
        class Truncating(FakeClient):
            def chat_raw(self, request):
                raw = super().chat_raw(request)
                if request['temperature'] == 0:
                    response = json.loads(raw)
                    response['choices'][0]['finish_reason'] = 'length'
                    self.last_response_bytes = json.dumps(response).encode()
                    raise LlamaCppIncompleteError('llama.cpp response is unfinished or truncated')
                return raw
        client = Truncating()
        self.assertEqual(self.run_pass(client)['status'], 'completed')
        attempt = self.output / 'calls/frame-0000/attempt-0001'
        self.assertEqual(json.loads((attempt / 'response.json').read_bytes())['choices'][0]['finish_reason'], 'length')
        self.assertEqual(len(client.requests), 4)

    def test_runtime_failure_stops_the_batch_and_resume_retries_the_same_stage(self):
        class Unreachable(FakeClient):
            def chat_raw(self, request):
                self.requests.append(copy.deepcopy(request))
                raise LlamaCppError('Cannot reach owned llama.cpp server')
        with self.assertRaisesRegex(LlamaCppError, 'Cannot reach'):
            self.run_pass(Unreachable())
        self.assertFalse((self.output / 'calls/frame-0000/attempt-0001/response.json').exists())
        client = FakeClient()
        self.assertEqual(self.run_pass(client, resume=True)['status'], 'completed')
        self.assertEqual(client.requests[0]['temperature'], 0)
        self.assertTrue((self.output / 'calls/frame-0000/attempt-0002/receipt.json').is_file())

    def test_fallback_can_be_disabled(self):
        self.config = replace(self.config, fallback_sampling=False)
        client = FakeClient(lambda response, request: set_answer(response, counting_loop))
        summary = self.run_pass(client)
        self.assertEqual(summary['failed_frame_count'], 2)
        self.assertEqual(len(client.requests), 2)

    def test_prepare_refuses_an_output_budget_below_the_largest_permitted_answer(self):
        self.config = replace(self.config, num_predict=4096)
        client = FakeClient()
        with self.assertRaisesRegex(ContractError, 'output budget'):
            self.run_pass(client)
        self.assertFalse(self.output.exists())
        self.assertEqual(client.requests, [])

    def test_quality_flags_reach_the_published_row_and_summary(self):
        def repeated_citation(response, request):
            return set_answer(response, lambda answer: answer['claims'][0].update(evidence_view_ids=['T', 'T']))
        summary = self.run_pass(FakeClient(repeated_citation))
        rows = read(self.output / 'annotations.json')['annotations']
        self.assertEqual([r['quality_flags'] for r in rows], [['duplicate_citations_removed']] * 2)
        self.assertEqual(summary['quality_flag_counts'], {'duplicate_citations_removed': 2})

    def test_relocated_source_dir_annotates_from_verified_copy(self):
        copy_dir = self.root / 'copy/frames/S1A1'
        shutil.move(self.release, copy_dir)
        with self.assertRaisesRegex(ContractError, 'Evidence changed or is missing'):
            self.run_pass(prepare_only=True)
        self.assertEqual(self.run_pass(prepare_only=True, source_dir=copy_dir)['status'], 'prepared')
        self.assertEqual(read(self.output / 'source.json'), self.source)
        self.assertEqual(read(self.output / 'run.json')['source_relocation'],
                         {'from': str(self.release), 'to': str(copy_dir)})
        for frame in read(self.output / 'evidence/frame-0000.json')['frames']:
            self.assertEqual(Path(frame['source_path']).parent, copy_dir)
        client = FakeClient()
        self.assertEqual(self.run_pass(client, resume=True)['status'], 'completed')
        self.assertEqual(len(client.requests), 2)

    def test_relocated_source_dir_must_hold_identical_bytes(self):
        copy_dir = self.root / 'copy/frames/S1A1'
        shutil.copytree(self.release, copy_dir)
        Image.new('RGB', (48, 32), 'white').save(copy_dir / 'S1A1_frame_00000002.jpeg')
        with self.assertRaisesRegex(ContractError, 'Evidence changed or is missing'):
            self.run_pass(prepare_only=True, source_dir=copy_dir)
        self.assertFalse(self.output.exists())

    def test_source_dir_applies_only_when_preparing(self):
        self.run_pass(prepare_only=True)
        with self.assertRaisesRegex(ContractError, 'only when preparing'):
            self.run_pass(resume=True, source_dir=self.release)

    def test_zero_context_no_crops_ablation(self):
        self.config = replace(self.config, before_frames=0, after_frames=0, detail_crops=False)
        client = FakeClient()
        self.run_pass(client)
        self.assertEqual([image_labels(req) for req in client.requests], [['T'], ['T']])


if __name__ == '__main__':
    unittest.main()

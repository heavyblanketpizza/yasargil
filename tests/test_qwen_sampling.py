"""Regressions for Qwen's explicit model-card sampling request profile."""
import json
import unittest

from yasargil.qwen_sampling import (
    qwen_non_thinking_parameters, qwen_sampling_receipt, verify_qwen_sampling,
)


SAMPLER_LOG = """0.52.312.504 I slot launch_slot_: id  0 | task -1 | sampler chain: logits -> penalties -> temp-ext -> top-k -> top-p -> ?min-p -> dist
0.52.312.511 I slot launch_slot_: id  0 | task -1 | sampler params:
\trepeat_last_n = 12288, repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 1.500
\tdry_multiplier = 0.000, dry_base = 1.750, dry_allowed_length = 2, dry_penalty_last_n = 64
\ttop_k = 20, top_p = 0.800, min_p = 0.000, xtc_probability = 0.000, xtc_threshold = 0.100, typical_p = 1.000, top_n_sigma = -1.000, temp = 0.700
\tmirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000, adaptive_target = -1.000, adaptive_decay = 0.900
\tsamplers_generated_only = 1, sampler_history_scope = generated
0.52.312.513 I slot launch_slot_: id  0 | task 0 | processing task, is_child = 0
"""


class QwenSamplingTests(unittest.TestCase):
    def test_official_non_thinking_values_use_llama_api_field_names(self):
        parameters = qwen_non_thinking_parameters()
        # Keep a literal expected contract: accidentally restoring the previous
        # temperature or inheriting top-p/min-p defaults changes generation.
        self.assertEqual(parameters, {
            "temperature": 0.7, "top_p": 0.8, "top_k": 20,
            "min_p": 0.0, "presence_penalty": 1.5, "repeat_penalty": 1.0,
            "frequency_penalty": 0.0, "seed": 42,
        })
        self.assertNotIn("repetition_penalty", parameters)
        self.assertEqual(json.loads(json.dumps(parameters, allow_nan=False)), parameters)

    def test_one_request_cannot_mutate_future_requests_or_provenance(self):
        first = qwen_non_thinking_parameters()
        receipt = qwen_sampling_receipt()
        first["temperature"] = 0.1
        receipt["requested_parameters"]["presence_penalty"] = 0.0
        receipt["local_choices"].append("temperature")
        fresh = qwen_non_thinking_parameters()
        fresh_receipt = qwen_sampling_receipt()
        self.assertEqual(fresh["temperature"], 0.7)
        self.assertEqual(fresh_receipt["requested_parameters"], fresh)
        self.assertEqual(fresh_receipt["local_choices"], ["frequency_penalty", "seed"])
        self.assertFalse(fresh_receipt["backend_equivalence_verified"])

    def test_saved_effective_snapshot_verifies_generation_controls(self):
        receipt = verify_qwen_sampling(SAMPLER_LOG, 12288)
        self.assertTrue(receipt["verified"])
        self.assertEqual(receipt["observed_parameters"]["repeat_last_n"], 12288)
        self.assertEqual(receipt["observed_parameters"]["sampler_history_scope"], "generated")
        self.assertEqual(receipt["unobserved_requested_fields"], ["seed"])
        self.assertFalse(receipt["backend_equivalence_verified"])
        self.assertEqual(json.loads(json.dumps(receipt, allow_nan=False)), receipt)
        # The pinned sampler may report an inactive min-p filter for p=0.
        self.assertTrue(verify_qwen_sampling(SAMPLER_LOG.replace("?min-p", "min-p"), 12288)["verified"])

    def test_wrong_values_or_missing_fields_cannot_pass_on_the_request_alone(self):
        wrong_values = {
            "repeat_last_n = 12288": "repeat_last_n = 64",
            "repeat_penalty = 1.000": "repeat_penalty = 1.100",
            "frequency_penalty = 0.000": "frequency_penalty = 0.100",
            "presence_penalty = 1.500": "presence_penalty = 0.000",
            "top_k = 20": "top_k = 40",
            "top_p = 0.800": "top_p = 0.950",
            "min_p = 0.000": "min_p = 0.050",
            "temp = 0.700": "temp = 0.100",
            "samplers_generated_only = 1": "samplers_generated_only = 0",
            "sampler_history_scope = generated": "sampler_history_scope = prompt_and_generated",
        }
        for field, wrong in wrong_values.items():
            for replacement in (wrong, ""):
                with self.subTest(field=field, replacement=replacement), self.assertRaises(ValueError):
                    verify_qwen_sampling(SAMPLER_LOG.replace(field, replacement), 12288)
        for invalid in ("nan", "inf", "not-a-number"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                verify_qwen_sampling(SAMPLER_LOG.replace("temp = 0.700", f"temp = {invalid}"), 12288)

    def test_wrong_order_inactive_required_sampler_or_extra_sampler_rejected(self):
        for before, after in (("penalties", "?penalties"), ("temp-ext", "?temp-ext"),
                              ("temp-ext -> top-k", "top-k -> temp-ext"),
                              ("top-p -> ?min-p", "top-p -> dry -> ?min-p")):
            with self.subTest(after=after), self.assertRaisesRegex(ValueError, "sampler order"):
                verify_qwen_sampling(SAMPLER_LOG.replace(before, after), 12288)

    def test_ambiguous_or_incomplete_logs_rejected(self):
        cases = ["", SAMPLER_LOG + SAMPLER_LOG,
                 SAMPLER_LOG.replace("sampler chain:", "chain:"),
                 SAMPLER_LOG.replace("sampler params:", "params:"),
                 SAMPLER_LOG.replace("top_p = 0.800", "top_p = 0.800, top_p = 0.800")]
        for segment in cases:
            with self.subTest(segment=segment[:100]), self.assertRaises(ValueError):
                verify_qwen_sampling(segment, 12288)
        # A later unrelated line cannot repair a missing effective setting.
        missing = SAMPLER_LOG.replace("presence_penalty = 1.500", "")
        with self.assertRaisesRegex(ValueError, "Missing effective sampler field"):
            verify_qwen_sampling(missing + "request presence_penalty = 1.500\n", 12288)

    def test_invalid_output_budget_and_input_type_rejected(self):
        for budget in (0, -1, True, 12288.0, 2**31):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                verify_qwen_sampling(SAMPLER_LOG, budget)
        with self.assertRaises(ValueError):
            verify_qwen_sampling(None, 12288)


if __name__ == "__main__":
    unittest.main()

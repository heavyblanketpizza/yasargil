"""CPU-only contract tests. They do not test a real tokenizer, VLM, or GPU."""

from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from yasargil import training as recipe

RECIPE_PATH = Path(recipe.__file__).resolve()


class FakeBaseCollator:
    def __init__(self, model, processor, train_on_responses_only=False,
                 last_response_only=False, **kwargs):
        self.responses = train_on_responses_only
        self.last = last_response_only
        self.max_seq_length = kwargs["max_seq_length"]
        self.truncation = True
        self.calls = 0

    def __call__(self, examples):
        self.calls += 1
        raise AssertionError("This fake must never execute real collation.")


class FakePublicWrapper(FakeBaseCollator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class OldCollatorWithoutFinalScope:
    def __init__(self, model, processor, train_on_responses_only=False, **kwargs):
        raise AssertionError("Unsupported collator must fail before construction.")


class RecipeContractTests(unittest.TestCase):
    def make(self, cls=FakePublicWrapper, scope="final_assistant_turn", **changes):
        args = dict(instruction_part="<user>\n", response_part="<assistant>\n",
                    loss_scope=scope, max_seq_length=4096, resize="max")
        args.update(changes)
        return recipe._make_collator(cls, object(), object(), **args)

    def test_import_has_no_ml_side_effects(self):
        # A fresh process prevents other tests' imports from hiding a regression.
        code = (
            "import importlib.util,sys;"
            "s=importlib.util.spec_from_file_location('probe',sys.argv[1]);"
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
            "assert not any(n in sys.modules for n in "
            "('torch','transformers','trl','unsloth','datasets','PIL'))"
        )
        subprocess.run([sys.executable, "-B", "-I", "-c", code, str(RECIPE_PATH.resolve())],
                       check=True, capture_output=True, text=True)

    def test_final_scope_survives_kwargs_wrapper(self):
        collator = self.make()
        self.assertTrue(collator.inner.responses)
        self.assertTrue(collator.inner.last)

    def test_all_assistant_scope_is_explicit(self):
        collator = self.make(scope="all_assistant_turns")
        self.assertTrue(collator.inner.responses)
        self.assertFalse(collator.inner.last)

    def test_old_api_never_silently_falls_back(self):
        with self.assertRaisesRegex(RuntimeError, "last_response_only"):
            self.make(cls=OldCollatorWithoutFinalScope)

    def test_model_inherited_truncation_is_disabled(self):
        collator = self.make()
        self.assertEqual(collator.max_seq_length, 4096)
        self.assertIsNone(collator.inner.max_seq_length)
        self.assertFalse(collator.inner.truncation)

    def test_reenabled_truncation_fails_before_collation(self):
        collator = self.make()
        collator.inner.truncation = True
        with self.assertRaisesRegex(RuntimeError, "re-enabled"):
            collator([{"messages": []}])
        self.assertEqual(collator.inner.calls, 0)

    def test_markers_must_be_explicit_and_distinct(self):
        for changed in ({"instruction_part": ""}, {"response_part": ""},
                        {"response_part": "<user>\n"}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.make(**changed)

    def test_invalid_build_inputs_fail_before_ml_environment_check(self):
        with patch.object(recipe, "check_runtime_versions") as versions:
            for rows, cap, scope in (([], 4096, "final_assistant_turn"),
                                     ([{}], 0, "final_assistant_turn"),
                                     ([{}], True, "final_assistant_turn"),
                                     ([{}], 4096, "unknown")):
                with self.subTest(cap=cap, scope=scope), self.assertRaises(ValueError):
                    recipe.build_sft_trainer(
                        object(), object(), rows, output_dir="unused",
                        instruction_part="user", response_part="assistant",
                        max_seq_length=cap, loss_scope=scope,
                    )
            versions.assert_not_called()


if __name__ == "__main__":
    unittest.main()

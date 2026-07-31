import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "train_sft.py"
SPEC = importlib.util.spec_from_file_location("train_sft", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TrainingConfigTest(unittest.TestCase):
    def test_resolves_deepspeed_auto_batch_values_before_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ds.json"
            path.write_text(
                json.dumps(
                    {
                        "train_batch_size": "auto",
                        "train_micro_batch_size_per_gpu": "auto",
                        "gradient_accumulation_steps": "auto",
                        "zero_optimization": {"stage": 3},
                    }
                ),
                encoding="utf-8",
            )
            config = MODULE.resolve_deepspeed_config(
                path, batch_size=1, gradient_accumulation=4, world_size=4
            )
        self.assertEqual(config["train_micro_batch_size_per_gpu"], 1)
        self.assertEqual(config["gradient_accumulation_steps"], 4)
        self.assertEqual(config["train_batch_size"], 16)

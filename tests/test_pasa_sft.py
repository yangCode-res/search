import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_pasa_sft.py"
SPEC = importlib.util.spec_from_file_location("prepare_pasa_sft", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PasaSFTTest(unittest.TestCase):
    def test_parse_selector_record(self):
        record = {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Searched Paper:\nTitle: Relevant title\nAbstract: Relevant abstract.\n\n"
                        "User Query: papers about academic search agents\n\n"
                        "Output format: Decision: True/False\nReason:...\nDecision:"
                    ),
                },
                {"role": "assistant", "content": "True\nReason: directly studies the query"},
            ]
        }
        parsed = MODULE.parse_selector_record(record)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["query"], "papers about academic search agents")
        self.assertEqual(parsed["candidate"]["label"], "SELECT")
        self.assertEqual(parsed["candidate"]["title"], "Relevant title")

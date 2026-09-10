"""Offline validation for multi-GPU dispatch configuration parsing."""

import argparse
from pathlib import Path
import unittest

from kv_semantic_attack.run_multi_gpu_defense import Job, parse_gpu_slots


class MultiGpuDispatchTests(unittest.TestCase):
    def test_parse_gpu_slots(self):
        self.assertEqual(parse_gpu_slots(["0=2", "2=1", "3=4"]), {0: 2, 2: 1, 3: 4})
        for value in ([], ["0=0"], ["0=-1"], ["gpu=2"], ["0=2", "0=1"]):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    parse_gpu_slots(value)

    def test_job_is_immutable(self):
        job = Job("full", "full", None, __import__("pathlib").Path("full.json"))
        with self.assertRaises(AttributeError):
            job.name = "reuse"

    def test_event_driven_dispatch_has_no_poll_interval(self):
        source = Path(__file__).resolve().parents[1] / "run_multi_gpu_defense.py"
        text = source.read_text(encoding="utf-8")
        self.assertIn("os.wait()", text)
        self.assertNotIn("poll-seconds", text)


if __name__ == "__main__":
    unittest.main()
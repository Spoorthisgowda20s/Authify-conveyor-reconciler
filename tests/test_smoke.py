"""Smoke test: generate 90 s, track, score; count must be near truth and nothing counted in a stall.
    python3 -m unittest discover tests
"""
import os, shutil, tempfile, unittest
from conveyor import evaluate, generate, track


class Smoke(unittest.TestCase):
    def test_pipeline(self):
        d = tempfile.mkdtemp()
        try:
            generate.main(["--out", d, "--seed", 7, "--duration", "90"] if False else ["--out", d, "--seed", "7", "--duration", "90"])
            track.run(d)
            r = evaluate.evaluate(d)
            self.assertLess(abs(r["total"]["error_pct"]), 5)
            self.assertEqual(r["stall"]["count_events_during_stall"], 0)
        finally:
            shutil.rmtree(d)


if __name__ == "__main__":
    unittest.main()

import sys
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training_config import get_training_config


class TrainingConfigTests(unittest.TestCase):
    def test_faster_convergence_defaults(self):
        config = get_training_config()

        self.assertGreater(config["alpha"], 0.25)
        self.assertLess(config["epsilon_decay"], 0.995)
        self.assertGreaterEqual(config["epsilon_min"], 0.03)
        self.assertLess(config["temperature"], 0.9)
        self.assertGreater(config["distance_reward_factor"], 0.7)


if __name__ == "__main__":
    unittest.main()

import unittest
import numpy as np

from kociu_shortcuts.data import Standardizer, make_dataset
from kociu_shortcuts.detector import paired_loso_predictions
from kociu_shortcuts.metrics import spectral_concentration, task_structured_separability

class CoreTests(unittest.TestCase):
    def test_linear_and_matched_control(self):
        exposed = make_dataset(10000, "linear", 19)
        control = make_dataset(10000, "matched_control", 19)
        self.assertGreater(abs(np.corrcoef(exposed.X[:, 2], exposed.y)[0, 1]), 0.95)
        self.assertLess(abs(np.corrcoef(control.X[:, 2], control.y)[0, 1]), 0.05)
        self.assertLess(abs(exposed.X[:, 2].std() - control.X[:, 2].std()), 0.02)

    def test_standardizer_and_metrics(self):
        data = make_dataset(600, "noise", 4)
        transformed = Standardizer.fit(data.X).transform(data.X)
        np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-6)
        labels = np.repeat([0, 1], 200)
        rng = np.random.RandomState(0)
        separated = rng.randn(400, 8) * 0.2
        separated[labels == 0] -= 5.0
        separated[labels == 1] += 5.0
        self.assertGreater(task_structured_separability(separated, labels), 0.7)
        value = spectral_concentration(rng.randn(500, 12))
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_pairwise_loso(self):
        pairs = np.repeat(np.arange(10), 2)
        labels = np.tile([1, 0], 10)
        profiles = np.column_stack([labels * 2.0 - 1.0, np.zeros(20), np.ones(20)])
        result = paired_loso_predictions(profiles, labels, pairs)
        self.assertEqual(result.accuracy, 1.0)
        self.assertEqual(result.complete_pair_rate, 1.0)

if __name__ == "__main__":
    unittest.main()

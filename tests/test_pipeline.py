"""Smoke test: train a tiny model end-to-end on synthetic data."""
import unittest

import numpy as np
import pandas as pd

from src.preprocess import engineer_features
from src.train import build_pipelines
from tests.test_preprocess import make_df


class TestPipeline(unittest.TestCase):
    def test_all_pipelines_fit_and_predict(self):
        df = engineer_features(make_df(n=80))
        X, y = df.drop(columns=["price_range"]), df["price_range"]
        for name, pipe in build_pipelines(seed=0).items():
            pipe.fit(X, y)
            preds = pipe.predict(X)
            self.assertEqual(len(preds), len(X), msg=name)
            self.assertTrue(set(np.unique(preds)) <= {0, 1, 2, 3}, msg=name)


if __name__ == "__main__":
    unittest.main()

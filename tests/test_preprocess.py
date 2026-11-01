import unittest

import numpy as np
import pandas as pd

from src.preprocess import RAW_FEATURES, engineer_features, validate


def make_df(n=20, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.integers(1, 100, n) for c in RAW_FEATURES})
    df["price_range"] = rng.integers(0, 4, n)
    return df


class TestPreprocess(unittest.TestCase):
    def test_validate_passes_on_valid_data(self):
        df = make_df()
        self.assertEqual(len(validate(df, "price_range")), len(df))

    def test_validate_raises_on_missing_column(self):
        with self.assertRaises(ValueError):
            validate(make_df().drop(columns=["ram"]), "price_range")

    def test_validate_fills_missing_values(self):
        df = make_df().astype(float)
        df.loc[0, "ram"] = np.nan
        self.assertEqual(validate(df, "price_range").isna().sum().sum(), 0)

    def test_validate_drops_duplicates(self):
        df = make_df()
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        self.assertEqual(len(validate(df, "price_range")), len(df) - 1)

    def test_engineered_features(self):
        out = engineer_features(make_df())
        for col in ["px_area", "screen_area", "px_density", "camera_total", "ram_per_core"]:
            self.assertIn(col, out.columns)
        self.assertTrue(np.isfinite(out.select_dtypes("number")).all().all())

    def test_engineer_handles_zero_screen_width(self):
        df = make_df()
        df.loc[0, ["sc_w", "n_cores"]] = 0
        out = engineer_features(df)
        self.assertTrue(np.isfinite(out.loc[0, ["px_density", "ram_per_core"]]).all())

    def test_engineer_does_not_mutate_input(self):
        df = make_df()
        cols = list(df.columns)
        engineer_features(df)
        self.assertEqual(list(df.columns), cols)


if __name__ == "__main__":
    unittest.main()

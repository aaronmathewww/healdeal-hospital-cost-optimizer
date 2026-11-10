import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ml.ces_engine import CESEngine, haversine_km, min_max_norm
from ml.cost_model import predict_cost

ROOT = Path(__file__).resolve().parents[1]


class SyntheticDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(ROOT / "data" / "hospitals.csv")

    def test_dataset_is_explicitly_synthetic_and_complete(self):
        self.assertEqual(len(self.data), 612)
        self.assertTrue(self.data["name"].str.startswith("Demo Hospital H").all())
        self.assertTrue(self.data["hospital_id"].is_unique)
        self.assertEqual(self.data["city"].nunique(), 20)
        self.assertFalse(self.data.isna().any().any())
        self.assertTrue(self.data["avg_cost_for_procedure"].ge(0).all())
        self.assertTrue(self.data["avg_rating"].between(1, 5).all())
        self.assertTrue(self.data["latitude"].between(-90, 90).all())
        self.assertTrue(self.data["longitude"].between(-180, 180).all())
        self.assertTrue(self.data["insurance_accepted"].str.contains("Sample Insurance").all())

    def test_model_predictions_are_nonnegative_and_finite(self):
        predictions = predict_cost(self.data.head(25))
        self.assertEqual(len(predictions), 25)
        self.assertTrue(np.isfinite(predictions).all())
        self.assertTrue((predictions >= 0).all())


class RankingTests(unittest.TestCase):
    def test_haversine_distance(self):
        distance = haversine_km(0, 0, 0, 1)
        self.assertAlmostEqual(distance, 111.19, delta=0.5)
        self.assertTrue(np.isnan(haversine_km(91, 0, 0, 0)))

    def test_min_max_normalization_handles_missing_values(self):
        result = min_max_norm(pd.Series([np.nan, np.nan]))
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(result.between(0, 1).all())

    def test_ces_orders_lower_cost_first_when_cost_is_only_weight(self):
        frame = pd.DataFrame({
            "name": ["higher", "lower"],
            "avg_cost_for_procedure": [200.0, 100.0],
            "avg_rating": [4.0, 4.0],
            "latitude": [0.0, 0.0],
            "longitude": [0.0, 0.0],
            "sentiment_score": [0.5, 0.5],
        })
        result = CESEngine({"cost": 1, "rating": 0, "distance": 0, "sentiment": 0}).score(
            frame, 0.0, 0.0
        )
        self.assertEqual(result.iloc[0]["name"], "lower")
        self.assertTrue(result["CES"].is_monotonic_increasing)

    def test_invalid_weights_fall_back_safely(self):
        engine = CESEngine({"cost": -1, "rating": -1, "distance": 0, "sentiment": 0})
        self.assertAlmostEqual(sum(engine.weights.values()), 1.0)
        self.assertTrue(all(value >= 0 for value in engine.weights.values()))


if __name__ == "__main__":
    unittest.main()

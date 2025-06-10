import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import Config
from src.data_preprocessing import DataPreprocessor


class TestDataPreprocessing(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.preprocessor = DataPreprocessor(self.config)

    def test_generate_synthetic_data(self):
        """Test synthetic data generation"""
        data = self.preprocessor.generate_synthetic_data(
            start_date="2023-01-01", end_date="2023-12-31"
        )

        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)
        self.assertIn("demand", data.columns)
        self.assertIn("date", data.columns)
        self.assertTrue(data["demand"].min() >= 0)

    def test_clean_data(self):
        """Test data cleaning functionality"""
        # Create test data with issues
        test_data = pd.DataFrame(
            {
                "demand": [100, 150, np.nan, 200, 1000],  # Has NaN and outlier
                "price": [50, 55, 52, np.nan, 48],
                "date": pd.date_range("2023-01-01", periods=5),
            }
        )

        cleaned_data = self.preprocessor.clean_data(test_data)

        # Should have no NaN values
        self.assertFalse(cleaned_data.isnull().any().any())

    def test_create_features(self):
        """Test feature creation"""
        base_data = self.preprocessor.generate_synthetic_data(
            start_date="2023-01-01", end_date="2023-03-31"
        )

        feature_data = self.preprocessor.create_features(base_data)

        # Check if lag features are created
        lag_features = [col for col in feature_data.columns if "lag" in col]
        self.assertGreater(len(lag_features), 0)

        # Check if rolling features are created
        rolling_features = [col for col in feature_data.columns if "rolling" in col]
        self.assertGreater(len(rolling_features), 0)


class TestTimeSeriesModels(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.preprocessor = DataPreprocessor(self.config)

        # Generate test data
        data = self.preprocessor.generate_synthetic_data(
            start_date="2023-01-01", end_date="2023-06-30"
        )
        self.test_data = self.preprocessor.prepare_for_modeling(data)

    def test_prophet_forecast(self):
        """Test Prophet model functionality"""
        from src.time_series_models import TimeSeriesModels

        ts_models = TimeSeriesModels(self.config)

        try:
            model, forecast = ts_models.prophet_forecast(
                self.test_data["data"], periods=30
            )

            self.assertIsNotNone(model)
            self.assertIsNotNone(forecast)
            self.assertGreater(len(forecast), 0)

        except ImportError:
            self.skipTest("Prophet not available")
        except Exception as e:
            self.fail(f"Prophet test failed: {str(e)}")


if __name__ == "__main__":
    unittest.main()

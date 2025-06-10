"""
Configuration settings for Supply Chain Analysis
Python 3.12 Compatible Version
"""

import os
from datetime import datetime, timedelta


class Config:
    def __init__(self):
        # Data settings
        self.DATA_PATH = "data/"
        self.RAW_DATA_PATH = "data/raw/"
        self.PROCESSED_DATA_PATH = "data/processed/"
        self.MODELS_PATH = "models/"
        self.RESULTS_PATH = "results/"

        # Analysis parameters
        self.RANDOM_SEED = 42
        self.FORECAST_HORIZON = 30
        self.CONFIDENCE_LEVEL = 0.95
        self.N_CROSS_VALIDATION_FOLDS = 5

        # Time series parameters
        self.LAG_FEATURES = [1, 2, 3, 7, 14, 30]
        self.ROLLING_WINDOWS = [7, 14, 30]

        # Optimization parameters
        self.HOLDING_COST_RATE = 0.25
        self.ORDER_COST = 100
        self.STOCKOUT_COST = 50
        self.TARGET_SERVICE_LEVEL = 0.95

        # Model parameters
        self.XGBOOST_PARAMS = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": self.RANDOM_SEED,
        }

        self.LIGHTGBM_PARAMS = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": self.RANDOM_SEED,
            "verbose": -1,
        }

        # Create directories
        for path in [
            self.DATA_PATH,
            self.RAW_DATA_PATH,
            self.PROCESSED_DATA_PATH,
            self.MODELS_PATH,
            self.RESULTS_PATH,
        ]:
            os.makedirs(path, exist_ok=True)

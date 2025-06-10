import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")


class DataPreprocessor:
    def __init__(self, config):
        self.config = config

    def generate_synthetic_data(self, start_date="2020-01-01", end_date="2024-12-31"):
        """Generate synthetic supply chain data for demonstration"""
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        n_days = len(date_range)

        # Set random seed for reproducibility
        np.random.seed(42)

        # Generate base demand with trend and seasonality
        trend = np.linspace(100, 150, n_days)
        seasonal_weekly = 10 * np.sin(2 * np.pi * np.arange(n_days) / 7)
        seasonal_yearly = 20 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
        noise = np.random.normal(0, 5, n_days)
        base_demand = trend + seasonal_weekly + seasonal_yearly + noise

        # Generate causal variables
        data = pd.DataFrame(
            {
                "date": date_range,
                "base_demand": base_demand,
                "price": np.random.normal(50, 5, n_days),
                "promotional_activity": np.random.binomial(1, 0.1, n_days),
                "weather_score": np.random.normal(0, 1, n_days),
                "competitor_price": np.random.normal(52, 6, n_days),
                "economic_indicator": np.random.normal(100, 10, n_days),
                "supplier_reliability": np.random.beta(8, 2, n_days),
            }
        )

        # Create causal effects
        demand_shock = np.zeros(n_days)
        price_effect = -0.5 * (data["price"] - 50)  # Price elasticity
        promo_effect = data["promotional_activity"] * np.random.normal(15, 3, n_days)
        weather_effect = data["weather_score"] * 2
        competitor_effect = -0.3 * (data["competitor_price"] - 52)
        economic_effect = 0.1 * (data["economic_indicator"] - 100)

        # Add random demand shocks
        shock_days = np.random.choice(n_days, size=int(n_days * 0.05), replace=False)
        demand_shock[shock_days] = np.random.normal(0, 20, len(shock_days))

        # Calculate actual demand
        data["demand"] = (
            base_demand
            + price_effect
            + promo_effect
            + weather_effect
            + competitor_effect
            + economic_effect
            + demand_shock
        ).clip(lower=0)

        # Generate supply chain metrics
        data["lead_time"] = np.maximum(
            1, np.random.poisson(7, n_days) * (1 - data["supplier_reliability"] * 0.3)
        )

        data["inventory_level"] = self._simulate_inventory(
            data["demand"], data["lead_time"]
        )
        data["stockout_rate"] = (data["inventory_level"] <= 0).astype(int)

        # Calculate costs
        data["holding_cost"] = data["inventory_level"] * 0.1
        data["stockout_cost"] = data["stockout_rate"] * data["demand"] * 10
        data["total_cost"] = data["holding_cost"] + data["stockout_cost"]

        # Service level
        data["service_level"] = 1 - data["stockout_rate"]

        # Add derived variables
        data["demand_shock"] = demand_shock
        data["price_change"] = data["price"].diff()
        data["day_of_week"] = data["date"].dt.dayofweek
        data["month"] = data["date"].dt.month
        data["quarter"] = data["date"].dt.quarter

        return data

    def _simulate_inventory(self, demand, lead_time, initial_inventory=200):
        """Simulate inventory levels based on demand and lead time"""
        inventory = [initial_inventory]
        order_quantity = 100  # Fixed order quantity

        for i in range(1, len(demand)):
            # Simple (s,S) inventory policy
            current_inv = inventory[-1] - demand.iloc[i - 1]

            # Reorder if inventory falls below reorder point
            reorder_point = np.mean(demand[:i]) * lead_time.iloc[i] if i > 7 else 50

            if current_inv <= reorder_point:
                current_inv += order_quantity

            inventory.append(max(0, current_inv))

        return inventory

    def clean_data(self, data):
        """Clean and validate the dataset"""
        # Handle missing values
        data = data.fillna(method="forward").fillna(method="backward")

        # Remove outliers using IQR method
        for col in ["demand", "price", "inventory_level"]:
            if col in data.columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)

        return data

    def create_features(self, data):
        """Create additional features for modeling"""
        # Lag features
        for lag in [1, 7, 30]:
            data[f"demand_lag_{lag}"] = data["demand"].shift(lag)
            data[f"price_lag_{lag}"] = data["price"].shift(lag)

        # Rolling statistics
        for window in [7, 14, 30]:
            data[f"demand_rolling_mean_{window}"] = (
                data["demand"].rolling(window).mean()
            )
            data[f"demand_rolling_std_{window}"] = data["demand"].rolling(window).std()

        # Interaction features
        data["price_promo_interaction"] = data["price"] * data["promotional_activity"]
        data["weather_seasonal"] = data["weather_score"] * np.sin(
            2 * np.pi * data["day_of_week"] / 7
        )

        # Volatility measures
        data["demand_volatility"] = data["demand"].rolling(30).std()
        data["price_volatility"] = data["price"].rolling(30).std()

        return data

    def prepare_for_modeling(self, data, target_variable="demand"):
        """Prepare data for causal and time series modeling"""
        # Create time-based splits
        data = data.sort_values("date").reset_index(drop=True)

        # Remove rows with NaN values created by lag features
        data_clean = data.dropna()

        # Define feature groups
        causal_features = [
            "price",
            "promotional_activity",
            "weather_score",
            "competitor_price",
            "economic_indicator",
            "supplier_reliability",
        ]

        time_features = ["day_of_week", "month", "quarter"] + [
            col for col in data_clean.columns if "lag" in col or "rolling" in col
        ]

        outcome_features = [
            "inventory_level",
            "stockout_rate",
            "lead_time",
            "service_level",
            "total_cost",
        ]

        return {
            "data": data_clean,
            "causal_features": causal_features,
            "time_features": time_features,
            "outcome_features": outcome_features,
            "target": target_variable,
        }

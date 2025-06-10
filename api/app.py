"""
Simplified Flask API - No Complex Imports
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)


# Simple configuration
class SimpleConfig:
    def __init__(self):
        self.RANDOM_SEED = 42
        self.HOLDING_COST_RATE = 0.25
        self.ORDER_COST = 100
        self.STOCKOUT_COST = 50


config = SimpleConfig()
np.random.seed(config.RANDOM_SEED)


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "simplified_api",
        }
    )


@app.route("/predict/demand", methods=["POST"])
def predict_demand():
    """Predict demand using simple model"""
    try:
        # Check if request has JSON data
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()
        if data is None:
            return jsonify({"error": "No JSON data provided"}), 400

        if "features" not in data:
            return jsonify({"error": 'Missing "features" key'}), 400

        # Convert to DataFrame
        input_df = pd.DataFrame(data["features"])

        # Simple demand prediction model
        predictions = []
        for _, row in input_df.iterrows():
            base_demand = 100

            # Price effect (if price column exists)
            if "price" in row:
                price_effect = -2.0 * (row["price"] - 50)
            else:
                price_effect = 0

            # Promotional effect
            if "promotional_activity" in row:
                promo_effect = 20 * row["promotional_activity"]
            else:
                promo_effect = 0

            # Weather effect
            if "weather_score" in row:
                weather_effect = 5 * row["weather_score"]
            else:
                weather_effect = 0

            # Calculate final prediction
            prediction = base_demand + price_effect + promo_effect + weather_effect
            prediction = max(0, prediction)  # Ensure non-negative
            predictions.append(prediction)

        return jsonify(
            {
                "predictions": predictions,
                "model_type": "simple_linear",
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/optimize/inventory", methods=["POST"])
def optimize_inventory():
    """Simple inventory optimization"""
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()
        if data is None or "demand_forecast" not in data:
            return jsonify({"error": "Missing demand_forecast"}), 400

        demand_forecast = np.array(data["demand_forecast"])

        # Simple inventory optimization
        avg_demand = np.mean(demand_forecast)
        demand_std = np.std(demand_forecast)

        # Calculate optimal parameters using simple formulas
        optimal_order_quantity = np.sqrt(
            2 * config.ORDER_COST * avg_demand / config.HOLDING_COST_RATE
        )
        optimal_reorder_point = avg_demand * 7 + 1.96 * demand_std * np.sqrt(
            7
        )  # 7-day lead time
        safety_stock = 1.96 * demand_std * np.sqrt(7)

        # Simulate performance
        total_cost = (optimal_order_quantity / 2) * config.HOLDING_COST_RATE + (
            avg_demand / optimal_order_quantity
        ) * config.ORDER_COST

        service_level = 0.95  # Assumed based on safety stock calculation

        return jsonify(
            {
                "optimal_parameters": {
                    "reorder_point": float(optimal_reorder_point),
                    "order_quantity": float(optimal_order_quantity),
                    "safety_stock": float(safety_stock),
                },
                "optimal_cost": float(total_cost),
                "performance_metrics": {
                    "service_level": service_level,
                    "avg_inventory": float(optimal_order_quantity / 2 + safety_stock),
                },
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return jsonify({"error": f"Optimization failed: {str(e)}"}), 500


@app.route("/analyze/causal", methods=["POST"])
def analyze_causal():
    """Simple causal analysis"""
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()
        if data is None:
            return jsonify({"error": "No JSON data provided"}), 400

        required_fields = ["data", "treatment", "outcome"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing {field}"}), 400

        # Convert to DataFrame
        df = pd.DataFrame(data["data"])
        treatment = data["treatment"]
        outcome = data["outcome"]

        # Simple correlation analysis
        if treatment in df.columns and outcome in df.columns:
            correlation = df[treatment].corr(df[outcome])

            # Simple linear regression effect
            from sklearn.linear_model import LinearRegression

            X = df[[treatment]].values
            y = df[outcome].values

            # Remove NaN values
            mask = ~(pd.isna(X).any(axis=1) | pd.isna(y))
            X_clean = X[mask]
            y_clean = y[mask]

            if len(X_clean) > 1:
                model = LinearRegression()
                model.fit(X_clean, y_clean)
                effect_size = model.coef_[0]
                r_squared = model.score(X_clean, y_clean)
            else:
                effect_size = 0
                r_squared = 0

            return jsonify(
                {
                    "causal_analysis": {
                        "correlation": float(correlation)
                        if not pd.isna(correlation)
                        else 0,
                        "linear_effect": float(effect_size),
                        "r_squared": float(r_squared),
                        "significant": abs(correlation) > 0.1
                        if not pd.isna(correlation)
                        else False,
                    },
                    "treatment": treatment,
                    "outcome": outcome,
                    "timestamp": datetime.now().isoformat(),
                }
            )
        else:
            return jsonify(
                {"error": f"Columns {treatment} or {outcome} not found in data"}
            ), 400

    except Exception as e:
        return jsonify({"error": f"Causal analysis failed: {str(e)}"}), 500


@app.route("/demo/data", methods=["GET"])
def demo_data():
    """Generate demo data"""
    try:
        # Generate simple demo data
        dates = pd.date_range("2024-01-01", periods=30, freq="D")

        demo_data = []
        for i, date in enumerate(dates):
            demo_data.append(
                {
                    "date": date.isoformat(),
                    "demand": 100 + 20 * np.sin(i / 7) + np.random.normal(0, 5),
                    "price": 50 + np.random.normal(0, 2),
                    "promotional_activity": np.random.choice([0, 1], p=[0.9, 0.1]),
                    "weather_score": np.random.normal(0, 1),
                    "inventory_level": 150 + np.random.normal(0, 20),
                }
            )

        return jsonify(
            {
                "demo_data": demo_data[:10],  # Return first 10 records
                "total_records": len(demo_data),
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return jsonify({"error": f"Demo data generation failed: {str(e)}"}), 500


@app.route("/test", methods=["GET"])
def test_endpoint():
    """Test endpoint"""
    return jsonify(
        {
            "message": "Simplified API is working!",
            "endpoints": [
                "GET  /health",
                "GET  /test",
                "GET  /demo/data",
                "POST /predict/demand",
                "POST /optimize/inventory",
                "POST /analyze/causal",
            ],
            "timestamp": datetime.now().isoformat(),
        }
    )


if __name__ == "__main__":
    print("🚀 Starting Simplified Supply Chain API...")
    print("📊 Available at: http://localhost:5000")
    print("🔍 Test endpoint: http://localhost:5000/test")

    app.run(debug=True, host="0.0.0.0", port=5000)

"""
Model deployment utilities for production use
"""

import pickle
import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime
import logging


class ModelDeployment:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.preprocessors = {}

    def save_model(self, model, model_name, model_type="sklearn"):
        """Save trained model to disk"""
        model_path = f"{self.config.MODELS_PATH}/{model_name}"

        if model_type == "sklearn":
            joblib.dump(model, f"{model_path}.joblib")
        elif model_type == "tensorflow":
            model.save(f"{model_path}.h5")
        elif model_type == "prophet":
            with open(f"{model_path}.pkl", "wb") as f:
                pickle.dump(model, f)
        else:
            # Generic pickle
            with open(f"{model_path}.pkl", "wb") as f:
                pickle.dump(model, f)

        # Save metadata
        metadata = {
            "model_name": model_name,
            "model_type": model_type,
            "created_date": datetime.now().isoformat(),
            "config": self.config.__dict__,
        }

        with open(f"{model_path}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logging.info(f"Model {model_name} saved successfully")

    def load_model(self, model_name, model_type="sklearn"):
        """Load trained model from disk"""
        model_path = f"{self.config.MODELS_PATH}/{model_name}"

        try:
            if model_type == "sklearn":
                model = joblib.load(f"{model_path}.joblib")
            elif model_type == "tensorflow":
                from tensorflow import keras

                model = keras.models.load_model(f"{model_path}.h5")
            else:
                with open(f"{model_path}.pkl", "rb") as f:
                    model = pickle.load(f)

            self.models[model_name] = model
            logging.info(f"Model {model_name} loaded successfully")
            return model

        except Exception as e:
            logging.error(f"Failed to load model {model_name}: {str(e)}")
            return None

    def create_prediction_pipeline(self, model_name, preprocessing_steps):
        """Create end-to-end prediction pipeline"""

        class PredictionPipeline:
            def __init__(self, model, preprocessor, config):
                self.model = model
                self.preprocessor = preprocessor
                self.config = config

            def predict(self, input_data):
                """Make predictions on new data"""
                try:
                    # Apply preprocessing
                    processed_data = self.preprocessor.transform(input_data)

                    # Make prediction
                    predictions = self.model.predict(processed_data)

                    # Post-process predictions if needed
                    predictions = np.maximum(predictions, 0)  # Ensure non-negative

                    return predictions

                except Exception as e:
                    logging.error(f"Prediction failed: {str(e)}")
                    return None

            def predict_with_confidence(self, input_data):
                """Make predictions with confidence intervals"""
                predictions = self.predict(input_data)

                # Simple confidence interval (replace with proper uncertainty quantification)
                std_error = np.std(predictions) * 0.1
                lower_bound = predictions - 1.96 * std_error
                upper_bound = predictions + 1.96 * std_error

                return {
                    "predictions": predictions,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "confidence_level": 0.95,
                }

        model = self.models.get(model_name)
        if model is None:
            model = self.load_model(model_name)

        pipeline = PredictionPipeline(model, preprocessing_steps, self.config)
        return pipeline

    def batch_predict(self, model_name, input_file, output_file):
        """Run batch predictions on large dataset"""
        model = self.models.get(model_name)
        if model is None:
            model = self.load_model(model_name)

        # Load input data
        input_data = pd.read_csv(input_file)

        # Make predictions in chunks to handle large datasets
        chunk_size = 10000
        predictions = []

        for i in range(0, len(input_data), chunk_size):
            chunk = input_data.iloc[i : i + chunk_size]
            chunk_predictions = model.predict(chunk)
            predictions.extend(chunk_predictions)

        # Save results
        results = input_data.copy()
        results["predictions"] = predictions
        results.to_csv(output_file, index=False)

        logging.info(f"Batch predictions saved to {output_file}")
        return results

    def model_monitoring(self, model_name, new_data, reference_data):
        """Monitor model performance and detect drift"""
        from scipy import stats

        # Calculate distribution differences
        drift_metrics = {}

        for column in new_data.columns:
            if column in reference_data.columns:
                # KS test for distribution drift
                ks_stat, p_value = stats.ks_2samp(
                    reference_data[column].dropna(), new_data[column].dropna()
                )

                drift_metrics[column] = {
                    "ks_statistic": ks_stat,
                    "p_value": p_value,
                    "drift_detected": p_value < 0.05,
                }

        # Performance metrics on new data (if target is available)
        if "actual" in new_data.columns and "predicted" in new_data.columns:
            from sklearn.metrics import mean_absolute_error, mean_squared_error

            mae = mean_absolute_error(new_data["actual"], new_data["predicted"])
            rmse = np.sqrt(
                mean_squared_error(new_data["actual"], new_data["predicted"])
            )

            drift_metrics["performance"] = {"mae": mae, "rmse": rmse}

        return drift_metrics

    def auto_retrain_trigger(self, drift_metrics, performance_threshold=0.1):
        """Determine if model needs retraining based on drift metrics"""

        # Check for significant drift
        drift_detected = any(
            metric.get("drift_detected", False)
            for metric in drift_metrics.values()
            if isinstance(metric, dict)
        )

        # Check for performance degradation
        performance_degraded = False
        if "performance" in drift_metrics:
            # Compare with baseline (would need to store baseline metrics)
            # This is a simplified check
            performance_degraded = (
                drift_metrics["performance"]["rmse"] > performance_threshold
            )

        retrain_recommended = drift_detected or performance_degraded

        return {
            "retrain_recommended": retrain_recommended,
            "reasons": {
                "drift_detected": drift_detected,
                "performance_degraded": performance_degraded,
            },
            "drift_metrics": drift_metrics,
        }

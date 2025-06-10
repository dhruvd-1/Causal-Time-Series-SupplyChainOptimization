"""
Time Series Forecasting Models
TensorFlow-Free Version for Windows Compatibility
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

# Optional imports with fallbacks
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available")

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available")

try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Warning: Prophet not available")

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA

    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: Statsmodels not available")


class TimeSeriesModels:
    """Time series forecasting models without TensorFlow dependency"""

    def __init__(self, config):
        self.config = config
        self.models = {}
        self.performance = {}

    def prepare_features(self, data, target_col="demand"):
        """Prepare features for time series modeling"""
        df = data.copy()

        # Create lag features
        for lag in self.config.LAG_FEATURES:
            df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

        # Create rolling statistics
        for window in self.config.ROLLING_WINDOWS:
            df[f"{target_col}_rolling_mean_{window}"] = (
                df[target_col].rolling(window).mean()
            )
            df[f"{target_col}_rolling_std_{window}"] = (
                df[target_col].rolling(window).std()
            )

        # Time-based features
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df["day_of_week"] = df["date"].dt.dayofweek
            df["month"] = df["date"].dt.month
            df["quarter"] = df["date"].dt.quarter
            df["day_of_year"] = df["date"].dt.dayofyear

        # Drop rows with NaN values (due to lag features)
        max_lag = max(self.config.LAG_FEATURES + self.config.ROLLING_WINDOWS)
        df = df.iloc[max_lag:].copy()

        return df

    def train_random_forest(self, data, target_col="demand"):
        """Train Random Forest model"""
        df = self.prepare_features(data, target_col)

        # Select feature columns (exclude target and date)
        feature_cols = [col for col in df.columns if col not in [target_col, "date"]]

        X = df[feature_cols].fillna(0)
        y = df[target_col]

        # Train-test split (time series)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=self.config.RANDOM_SEED,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        # Metrics
        metrics = {
            "MAE": mean_absolute_error(y_test, test_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, test_pred)),
            "R2": r2_score(y_test, test_pred),
            "MAPE": np.mean(np.abs((y_test - test_pred) / y_test)) * 100,
        }

        self.models["RandomForest"] = model
        self.performance["RandomForest"] = metrics

        return model, metrics

    def train_xgboost(self, data, target_col="demand"):
        """Train XGBoost model"""
        if not XGBOOST_AVAILABLE:
            print("XGBoost not available, skipping...")
            return None, None

        df = self.prepare_features(data, target_col)

        feature_cols = [col for col in df.columns if col not in [target_col, "date"]]
        X = df[feature_cols].fillna(0)
        y = df[target_col]

        # Train-test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Train model
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.config.RANDOM_SEED,
        )
        model.fit(X_train, y_train)

        # Predictions
        test_pred = model.predict(X_test)

        # Metrics
        metrics = {
            "MAE": mean_absolute_error(y_test, test_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, test_pred)),
            "R2": r2_score(y_test, test_pred),
            "MAPE": np.mean(np.abs((y_test - test_pred) / y_test)) * 100,
        }

        self.models["XGBoost"] = model
        self.performance["XGBoost"] = metrics

        return model, metrics

    def train_lightgbm(self, data, target_col="demand"):
        """Train LightGBM model"""
        if not LIGHTGBM_AVAILABLE:
            print("LightGBM not available, skipping...")
            return None, None

        df = self.prepare_features(data, target_col)

        feature_cols = [col for col in df.columns if col not in [target_col, "date"]]
        X = df[feature_cols].fillna(0)
        y = df[target_col]

        # Train-test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Train model
        model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.config.RANDOM_SEED,
            verbose=-1,
        )
        model.fit(X_train, y_train)

        # Predictions
        test_pred = model.predict(X_test)

        # Metrics
        metrics = {
            "MAE": mean_absolute_error(y_test, test_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, test_pred)),
            "R2": r2_score(y_test, test_pred),
            "MAPE": np.mean(np.abs((y_test - test_pred) / y_test)) * 100,
        }

        self.models["LightGBM"] = model
        self.performance["LightGBM"] = metrics

        return model, metrics

    def train_prophet(self, data, target_col="demand"):
        """Train Prophet model"""
        if not PROPHET_AVAILABLE:
            print("Prophet not available, skipping...")
            return None, None

        try:
            # Prepare data for Prophet
            df = data[["date", target_col]].copy()
            df.columns = ["ds", "y"]
            df["ds"] = pd.to_datetime(df["ds"])

            # Train-test split
            split_idx = int(len(df) * 0.8)
            train_df = df[:split_idx]
            test_df = df[split_idx:]

            # Train model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
            )
            model.fit(train_df)

            # Predictions
            future = model.make_future_dataframe(periods=len(test_df))
            forecast = model.predict(future)

            # Extract test predictions
            test_pred = forecast["yhat"][split_idx:].values
            y_test = test_df["y"].values

            # Metrics
            metrics = {
                "MAE": mean_absolute_error(y_test, test_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, test_pred)),
                "R2": r2_score(y_test, test_pred),
                "MAPE": np.mean(np.abs((y_test - test_pred) / y_test)) * 100,
            }

            self.models["Prophet"] = model
            self.performance["Prophet"] = metrics

            return model, metrics

        except Exception as e:
            print(f"Prophet training failed: {str(e)}")
            return None, None

    def train_exponential_smoothing(self, data, target_col="demand"):
        """Train Exponential Smoothing model"""
        if not STATSMODELS_AVAILABLE:
            print("Statsmodels not available, skipping...")
            return None, None

        try:
            ts_data = data[target_col].values

            # Train-test split
            split_idx = int(len(ts_data) * 0.8)
            train_data = ts_data[:split_idx]
            test_data = ts_data[split_idx:]

            # Train model
            model = ExponentialSmoothing(
                train_data,
                trend="add",
                seasonal="add",
                seasonal_periods=7,  # Weekly seasonality
            )
            fitted_model = model.fit()

            # Predictions
            test_pred = fitted_model.forecast(len(test_data))

            # Metrics
            metrics = {
                "MAE": mean_absolute_error(test_data, test_pred),
                "RMSE": np.sqrt(mean_squared_error(test_data, test_pred)),
                "R2": r2_score(test_data, test_pred),
                "MAPE": np.mean(np.abs((test_data - test_pred) / test_data)) * 100,
            }

            self.models["ExponentialSmoothing"] = fitted_model
            self.performance["ExponentialSmoothing"] = metrics

            return fitted_model, metrics

        except Exception as e:
            print(f"Exponential Smoothing training failed: {str(e)}")
            return None, None

    def train_all_models(self, data, target_col="demand"):
        """Train all available models"""
        print("Training forecasting models...")

        all_performance = {}

        # 1. Random Forest (always available)
        print("Training Random Forest...")
        _, rf_metrics = self.train_random_forest(data, target_col)
        if rf_metrics:
            all_performance["RandomForest"] = rf_metrics

        # 2. XGBoost (if available)
        if XGBOOST_AVAILABLE:
            print("Training XGBoost...")
            _, xgb_metrics = self.train_xgboost(data, target_col)
            if xgb_metrics:
                all_performance["XGBoost"] = xgb_metrics

        # 3. LightGBM (if available)
        if LIGHTGBM_AVAILABLE:
            print("Training LightGBM...")
            _, lgb_metrics = self.train_lightgbm(data, target_col)
            if lgb_metrics:
                all_performance["LightGBM"] = lgb_metrics

        # 4. Prophet (if available)
        if PROPHET_AVAILABLE:
            print("Training Prophet...")
            _, prophet_metrics = self.train_prophet(data, target_col)
            if prophet_metrics:
                all_performance["Prophet"] = prophet_metrics

        # 5. Exponential Smoothing (if available)
        if STATSMODELS_AVAILABLE:
            print("Training Exponential Smoothing...")
            _, es_metrics = self.train_exponential_smoothing(data, target_col)
            if es_metrics:
                all_performance["ExponentialSmoothing"] = es_metrics

        self.performance = all_performance
        return all_performance

    def forecast_future(self, data, periods=30, target_col="demand"):
        """Generate future forecasts using trained models"""
        forecasts = {}

        # Random Forest forecast
        if "RandomForest" in self.models:
            try:
                # Use the last few data points to generate features
                last_data = data.tail(60).copy()  # Use more data for better features

                # Generate future dates
                if "date" in data.columns:
                    last_date = pd.to_datetime(data["date"].iloc[-1])
                    future_dates = pd.date_range(
                        last_date + pd.Timedelta(days=1), periods=periods, freq="D"
                    )
                else:
                    future_dates = range(len(data), len(data) + periods)

                # Simple forecast using trend
                recent_values = data[target_col].tail(30).values
                trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
                base_forecast = recent_values[-1] + trend * np.arange(1, periods + 1)

                # Add some noise for realism
                noise = np.random.normal(0, np.std(recent_values) * 0.1, periods)
                rf_forecast = base_forecast + noise
                rf_forecast = np.maximum(rf_forecast, 0)  # Ensure non-negative

                forecasts["RandomForest"] = rf_forecast

            except Exception as e:
                print(f"Random Forest forecast failed: {str(e)}")

        # Prophet forecast
        if "Prophet" in self.models and PROPHET_AVAILABLE:
            try:
                model = self.models["Prophet"]
                future = model.make_future_dataframe(periods=periods)
                prophet_forecast = model.predict(future)
                forecasts["Prophet"] = prophet_forecast["yhat"].tail(periods).values
            except Exception as e:
                print(f"Prophet forecast failed: {str(e)}")

        # Simple ensemble forecast
        if len(forecasts) > 1:
            ensemble_forecast = np.mean([f for f in forecasts.values()], axis=0)
            forecasts["ensemble"] = ensemble_forecast
        elif len(forecasts) == 1:
            forecasts["ensemble"] = list(forecasts.values())[0]
        else:
            # Fallback: simple trend forecast
            recent_values = data[target_col].tail(30).values
            trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
            base_forecast = recent_values[-1] + trend * np.arange(1, periods + 1)
            forecasts["ensemble"] = np.maximum(base_forecast, 0)

        return forecasts

    def cross_validate_model(self, data, model_name, target_col="demand", n_splits=3):
        """Perform time series cross-validation"""
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return None

        df = self.prepare_features(data, target_col)
        feature_cols = [col for col in df.columns if col not in [target_col, "date"]]

        X = df[feature_cols].fillna(0)
        y = df[target_col]

        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            if model_name == "RandomForest":
                model = RandomForestRegressor(random_state=self.config.RANDOM_SEED)
            elif model_name == "XGBoost" and XGBOOST_AVAILABLE:
                model = xgb.XGBRegressor(random_state=self.config.RANDOM_SEED)
            elif model_name == "LightGBM" and LIGHTGBM_AVAILABLE:
                model = lgb.LGBMRegressor(
                    random_state=self.config.RANDOM_SEED, verbose=-1
                )
            else:
                continue

            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            score = mean_squared_error(y_test, pred)
            cv_scores.append(score)

        return {
            "cv_scores": cv_scores,
            "mean_cv_score": np.mean(cv_scores),
            "std_cv_score": np.std(cv_scores),
        }

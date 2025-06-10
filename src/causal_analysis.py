"""
Simplified Causal Analysis - Python 3.12 Compatible
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Optional imports
try:
    from scipy import stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.linear_model import LinearRegression

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


class BasicCausalAnalyzer:
    """Simplified causal analysis class"""

    def __init__(self, config):
        self.config = config

    def simple_correlation(self, data, treatment, outcome):
        """Simple correlation analysis"""
        try:
            if treatment not in data.columns or outcome not in data.columns:
                return {"error": f"Columns {treatment} or {outcome} not found"}

            # Get clean data
            df_clean = data[[treatment, outcome]].dropna()

            if len(df_clean) < 2:
                return {"error": "Insufficient data for correlation"}

            # Calculate correlation
            correlation = df_clean[treatment].corr(df_clean[outcome])

            # Simple p-value approximation (if scipy not available)
            if SCIPY_AVAILABLE:
                try:
                    _, p_value = stats.pearsonr(df_clean[treatment], df_clean[outcome])
                except:
                    p_value = 0.5  # Default
            else:
                # Rough approximation based on sample size and correlation
                n = len(df_clean)
                t_stat = (
                    correlation * np.sqrt((n - 2) / (1 - correlation**2))
                    if correlation != 1
                    else 0
                )
                p_value = 0.05 if abs(t_stat) > 2 else 0.5  # Very rough approximation

            return {
                "correlation": float(correlation) if not pd.isna(correlation) else 0.0,
                "p_value": float(p_value),
                "significant": float(p_value) < 0.05,
                "n_observations": len(df_clean),
            }

        except Exception as e:
            return {"error": str(e)}

    def simple_regression(self, data, treatment, outcome):
        """Simple regression analysis"""
        try:
            if not SKLEARN_AVAILABLE:
                return {"error": "Sklearn not available"}

            if treatment not in data.columns or outcome not in data.columns:
                return {"error": f"Columns {treatment} or {outcome} not found"}

            # Get clean data
            df_clean = data[[treatment, outcome]].dropna()

            if len(df_clean) < 3:
                return {"error": "Insufficient data for regression"}

            # Prepare data
            X = df_clean[[treatment]]
            y = df_clean[outcome]

            # Fit model
            model = LinearRegression()
            model.fit(X, y)

            # Get results
            treatment_effect = model.coef_[0]
            r_squared = model.score(X, y)

            return {
                "treatment_effect": float(treatment_effect),
                "r_squared": float(r_squared),
                "n_observations": len(df_clean),
            }

        except Exception as e:
            return {"error": str(e)}

    def comprehensive_analysis(self, data, treatment_outcome_pairs):
        """Run analysis for multiple treatment-outcome pairs"""
        results = {}

        for treatment, outcome in treatment_outcome_pairs:
            pair_name = f"{treatment}_on_{outcome}"
            results[pair_name] = {}

            print(f"Analyzing: {pair_name}")

            # Correlation analysis
            corr_result = self.simple_correlation(data, treatment, outcome)
            results[pair_name]["correlation"] = corr_result

            # Regression analysis
            reg_result = self.simple_regression(data, treatment, outcome)
            results[pair_name]["regression"] = reg_result

        return results

    def visualize_causal_effects(self, results):
        """Simple visualization of results"""
        if not PLOTTING_AVAILABLE:
            print("Plotting libraries not available")
            return None

        try:
            # Extract data for plotting
            plot_data = []

            for pair_name, analyses in results.items():
                # Correlation data
                if "correlation" in analyses and "error" not in analyses["correlation"]:
                    plot_data.append(
                        {
                            "pair": pair_name,
                            "method": "Correlation",
                            "effect": analyses["correlation"]["correlation"],
                        }
                    )

                # Regression data
                if "regression" in analyses and "error" not in analyses["regression"]:
                    plot_data.append(
                        {
                            "pair": pair_name,
                            "method": "Regression",
                            "effect": analyses["regression"]["treatment_effect"],
                        }
                    )

            if plot_data:
                df = pd.DataFrame(plot_data)

                plt.figure(figsize=(10, 6))

                # Simple bar plot
                pairs = df["pair"].unique()
                methods = df["method"].unique()

                x = np.arange(len(pairs))
                width = 0.35

                for i, method in enumerate(methods):
                    method_data = df[df["method"] == method]
                    effects = [
                        method_data[method_data["pair"] == pair]["effect"].iloc[0]
                        if len(method_data[method_data["pair"] == pair]) > 0
                        else 0
                        for pair in pairs
                    ]

                    plt.bar(x + i * width, effects, width, label=method)

                plt.xlabel("Treatment-Outcome Pairs")
                plt.ylabel("Effect Size")
                plt.title("Causal Effects Analysis")
                plt.xticks(x + width / 2, pairs, rotation=45)
                plt.legend()
                plt.tight_layout()
                plt.show()

                return df
            else:
                print("No valid data for visualization")
                return None

        except Exception as e:
            print(f"Visualization error: {e}")
            return None


# Simple test function
def test_causal_analyzer():
    """Test the causal analyzer with sample data"""

    # Create sample data
    np.random.seed(42)
    n = 100

    data = pd.DataFrame(
        {
            "price": np.random.normal(50, 5, n),
            "promotional_activity": np.random.choice([0, 1], n, p=[0.8, 0.2]),
            "weather_score": np.random.normal(0, 1, n),
            "demand": np.random.normal(100, 10, n),
        }
    )

    # Add some causal relationships
    data["demand"] = (
        data["demand"] - 2 * (data["price"] - 50) + 15 * data["promotional_activity"]
    )

    # Test analyzer
    config = type("Config", (), {})()  # Simple config object
    analyzer = BasicCausalAnalyzer(config)

    # Run analysis
    results = analyzer.comprehensive_analysis(
        data, [("price", "demand"), ("promotional_activity", "demand")]
    )

    # Print results
    for pair, analyses in results.items():
        print(f"\n{pair}:")
        for method, result in analyses.items():
            print(f"  {method}: {result}")

    return results


if __name__ == "__main__":
    print("Testing Causal Analyzer...")
    test_results = test_causal_analyzer()

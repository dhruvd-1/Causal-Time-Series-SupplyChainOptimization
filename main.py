"""
Main execution script - Python 3.12 Compatible Version
"""

import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import our modules
from config import Config
from src.data_preprocessing import DataPreprocessor
from src.time_series_models import TimeSeriesModels
from src.optimization import SupplyChainOptimizer
from src.visualization import Visualizer

# Import basic causal analysis
from causal_analysis import BasicCausalAnalyzer


def main():
    """Main execution function"""

    print("=" * 60)
    print("SUPPLY CHAIN CAUSAL ANALYSIS - Python 3.12 Compatible")
    print("=" * 60)

    # Initialize configuration
    config = Config()
    np.random.seed(config.RANDOM_SEED)

    # Initialize components
    preprocessor = DataPreprocessor(config)
    ts_models = TimeSeriesModels(config)
    optimizer = SupplyChainOptimizer(config)
    visualizer = Visualizer(config)
    causal_analyzer = BasicCausalAnalyzer(config)

    # 1. Data Generation and Preprocessing
    print("\n1. Generating and preprocessing data...")

    # Generate synthetic data
    raw_data = preprocessor.generate_synthetic_data(
        start_date="2023-01-01", end_date="2024-06-01"
    )

    # Prepare for modeling
    modeling_data = preprocessor.prepare_for_modeling(raw_data)
    print(f"Generated {len(modeling_data['data'])} data points")

    # 2. Basic Causal Analysis
    print("\n2. Running causal analysis...")

    treatment_outcome_pairs = [
        ("price", "demand"),
        ("promotional_activity", "demand"),
        ("competitor_price", "demand"),
        ("weather_score", "demand"),
    ]

    causal_results = causal_analyzer.comprehensive_analysis(
        modeling_data["data"], treatment_outcome_pairs
    )

    # Display causal results
    for pair, results in causal_results.items():
        print(f"\nCausal Effect: {pair}")
        for method, result in results.items():
            if method == "correlation":
                print(
                    f"  Correlation: {result['correlation']:.3f} (p={result['p_value']:.3f})"
                )
            elif method == "regression":
                print(
                    f"  Regression Effect: {result['treatment_effect']:.3f} (R²={result['r_squared']:.3f})"
                )

    # 3. Time Series Forecasting
    print("\n3. Training forecasting models...")

    # Train models
    model_performance = ts_models.train_all_models(modeling_data["data"])

    # Display performance
    print("\nModel Performance:")
    for model, metrics in model_performance.items():
        print(f"  {model}: RMSE={metrics['RMSE']:.2f}, R²={metrics['R2']:.3f}")

    # Generate forecasts
    forecasts = ts_models.forecast_future(modeling_data["data"], periods=30)

    # 4. Supply Chain Optimization
    print("\n4. Optimizing supply chain parameters...")

    # Use best forecast for optimization
    if forecasts:
        best_forecast = forecasts["ensemble"]

        # Run optimization
        optimization_result = optimizer.optimize_inventory_policy(
            best_forecast, method="scipy"
        )

        print(f"\nOptimization Results:")
        print(f"  Optimal Cost: ${optimization_result['optimal_cost']:.2f}")
        print(
            f"  Reorder Point: {optimization_result['optimal_params']['reorder_point']:.0f}"
        )
        print(
            f"  Order Quantity: {optimization_result['optimal_params']['order_quantity']:.0f}"
        )
        print(
            f"  Service Level: {optimization_result['performance_metrics']['service_level']:.1%}"
        )

    # 5. Visualization
    print("\n5. Creating visualizations...")

    # Create comprehensive dashboard
    visualizer.create_dashboard(
        data=modeling_data["data"],
        forecasts=forecasts,
        model_performance=model_performance,
        causal_results=causal_results,
    )

    # 6. Risk Analysis
    print("\n6. Conducting risk analysis...")

    # Monte Carlo simulation
    n_simulations = 500  # Reduced for faster execution
    simulation_results = []

    for i in range(n_simulations):
        # Generate random demand scenario
        base_demand = (
            forecasts["ensemble"]
            if forecasts
            else modeling_data["data"]["demand"].values[-30:]
        )
        noise = np.random.normal(0, np.std(base_demand) * 0.2, len(base_demand))
        random_demand = base_demand + noise
        random_demand = np.maximum(random_demand, 0)  # Ensure non-negative

        try:
            result = optimizer.optimize_inventory_policy(random_demand, method="scipy")
            simulation_results.append(
                {
                    "cost": result["optimal_cost"],
                    "service_level": result["performance_metrics"]["service_level"],
                    "reorder_point": result["optimal_params"]["reorder_point"],
                }
            )
        except:
            continue

    if simulation_results:
        sim_df = pd.DataFrame(simulation_results)

        print(f"\nRisk Analysis Results ({len(simulation_results)} simulations):")
        print(
            f"  Average Cost: ${sim_df['cost'].mean():.2f} ± ${sim_df['cost'].std():.2f}"
        )
        print(f"  95th Percentile Cost: ${sim_df['cost'].quantile(0.95):.2f}")
        print(f"  Average Service Level: {sim_df['service_level'].mean():.1%}")

        # Plot risk distribution
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.hist(sim_df["cost"], bins=30, alpha=0.7, edgecolor="black")
        plt.axvline(sim_df["cost"].mean(), color="red", linestyle="--", label="Mean")
        plt.title("Cost Distribution")
        plt.xlabel("Total Cost")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.hist(sim_df["service_level"], bins=30, alpha=0.7, edgecolor="black")
        plt.axvline(
            sim_df["service_level"].mean(), color="red", linestyle="--", label="Mean"
        )
        plt.title("Service Level Distribution")
        plt.xlabel("Service Level")
        plt.legend()

        plt.tight_layout()
        plt.show()

    # 7. Save Results
    print("\n7. Saving results...")

    # Save data
    modeling_data["data"].to_csv(
        f"{config.RESULTS_PATH}processed_data.csv", index=False
    )

    # Save performance metrics
    pd.DataFrame(model_performance).T.to_csv(
        f"{config.RESULTS_PATH}model_performance.csv"
    )

    if simulation_results:
        sim_df.to_csv(f"{config.RESULTS_PATH}risk_analysis.csv", index=False)

    print(f"Results saved to {config.RESULTS_PATH}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    # Summary recommendations
    print("\nKEY INSIGHTS & RECOMMENDATIONS:")
    print("• Demand forecasting models trained and evaluated")
    print("• Causal relationships identified between key variables")
    print("• Optimal inventory policies calculated")
    print("• Risk analysis completed with Monte Carlo simulation")
    print("• Interactive visualizations generated")

    return {
        "data": modeling_data,
        "causal_results": causal_results,
        "model_performance": model_performance,
        "forecasts": forecasts,
        "optimization_result": optimization_result
        if "optimization_result" in locals()
        else None,
        "risk_analysis": sim_df if "sim_df" in locals() else None,
    }


if __name__ == "__main__":
    results = main()

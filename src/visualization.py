import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class SupplyChainVisualizer:
    def __init__(self, config):
        self.config = config
        self.fig_size = (12, 8)

        # Set style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

    def plot_time_series(self, data, columns, title="Time Series Analysis"):
        """Plot multiple time series"""
        fig, axes = plt.subplots(len(columns), 1, figsize=(15, 4 * len(columns)))
        if len(columns) == 1:
            axes = [axes]

        for i, col in enumerate(columns):
            axes[i].plot(data["date"], data[col], linewidth=2)
            axes[i].set_title(f"{col.replace('_', ' ').title()}")
            axes[i].set_xlabel("Date")
            axes[i].set_ylabel(col)
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.suptitle(title, y=1.02, fontsize=16)
        return fig

    def plot_causal_effects(self, causal_results):
        """Visualize causal analysis results"""
        # Extract treatment effects
        effects_data = []
        for treatment_outcome, results in causal_results.items():
            if "error" not in results and "treatment_effects" in results:
                for method, effect in results["treatment_effects"].items():
                    effects_data.append(
                        {
                            "Treatment_Outcome": treatment_outcome,
                            "Method": method,
                            "Effect": effect,
                        }
                    )

        if not effects_data:
            print("No causal effects data to plot")
            return None

        df_effects = pd.DataFrame(effects_data)

        # Create subplot for causal effects
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Bar plot of treatment effects
        sns.barplot(
            data=df_effects, x="Treatment_Outcome", y="Effect", hue="Method", ax=ax1
        )
        ax1.set_title("Causal Treatment Effects by Method")
        ax1.set_xlabel("Treatment → Outcome")
        ax1.set_ylabel("Effect Size")
        ax1.tick_params(axis="x", rotation=45)

        # Box plot showing distribution of effects
        sns.boxplot(data=df_effects, x="Method", y="Effect", ax=ax2)
        ax2.set_title("Distribution of Treatment Effects")
        ax2.set_xlabel("Estimation Method")
        ax2.set_ylabel("Effect Size")

        plt.tight_layout()
        return fig

    def plot_model_performance(self, model_performance):
        """Visualize model performance metrics"""
        # Convert to DataFrame
        metrics_df = pd.DataFrame(model_performance).T

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()

        metrics = ["MAE", "RMSE", "MAPE", "R2"]

        for i, metric in enumerate(metrics):
            if metric in metrics_df.columns:
                metrics_df[metric].plot(kind="bar", ax=axes[i])
                axes[i].set_title(f"{metric} Comparison")
                axes[i].set_xlabel("Models")
                axes[i].set_ylabel(metric)
                axes[i].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        return fig

    def plot_forecasts(self, actual, predictions, model_names, forecast_horizon=30):
        """Plot actual vs predicted values"""
        fig, ax = plt.subplots(figsize=(15, 8))

        # Plot actual values
        ax.plot(range(len(actual)), actual, label="Actual", linewidth=2, color="black")

        # Plot predictions
        colors = plt.cm.Set3(np.linspace(0, 1, len(predictions)))
        for i, (model_name, pred) in enumerate(zip(model_names, predictions)):
            ax.plot(
                range(len(actual) - len(pred), len(actual)),
                pred,
                label=f"{model_name} Prediction",
                linewidth=2,
                color=colors[i],
            )

        ax.axvline(
            x=len(actual) - forecast_horizon,
            color="red",
            linestyle="--",
            alpha=0.7,
            label="Forecast Start",
        )

        ax.set_title("Demand Forecasting Results")
        ax.set_xlabel("Time Period")
        ax.set_ylabel("Demand")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig

    def plot_optimization_results(self, optimization_results):
        """Visualize optimization results"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Extract optimization methods and metrics
        methods = list(optimization_results.keys())
        if "robust" in optimization_results:
            methods.remove("robust")  # Handle separately

        # 1. Optimal parameters comparison
        params_data = []
        for method in methods:
            if "optimal_params" in optimization_results[method]:
                params = optimization_results[method]["optimal_params"]
                for param, value in params.items():
                    params_data.append(
                        {"Method": method, "Parameter": param, "Value": value}
                    )

        if params_data:
            df_params = pd.DataFrame(params_data)
            sns.barplot(
                data=df_params, x="Parameter", y="Value", hue="Method", ax=axes[0, 0]
            )
            axes[0, 0].set_title("Optimal Parameters by Method")
            axes[0, 0].tick_params(axis="x", rotation=45)

        # 2. Performance metrics comparison
        metrics_data = []
        for method in methods:
            if "performance_metrics" in optimization_results[method]:
                metrics = optimization_results[method]["performance_metrics"]
                for metric, value in metrics.items():
                    metrics_data.append(
                        {"Method": method, "Metric": metric, "Value": value}
                    )

        if metrics_data:
            df_metrics = pd.DataFrame(metrics_data)
            sns.barplot(
                data=df_metrics, x="Metric", y="Value", hue="Method", ax=axes[0, 1]
            )
            axes[0, 1].set_title("Performance Metrics by Method")
            axes[0, 1].tick_params(axis="x", rotation=45)

        # 3. Cost comparison
        costs_data = []
        for method in methods:
            if "optimal_cost" in optimization_results[method]:
                costs_data.append(
                    {
                        "Method": method,
                        "Cost": optimization_results[method]["optimal_cost"],
                    }
                )

        if costs_data:
            df_costs = pd.DataFrame(costs_data)
            sns.barplot(data=df_costs, x="Method", y="Cost", ax=axes[1, 0])
            axes[1, 0].set_title("Optimal Cost by Method")
            axes[1, 0].tick_params(axis="x", rotation=45)

        # 4. Robust optimization analysis (if available)
        if "robust" in optimization_results:
            robust_data = optimization_results["robust"]
            scenario_costs = [
                r["optimal_cost"] for r in robust_data["scenario_results"]
            ]

            axes[1, 1].hist(scenario_costs, bins=20, alpha=0.7, edgecolor="black")
            axes[1, 1].axvline(
                robust_data["cvar"],
                color="red",
                linestyle="--",
                label=f"CVaR: {robust_data['cvar']:.2f}",
            )
            axes[1, 1].axvline(
                robust_data["average_cost"],
                color="green",
                linestyle="--",
                label=f"Mean: {robust_data['average_cost']:.2f}",
            )
            axes[1, 1].set_title("Cost Distribution Across Scenarios")
            axes[1, 1].set_xlabel("Cost")
            axes[1, 1].set_ylabel("Frequency")
            axes[1, 1].legend()

        plt.tight_layout()
        return fig

    def plot_interactive_dashboard(self, data, predictions, causal_results):
        """Create interactive dashboard using Plotly"""
        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Demand Time Series",
                "Prediction Accuracy",
                "Causal Effects",
                "Cost Analysis",
            ),
            specs=[
                [{"secondary_y": True}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "scatter"}],
            ],
        )

        # 1. Time series plot
        fig.add_trace(
            go.Scatter(
                x=data["date"],
                y=data["demand"],
                name="Actual Demand",
                line=dict(color="blue"),
            ),
            row=1,
            col=1,
        )

        if "inventory_level" in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data["date"],
                    y=data["inventory_level"],
                    name="Inventory Level",
                    line=dict(color="green"),
                ),
                row=1,
                col=1,
                secondary_y=True,
            )

        # 2. Prediction accuracy
        if predictions:
            actual_test = (
                data["demand"].iloc[-len(list(predictions.values())[0]) :].values
            )
            for model_name, pred in predictions.items():
                fig.add_trace(
                    go.Scatter(
                        x=actual_test,
                        y=pred,
                        mode="markers",
                        name=f"{model_name}",
                        marker=dict(size=8, opacity=0.7),
                    ),
                    row=1,
                    col=2,
                )

            # Add perfect prediction line
            min_val, max_val = min(actual_test), max(actual_test)
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    name="Perfect Prediction",
                    line=dict(dash="dash", color="red"),
                ),
                row=1,
                col=2,
            )

        # 3. Causal effects
        if causal_results:
            effects_data = []
            for treatment_outcome, results in causal_results.items():
                if "error" not in results and "treatment_effects" in results:
                    for method, effect in results["treatment_effects"].items():
                        effects_data.append(
                            {
                                "Treatment_Outcome": treatment_outcome.replace(
                                    "_on_", " → "
                                ),
                                "Effect": effect,
                            }
                        )

            if effects_data:
                df_effects = pd.DataFrame(effects_data)
                fig.add_trace(
                    go.Bar(
                        x=df_effects["Treatment_Outcome"],
                        y=df_effects["Effect"],
                        name="Causal Effects",
                    ),
                    row=2,
                    col=1,
                )

        # 4. Cost analysis
        if "total_cost" in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data["date"],
                    y=data["total_cost"],
                    mode="lines",
                    name="Total Cost",
                    line=dict(color="orange"),
                ),
                row=2,
                col=2,
            )

        # Update layout
        fig.update_layout(
            height=800, title_text="Supply Chain Analytics Dashboard", showlegend=True
        )

        # Update x and y axis labels
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Actual Demand", row=1, col=2)
        fig.update_xaxes(title_text="Treatment → Outcome", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=2)

        fig.update_yaxes(title_text="Demand", row=1, col=1)
        fig.update_yaxes(title_text="Predicted Demand", row=1, col=2)
        fig.update_yaxes(title_text="Effect Size", row=2, col=1)
        fig.update_yaxes(title_text="Cost", row=2, col=2)

        return fig

    def plot_sensitivity_analysis(self, sensitivity_results):
        """Plot sensitivity analysis results"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Example sensitivity analysis plots
        # This would depend on the specific sensitivity analysis performed

        # 1. Parameter sensitivity
        if "parameter_sensitivity" in sensitivity_results:
            param_data = sensitivity_results["parameter_sensitivity"]

            for param, values in param_data.items():
                axes[0, 0].plot(
                    values["range"], values["objective"], label=param, marker="o"
                )

            axes[0, 0].set_title("Parameter Sensitivity Analysis")
            axes[0, 0].set_xlabel("Parameter Value")
            axes[0, 0].set_ylabel("Objective Function")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # 2. Demand uncertainty impact
        if "demand_uncertainty" in sensitivity_results:
            uncertainty_data = sensitivity_results["demand_uncertainty"]

            axes[0, 1].fill_between(
                uncertainty_data["time"],
                uncertainty_data["lower_bound"],
                uncertainty_data["upper_bound"],
                alpha=0.3,
                label="Uncertainty Band",
            )
            axes[0, 1].plot(
                uncertainty_data["time"],
                uncertainty_data["expected"],
                label="Expected Value",
                linewidth=2,
            )

            axes[0, 1].set_title("Demand Uncertainty Impact")
            axes[0, 1].set_xlabel("Time")
            axes[0, 1].set_ylabel("Cost")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # 3. Service level vs Cost trade-off
        if "tradeoff_analysis" in sensitivity_results:
            tradeoff_data = sensitivity_results["tradeoff_analysis"]

            axes[1, 0].scatter(
                tradeoff_data["service_levels"],
                tradeoff_data["costs"],
                c=tradeoff_data["inventory_levels"],
                cmap="viridis",
                s=100,
            )

            axes[1, 0].set_title("Service Level vs Cost Trade-off")
            axes[1, 0].set_xlabel("Service Level")
            axes[1, 0].set_ylabel("Total Cost")

            # Add colorbar
            cbar = plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0])
            cbar.set_label("Inventory Level")

        # 4. Pareto frontier
        if "pareto_frontier" in sensitivity_results:
            pareto_data = sensitivity_results["pareto_frontier"]

            axes[1, 1].plot(
                pareto_data["cost"],
                pareto_data["service_level"],
                "ro-",
                linewidth=2,
                markersize=8,
            )
            axes[1, 1].set_title("Pareto Frontier: Cost vs Service Level")
            axes[1, 1].set_xlabel("Total Cost")
            axes[1, 1].set_ylabel("Service Level")
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def generate_report_plots(self, data, models, causal_results, optimization_results):
        """Generate comprehensive set of plots for reporting"""
        plots = {}

        # 1. Executive summary plot
        fig_summary = plt.figure(figsize=(20, 12))
        gs = fig_summary.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # Demand overview
        ax1 = fig_summary.add_subplot(gs[0, :2])
        ax1.plot(data["date"], data["demand"], linewidth=2, color="blue")
        ax1.set_title("Demand Overview", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Demand")
        ax1.grid(True, alpha=0.3)

        # Cost breakdown
        ax2 = fig_summary.add_subplot(gs[0, 2:])
        if all(col in data.columns for col in ["holding_cost", "stockout_cost"]):
            cost_data = [data["holding_cost"].sum(), data["stockout_cost"].sum()]
            ax2.pie(
                cost_data,
                labels=["Holding Cost", "Stockout Cost"],
                autopct="%1.1f%%",
                startangle=90,
            )
            ax2.set_title("Cost Breakdown", fontsize=14, fontweight="bold")

        # Model performance
        ax3 = fig_summary.add_subplot(gs[1, :2])
        if hasattr(models, "model_performance") and models.model_performance:
            perf_df = pd.DataFrame(models.model_performance).T
            if "RMSE" in perf_df.columns:
                perf_df["RMSE"].plot(kind="bar", ax=ax3)
                ax3.set_title("Model RMSE Comparison", fontsize=14, fontweight="bold")
                ax3.set_ylabel("RMSE")
                ax3.tick_params(axis="x", rotation=45)

        # Optimization results
        ax4 = fig_summary.add_subplot(gs[1, 2:])
        if optimization_results:
            methods = [k for k in optimization_results.keys() if k != "robust"]
            costs = [optimization_results[m].get("optimal_cost", 0) for m in methods]
            ax4.bar(methods, costs)
            ax4.set_title(
                "Optimization Cost Comparison", fontsize=14, fontweight="bold"
            )
            ax4.set_ylabel("Total Cost")
            ax4.tick_params(axis="x", rotation=45)

        # Service level trend
        ax5 = fig_summary.add_subplot(gs[2, :])
        if "service_level" in data.columns:
            monthly_service = data.groupby(data["date"].dt.to_period("M"))[
                "service_level"
            ].mean()
            ax5.plot(
                monthly_service.index.astype(str),
                monthly_service.values,
                marker="o",
                linewidth=2,
                markersize=6,
            )
            ax5.set_title("Monthly Service Level Trend", fontsize=14, fontweight="bold")
            ax5.set_ylabel("Service Level")
            ax5.tick_params(axis="x", rotation=45)
            ax5.grid(True, alpha=0.3)

        plots["executive_summary"] = fig_summary

        # 2. Detailed analysis plots
        plots["time_series"] = self.plot_time_series(
            data, ["demand", "inventory_level", "total_cost"]
        )

        if hasattr(models, "model_performance") and models.model_performance:
            plots["model_performance"] = self.plot_model_performance(
                models.model_performance
            )

        if causal_results:
            plots["causal_effects"] = self.plot_causal_effects(causal_results)

        if optimization_results:
            plots["optimization"] = self.plot_optimization_results(optimization_results)

        return plots

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
import warnings

warnings.filterwarnings("ignore")


class SupplyChainOptimizer:
    def __init__(self, config):
        self.config = config
        self.optimization_results = {}

    def inventory_cost_function(self, params, demand_forecast, constraints):
        """
        Objective function for inventory optimization
        params: [reorder_point, order_quantity, safety_stock]
        """
        reorder_point, order_quantity, safety_stock = params

        # Simulate inventory levels
        inventory_levels = []
        stockouts = []
        holding_costs = []
        ordering_costs = []

        current_inventory = order_quantity + safety_stock

        for demand in demand_forecast:
            # Daily inventory dynamics
            current_inventory = max(0, current_inventory - demand)

            # Check for reorder
            if current_inventory <= reorder_point:
                current_inventory += order_quantity
                ordering_costs.append(constraints.get("fixed_order_cost", 50))
            else:
                ordering_costs.append(0)

            # Calculate costs
            if current_inventory <= 0:
                stockouts.append(abs(current_inventory))
                holding_costs.append(0)
            else:
                stockouts.append(0)
                holding_costs.append(
                    current_inventory * constraints.get("holding_cost_rate", 0.1)
                )

            inventory_levels.append(max(0, current_inventory))

        # Calculate total costs
        total_holding_cost = sum(holding_costs)
        total_ordering_cost = sum(ordering_costs)
        total_stockout_cost = sum(stockouts) * constraints.get("stockout_penalty", 10)

        total_cost = total_holding_cost + total_ordering_cost + total_stockout_cost

        # Add penalty for constraint violations
        penalty = 0
        service_level = 1 - (sum(s > 0 for s in stockouts) / len(stockouts))
        if service_level < constraints.get("min_service_level", 0.95):
            penalty += 1000 * (
                constraints.get("min_service_level", 0.95) - service_level
            )

        return total_cost + penalty

    def multi_objective_function(self, params, demand_forecast, constraints, weights):
        """
        Multi-objective function combining cost, service level, and inventory turnover
        """
        reorder_point, order_quantity, safety_stock = params

        # Simulate inventory
        results = self._simulate_inventory_policy(params, demand_forecast)

        # Calculate objectives
        total_cost = results["total_cost"]
        service_level = results["service_level"]
        inventory_turnover = results["inventory_turnover"]

        # Normalize objectives (assuming we want to minimize cost and maximize service level and turnover)
        normalized_cost = total_cost / constraints.get("cost_normalization", 10000)
        normalized_service = 1 - service_level  # Convert to minimization
        normalized_turnover = 1 / (inventory_turnover + 1e-8)  # Convert to minimization

        # Weighted sum
        objective = (
            weights.get("cost", 0.5) * normalized_cost
            + weights.get("service", 0.3) * normalized_service
            + weights.get("turnover", 0.2) * normalized_turnover
        )

        return objective

    def _simulate_inventory_policy(self, params, demand_forecast):
        """Simulate inventory policy and return performance metrics"""
        reorder_point, order_quantity, safety_stock = params

        inventory_levels = []
        stockouts = []
        orders = []

        current_inventory = order_quantity + safety_stock

        for demand in demand_forecast:
            current_inventory = max(0, current_inventory - demand)

            if current_inventory <= reorder_point:
                current_inventory += order_quantity
                orders.append(order_quantity)
            else:
                orders.append(0)

            if current_inventory <= 0:
                stockouts.append(1)
            else:
                stockouts.append(0)

            inventory_levels.append(max(0, current_inventory))

        # Calculate metrics
        avg_inventory = np.mean(inventory_levels)
        service_level = 1 - (sum(stockouts) / len(stockouts))
        inventory_turnover = sum(demand_forecast) / (avg_inventory + 1e-8)

        total_cost = (
            avg_inventory * 0.1 * len(demand_forecast)  # Holding cost
            + sum(orders) * 0.01  # Ordering cost
            + sum(stockouts) * np.mean(demand_forecast) * 10
        )  # Stockout cost

        return {
            "total_cost": total_cost,
            "service_level": service_level,
            "inventory_turnover": inventory_turnover,
            "avg_inventory": avg_inventory,
        }

    def optimize_inventory_policy(
        self, demand_forecast, method="scipy", constraints=None
    ):
        """Optimize inventory policy parameters"""
        if constraints is None:
            constraints = self.config.CONSTRAINTS

        # Parameter bounds: [reorder_point, order_quantity, safety_stock]
        bounds = [(10, 200), (50, 500), (10, 100)]

        if method == "scipy":
            result = minimize(
                self.inventory_cost_function,
                x0=[50, 100, 20],  # Initial guess
                args=(demand_forecast, constraints),
                bounds=bounds,
                method="L-BFGS-B",
            )

            optimal_params = result.x
            optimal_cost = result.fun

        elif method == "differential_evolution":
            result = differential_evolution(
                self.inventory_cost_function,
                bounds=bounds,
                args=(demand_forecast, constraints),
                seed=42,
            )

            optimal_params = result.x
            optimal_cost = result.fun

        elif method == "bayesian":
            optimal_params, optimal_cost = self._bayesian_optimization(
                demand_forecast, constraints, bounds
            )

        else:
            raise ValueError(f"Unknown optimization method: {method}")

        # Simulate optimal policy
        optimal_performance = self._simulate_inventory_policy(
            optimal_params, demand_forecast
        )

        return {
            "optimal_params": {
                "reorder_point": optimal_params[0],
                "order_quantity": optimal_params[1],
                "safety_stock": optimal_params[2],
            },
            "optimal_cost": optimal_cost,
            "performance_metrics": optimal_performance,
        }

    def _bayesian_optimization(
        self, demand_forecast, constraints, bounds, n_iterations=50
    ):
        """Bayesian optimization using Gaussian Process"""
        from sklearn.gaussian_process.kernels import RBF, Matern

        # Generate initial random samples
        n_initial = 10
        X_samples = []
        y_samples = []

        for _ in range(n_initial):
            params = [np.random.uniform(b[0], b[1]) for b in bounds]
            cost = self.inventory_cost_function(params, demand_forecast, constraints)
            X_samples.append(params)
            y_samples.append(cost)

        X_samples = np.array(X_samples)
        y_samples = np.array(y_samples)

        # Gaussian Process model
        kernel = Matern(length_scale=1.0, nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)

        best_params = X_samples[np.argmin(y_samples)]
        best_cost = np.min(y_samples)

        for i in range(n_iterations):
            # Fit GP
            gp.fit(X_samples, y_samples)

            # Acquisition function (Expected Improvement)
            def acquisition(x):
                x = np.array(x).reshape(1, -1)
                mu, sigma = gp.predict(x, return_std=True)
                with np.errstate(divide="warn"):
                    imp = best_cost - mu
                    Z = imp / (sigma + 1e-9)
                    ei = imp * stats.norm.cdf(Z) + sigma * stats.norm.pdf(Z)
                    ei[sigma == 0.0] = 0.0
                return -ei[0]  # Minimize negative EI

            # Optimize acquisition function
            result = minimize(
                acquisition, x0=best_params, bounds=bounds, method="L-BFGS-B"
            )

            # Evaluate new point
            new_params = result.x
            new_cost = self.inventory_cost_function(
                new_params, demand_forecast, constraints
            )

            # Update samples
            X_samples = np.vstack([X_samples, new_params])
            y_samples = np.append(y_samples, new_cost)

            # Update best
            if new_cost < best_cost:
                best_cost = new_cost
                best_params = new_params

        return best_params, best_cost

    def robust_optimization(self, demand_scenarios, confidence_level=0.95):
        """Robust optimization considering demand uncertainty"""
        scenario_results = []

        for i, scenario in enumerate(demand_scenarios):
            print(f"Optimizing for scenario {i + 1}/{len(demand_scenarios)}")
            result = self.optimize_inventory_policy(scenario, method="scipy")
            scenario_results.append(result)

        # Extract costs for each scenario
        costs = [r["optimal_cost"] for r in scenario_results]

        # Robust optimization: minimize worst-case cost
        worst_case_idx = np.argmax(costs)
        robust_solution = scenario_results[worst_case_idx]

        # Alternative: minimize CVaR (Conditional Value at Risk)
        sorted_costs = np.sort(costs)
        cvar_threshold = int((1 - confidence_level) * len(costs))
        cvar_costs = (
            sorted_costs[-cvar_threshold:] if cvar_threshold > 0 else [sorted_costs[-1]]
        )
        cvar = np.mean(cvar_costs)

        return {
            "robust_solution": robust_solution,
            "scenario_results": scenario_results,
            "cvar": cvar,
            "worst_case_cost": np.max(costs),
            "average_cost": np.mean(costs),
        }

    def dynamic_pricing_optimization(self, demand_model, cost_structure, price_bounds):
        """Optimize dynamic pricing strategy"""

        def pricing_objective(prices, demand_model, cost_structure):
            # Predict demand for given prices
            demands = demand_model.predict(np.array(prices).reshape(-1, 1))

            # Calculate revenues and costs
            revenues = prices * demands
            costs = demands * cost_structure["unit_cost"]

            # Total profit
            total_profit = np.sum(revenues - costs)

            return -total_profit  # Minimize negative profit

        # Optimize pricing
        result = minimize(
            pricing_objective,
            x0=np.mean(price_bounds) * np.ones(len(price_bounds)),
            args=(demand_model, cost_structure),
            bounds=price_bounds,
            method="L-BFGS-B",
        )

        optimal_prices = result.x
        optimal_profit = -result.fun

        return {"optimal_prices": optimal_prices, "optimal_profit": optimal_profit}

    def comprehensive_optimization(self, data, demand_forecast, scenarios=None):
        """Run comprehensive supply chain optimization"""
        results = {}

        # 1. Single-objective inventory optimization
        print("Running single-objective optimization...")
        single_obj_result = self.optimize_inventory_policy(
            demand_forecast, method="differential_evolution"
        )
        results["single_objective"] = single_obj_result

        # 2. Multi-objective optimization
        print("Running multi-objective optimization...")
        weights = {"cost": 0.4, "service": 0.4, "turnover": 0.2}

        bounds = [(10, 200), (50, 500), (10, 100)]
        multi_obj_result = minimize(
            self.multi_objective_function,
            x0=[50, 100, 20],
            args=(demand_forecast, self.config.CONSTRAINTS, weights),
            bounds=bounds,
            method="L-BFGS-B",
        )

        multi_obj_performance = self._simulate_inventory_policy(
            multi_obj_result.x, demand_forecast
        )

        results["multi_objective"] = {
            "optimal_params": {
                "reorder_point": multi_obj_result.x[0],
                "order_quantity": multi_obj_result.x[1],
                "safety_stock": multi_obj_result.x[2],
            },
            "performance_metrics": multi_obj_performance,
        }

        # 3. Robust optimization (if scenarios provided)
        if scenarios is not None:
            print("Running robust optimization...")
            robust_result = self.robust_optimization(scenarios)
            results["robust"] = robust_result

        # 4. Bayesian optimization
        print("Running Bayesian optimization...")
        bayesian_params, bayesian_cost = self._bayesian_optimization(
            demand_forecast, self.config.CONSTRAINTS, bounds, n_iterations=30
        )

        bayesian_performance = self._simulate_inventory_policy(
            bayesian_params, demand_forecast
        )

        results["bayesian"] = {
            "optimal_params": {
                "reorder_point": bayesian_params[0],
                "order_quantity": bayesian_params[1],
                "safety_stock": bayesian_params[2],
            },
            "optimal_cost": bayesian_cost,
            "performance_metrics": bayesian_performance,
        }

        self.optimization_results = results
        return results

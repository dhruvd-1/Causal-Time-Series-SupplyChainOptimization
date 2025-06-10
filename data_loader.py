"""
Data loader utilities for integrating real supply chain data
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import requests
import json
from datetime import datetime, timedelta


class DataLoader:
    def __init__(self, config):
        self.config = config

    def load_from_csv(self, file_path, date_col="date"):
        """Load data from CSV file"""
        data = pd.read_csv(file_path)
        data[date_col] = pd.to_datetime(data[date_col])
        return data

    def load_from_database(self, query, connection_string=None):
        """Load data from database"""
        if connection_string is None:
            connection_string = self.config.DATABASE_URL

        engine = create_engine(connection_string)
        data = pd.read_sql(query, engine)
        return data

    def load_from_api(self, api_endpoint, headers=None, params=None):
        """Load data from REST API"""
        response = requests.get(api_endpoint, headers=headers, params=params)
        if response.status_code == 200:
            data = pd.DataFrame(response.json())
            return data
        else:
            raise Exception(f"API request failed: {response.status_code}")

    def load_erp_data(self, erp_config):
        """Load data from ERP system (example for SAP/Oracle)"""
        # This would need to be customized based on your ERP system
        # Example structure:
        queries = {
            "sales_orders": """
                SELECT order_date, product_id, quantity, unit_price
                FROM sales_orders 
                WHERE order_date >= %s
            """,
            "inventory": """
                SELECT date, product_id, stock_level, location
                FROM inventory_levels
                WHERE date >= %s
            """,
            "suppliers": """
                SELECT delivery_date, supplier_id, lead_time, quality_score
                FROM supplier_performance
                WHERE delivery_date >= %s
            """,
        }

        data = {}
        for table_name, query in queries.items():
            data[table_name] = self.load_from_database(query)

        return data

    def validate_data_quality(self, data, required_columns):
        """Validate data quality and completeness"""
        issues = []

        # Check for required columns
        missing_cols = set(required_columns) - set(data.columns)
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")

        # Check for missing values
        missing_data = data.isnull().sum()
        if missing_data.any():
            issues.append(
                f"Missing values found: {missing_data[missing_data > 0].to_dict()}"
            )

        # Check for duplicates
        duplicates = data.duplicated().sum()
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate rows")

        # Check date continuity (if date column exists)
        if "date" in data.columns:
            date_gaps = self._check_date_gaps(data["date"])
            if date_gaps:
                issues.append(f"Date gaps found: {len(date_gaps)} gaps")

        return issues

    def _check_date_gaps(self, dates):
        """Check for gaps in date series"""
        dates_sorted = pd.to_datetime(dates).sort_values()
        date_range = pd.date_range(dates_sorted.min(), dates_sorted.max(), freq="D")
        missing_dates = set(date_range) - set(dates_sorted)
        return list(missing_dates)

#!/usr/bin/env python3
"""
Integration test for data validation system.

This script demonstrates the complete data validation pipeline
with sample diabetes data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from diabetes_lstm_pipeline.data_validation import DataValidator


def create_sample_diabetes_data():
    """Create sample diabetes data for testing."""
    np.random.seed(42)

    # Generate 1000 records over 5 days
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(minutes=5 * i) for i in range(1000)]

    # Create realistic diabetes data
    df = pd.DataFrame(
        {
            "EventDateTime": dates,
            "DeviceMode": np.random.choice(["Auto", "Manual"], 1000, p=[0.8, 0.2]),
            "BolusType": np.random.choice(
                ["Normal", "Extended", "Dual"], 1000, p=[0.7, 0.2, 0.1]
            ),
            "Basal": np.random.uniform(0.5, 2.5, 1000),
            "CorrectionDelivered": np.random.exponential(1.0, 1000),
            "TotalBolusInsulinDelivered": np.random.exponential(2.0, 1000),
            "FoodDelivered": np.random.exponential(10.0, 1000),
            "CarbSize": np.random.exponential(20.0, 1000),
            "CGM": np.random.normal(120, 30, 1000),  # Normal glucose around 120 mg/dL
        }
    )

    # Introduce some data quality issues for testing

    # Add missing values (5% missing in CGM)
    missing_indices = np.random.choice(1000, 50, replace=False)
    df.loc[missing_indices, "CGM"] = np.nan

    # Add some outliers
    outlier_indices = np.random.choice(1000, 10, replace=False)
    df.loc[outlier_indices, "CGM"] = np.random.choice(
        [500, 600, 20, 15], 10
    )  # Extreme values

    # Add some duplicates
    df.loc[100] = df.loc[99]
    df.loc[200] = df.loc[199]

    # Add some out-of-range values
    df.loc[50, "Basal"] = -0.5  # Negative basal rate
    df.loc[51, "CarbSize"] = 300  # Very high carb size

    return df


def main():
    """Run the integration test."""
    print("=" * 80)
    print("DIABETES LSTM PIPELINE - DATA VALIDATION INTEGRATION TEST")
    print("=" * 80)

    # Create sample data
    print("Creating sample diabetes data...")
    df = create_sample_diabetes_data()
    print(f"Created dataset with {len(df)} records and {len(df.columns)} columns")
    print(f"Date range: {df['EventDateTime'].min()} to {df['EventDateTime'].max()}")
    print()

    # Initialize validator
    config = {
        "report_output_dir": "test_reports",
        "random_state": 42,
        "zscore_threshold": 3.0,
        "contamination": 0.1,
    }

    validator = DataValidator(config)

    # Run comprehensive validation
    print("Running comprehensive data validation...")
    validation_result, quality_report, outlier_report = validator.validate_dataset(
        df, outlier_method="iqr", generate_report=True
    )

    # Display results summary
    print("\nVALIDATION RESULTS SUMMARY")
    print("-" * 40)
    print(f"Schema validation: {'PASSED' if validation_result.is_valid else 'FAILED'}")
    print(f"Quality score: {quality_report.quality_score:.2f}/100")
    print(f"Total records: {quality_report.total_records:,}")
    print(f"Duplicate records: {quality_report.duplicate_records}")
    print(f"Outliers detected: {len(outlier_report.outlier_indices)}")

    # Missing values summary
    print(f"\nMissing values by column:")
    for col, stats in quality_report.missing_value_stats.items():
        if stats["count"] > 0:
            print(f"  - {col}: {stats['count']} ({stats['percentage']:.1f}%)")

    # Outlier summary
    print(f"\nOutliers by column ({outlier_report.method} method):")
    for col, count in outlier_report.outlier_counts.items():
        if count > 0:
            percentage = outlier_report.outlier_percentages[col]
            print(f"  - {col}: {count} ({percentage:.1f}%)")

    # Recommendations
    if quality_report.recommendations:
        print(f"\nRecommendations:")
        for i, rec in enumerate(quality_report.recommendations, 1):
            print(f"  {i}. {rec}")

    # Test different outlier detection methods
    print(f"\nTesting different outlier detection methods:")
    methods = ["iqr", "zscore", "isolation_forest"]

    for method in methods:
        _, _, outlier_report_method = validator.validate_dataset(
            df, outlier_method=method, generate_report=False
        )
        print(
            f"  - {method}: {len(outlier_report_method.outlier_indices)} outliers detected"
        )

    print(f"\nIntegration test completed successfully!")
    print(
        f"Check the 'test_reports' directory for detailed validation reports and visualizations."
    )


if __name__ == "__main__":
    main()

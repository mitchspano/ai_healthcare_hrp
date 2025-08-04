"""
Demonstration of the clinical evaluation system for glucose prediction models.

This script shows how to use all components of the clinical evaluation system
to assess glucose prediction model performance using clinically relevant metrics.
"""

import numpy as np
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from diabetes_lstm_pipeline.evaluation import (
    ClinicalMetrics,
    ClarkeErrorGrid,
    ParkesErrorGrid,
    VisualizationGenerator,
)


def generate_realistic_test_data(n_samples: int = 1000) -> tuple:
    """
    Generate realistic glucose prediction test data.

    Args:
        n_samples: Number of samples to generate

    Returns:
        Tuple of (y_true, y_pred) arrays
    """
    np.random.seed(42)

    # Generate realistic glucose distribution
    # Mix of normal, hypoglycemic, and hyperglycemic values
    normal_samples = int(0.7 * n_samples)
    hypo_samples = int(0.15 * n_samples)
    hyper_samples = n_samples - normal_samples - hypo_samples

    # Normal range (70-180 mg/dL)
    normal_glucose = np.random.normal(120, 25, normal_samples)
    normal_glucose = np.clip(normal_glucose, 70, 180)

    # Hypoglycemic range (<70 mg/dL)
    hypo_glucose = np.random.normal(55, 10, hypo_samples)
    hypo_glucose = np.clip(hypo_glucose, 40, 69)

    # Hyperglycemic range (>180 mg/dL)
    hyper_glucose = np.random.normal(250, 40, hyper_samples)
    hyper_glucose = np.clip(hyper_glucose, 181, 400)

    # Combine all glucose values
    y_true = np.concatenate([normal_glucose, hypo_glucose, hyper_glucose])
    np.random.shuffle(y_true)

    # Generate predictions with realistic errors
    # MARD typically 10-15% for good CGM systems
    relative_errors = np.random.normal(0, 0.12, n_samples)  # 12% std dev
    absolute_errors = y_true * relative_errors

    # Add some systematic bias and noise
    bias = np.random.normal(2, 5, n_samples)  # Small positive bias
    y_pred = y_true + absolute_errors + bias

    # Ensure predictions are in realistic range
    y_pred = np.clip(y_pred, 40, 400)

    return y_true, y_pred


def main():
    """Demonstrate the clinical evaluation system."""
    print("Clinical Evaluation System Demonstration")
    print("=" * 50)

    # Generate test data
    print("\n1. Generating realistic test data...")
    y_true, y_pred = generate_realistic_test_data(1000)
    print(f"   Generated {len(y_true)} glucose prediction pairs")
    print(f"   True glucose range: {y_true.min():.1f} - {y_true.max():.1f} mg/dL")
    print(f"   Predicted glucose range: {y_pred.min():.1f} - {y_pred.max():.1f} mg/dL")

    # Initialize clinical evaluation components
    print("\n2. Initializing clinical evaluation components...")
    clinical_metrics = ClinicalMetrics()
    clarke_grid = ClarkeErrorGrid()
    parkes_grid = ParkesErrorGrid(grid_type="type1")
    viz_generator = VisualizationGenerator(output_dir="demo_reports")

    # Calculate basic clinical metrics
    print("\n3. Calculating basic clinical metrics...")
    basic_metrics = clinical_metrics.calculate_all_metrics(y_true, y_pred)

    print(f"   MARD: {basic_metrics['mard']:.2f}%")
    print(f"   MAE: {basic_metrics['mae']:.2f} mg/dL")
    print(f"   RMSE: {basic_metrics['rmse']:.2f} mg/dL")
    print(f"   Time-in-Range Accuracy: {basic_metrics['tir_prediction_accuracy']:.1f}%")
    print(
        f"   Hypoglycemia Detection Sensitivity: {basic_metrics['hypoglycemia_sensitivity']:.1f}%"
    )
    print(
        f"   Hyperglycemia Detection Sensitivity: {basic_metrics['hyperglycemia_sensitivity']:.1f}%"
    )

    # Perform Clarke Error Grid Analysis
    print("\n4. Performing Clarke Error Grid Analysis...")
    clarke_results = clarke_grid.analyze(y_true, y_pred)

    print(
        f"   Zone A (Clinically Accurate): {clarke_results['clarke_zone_a_percent']:.1f}%"
    )
    print(f"   Zone B (Benign Errors): {clarke_results['clarke_zone_b_percent']:.1f}%")
    print(f"   Zone C (Overcorrection): {clarke_results['clarke_zone_c_percent']:.1f}%")
    print(f"   Zone D (Dangerous): {clarke_results['clarke_zone_d_percent']:.1f}%")
    print(f"   Zone E (Erroneous): {clarke_results['clarke_zone_e_percent']:.1f}%")
    print(
        f"   Clinically Acceptable (A+B): {clarke_results['clarke_clinically_acceptable_percent']:.1f}%"
    )

    # Clinical interpretation
    clarke_interpretation = clarke_grid.get_clinical_interpretation(clarke_results)
    print(f"   Clinical Interpretation: {clarke_interpretation}")

    # Perform Parkes Error Grid Analysis
    print("\n5. Performing Parkes Error Grid Analysis...")
    parkes_results = parkes_grid.analyze(y_true, y_pred)

    print(
        f"   Zone A (Clinically Accurate): {parkes_results['parkes_zone_a_percent']:.1f}%"
    )
    print(f"   Zone B (Benign Errors): {parkes_results['parkes_zone_b_percent']:.1f}%")
    print(f"   Zone C (Overcorrection): {parkes_results['parkes_zone_c_percent']:.1f}%")
    print(f"   Zone D (Dangerous): {parkes_results['parkes_zone_d_percent']:.1f}%")
    print(f"   Zone E (Erroneous): {parkes_results['parkes_zone_e_percent']:.1f}%")
    print(
        f"   Clinically Acceptable (A+B): {parkes_results['parkes_clinically_acceptable_percent']:.1f}%"
    )

    # Clinical interpretation
    parkes_interpretation = parkes_grid.get_clinical_interpretation(parkes_results)
    print(f"   Clinical Interpretation: {parkes_interpretation}")

    # Compare Clarke and Parkes results
    print("\n6. Comparing Clarke and Parkes Error Grids...")
    comparison = parkes_grid.compare_with_clarke(parkes_results, clarke_results)
    print(f"   Comparison: {comparison}")

    # Generate visualizations
    print("\n7. Generating clinical evaluation visualizations...")
    try:
        # Create individual plots
        scatter_plot = viz_generator.plot_prediction_scatter(y_true, y_pred)
        print(f"   Prediction scatter plot: {scatter_plot}")

        residual_plot = viz_generator.plot_residuals(y_true, y_pred)
        print(f"   Residual analysis plot: {residual_plot}")

        clarke_plot = viz_generator.plot_clarke_error_grid(y_true, y_pred)
        print(f"   Clarke Error Grid plot: {clarke_plot}")

        parkes_plot = viz_generator.plot_parkes_error_grid(y_true, y_pred)
        print(f"   Parkes Error Grid plot: {parkes_plot}")

        # Create comprehensive summary
        summary_plot = viz_generator.plot_clinical_metrics_summary(
            {**basic_metrics, **clarke_results, **parkes_results}
        )
        print(f"   Clinical metrics summary: {summary_plot}")

        # Generate comprehensive report
        report_path = viz_generator.generate_evaluation_report(
            y_true, y_pred, basic_metrics, clarke_results, parkes_results
        )
        print(f"   Comprehensive evaluation report: {report_path}")

    except ImportError as e:
        print(f"   Visualization skipped (missing dependencies): {e}")
    except Exception as e:
        print(f"   Visualization error: {e}")

    # Summary assessment
    print("\n8. Overall Clinical Assessment:")
    print("   " + "=" * 40)

    mard = basic_metrics["mard"]
    clarke_acceptable = clarke_results["clarke_clinically_acceptable_percent"]
    parkes_acceptable = parkes_results["parkes_clinically_acceptable_percent"]

    # MARD assessment
    if mard <= 10:
        mard_assessment = "Excellent"
    elif mard <= 15:
        mard_assessment = "Good"
    elif mard <= 20:
        mard_assessment = "Acceptable"
    else:
        mard_assessment = "Poor"

    # Clinical acceptability assessment
    if clarke_acceptable >= 95:
        clinical_assessment = "Excellent"
    elif clarke_acceptable >= 90:
        clinical_assessment = "Good"
    elif clarke_acceptable >= 80:
        clinical_assessment = "Acceptable"
    else:
        clinical_assessment = "Poor"

    print(f"   MARD Performance: {mard_assessment} ({mard:.2f}%)")
    print(
        f"   Clinical Acceptability: {clinical_assessment} ({clarke_acceptable:.1f}% in zones A+B)"
    )
    print(
        f"   Safety Profile: {clarke_results['clarke_dangerous_errors_percent']:.1f}% dangerous errors"
    )

    # Recommendations
    print("\n9. Clinical Recommendations:")
    if mard > 15:
        print("   - Consider model retraining to improve MARD performance")
    if clarke_acceptable < 90:
        print("   - Clinical acceptability below recommended threshold")
    if clarke_results["clarke_dangerous_errors_percent"] > 5:
        print("   - High rate of dangerous errors - safety concerns")
    if parkes_acceptable < clarke_acceptable - 5:
        print("   - Parkes grid shows more stringent assessment - review edge cases")

    if (
        mard <= 15
        and clarke_acceptable >= 90
        and clarke_results["clarke_dangerous_errors_percent"] <= 5
    ):
        print("   - Model meets clinical performance standards")
        print("   - Suitable for clinical validation studies")

    print("\nClinical evaluation demonstration completed!")


if __name__ == "__main__":
    main()

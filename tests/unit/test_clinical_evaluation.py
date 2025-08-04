"""Unit tests for clinical evaluation metrics and error grid analyses."""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import os
from pathlib import Path

from diabetes_lstm_pipeline.evaluation.clinical_metrics import ClinicalMetrics
from diabetes_lstm_pipeline.evaluation.clarke_error_grid import ClarkeErrorGrid
from diabetes_lstm_pipeline.evaluation.parkes_error_grid import ParkesErrorGrid
from diabetes_lstm_pipeline.evaluation.visualization_generator import (
    VisualizationGenerator,
)


class TestClinicalMetrics(unittest.TestCase):
    """Test cases for ClinicalMetrics class."""

    def setUp(self):
        """Set up test fixtures."""
        self.metrics = ClinicalMetrics()

        # Create test data with known characteristics
        np.random.seed(42)
        self.y_true = np.array([100, 120, 80, 150, 200, 60, 180, 90, 110, 140])
        self.y_pred = np.array([105, 115, 85, 145, 195, 65, 175, 95, 115, 135])

    def test_calculate_mard(self):
        """Test MARD calculation."""
        mard = self.metrics.calculate_mard(self.y_true, self.y_pred)

        # MARD should be positive
        self.assertGreater(mard, 0)
        self.assertLess(mard, 100)  # Should be reasonable percentage

        # Test with perfect predictions
        perfect_mard = self.metrics.calculate_mard(self.y_true, self.y_true)
        self.assertEqual(perfect_mard, 0.0)

    def test_mard_edge_cases(self):
        """Test MARD calculation edge cases."""
        # Test with zero reference values
        with self.assertRaises(ValueError):
            self.metrics.calculate_mard(np.array([0, 0]), np.array([100, 100]))

        # Test with empty arrays
        with self.assertRaises(ValueError):
            self.metrics.calculate_mard(np.array([]), np.array([]))

        # Test with mismatched lengths
        with self.assertRaises(ValueError):
            self.metrics.calculate_mard(np.array([100]), np.array([100, 110]))

    def test_calculate_mae_rmse(self):
        """Test MAE and RMSE calculations."""
        mae = self.metrics.calculate_mae(self.y_true, self.y_pred)
        rmse = self.metrics.calculate_rmse(self.y_true, self.y_pred)

        # MAE and RMSE should be positive
        self.assertGreater(mae, 0)
        self.assertGreater(rmse, 0)

        # RMSE should be >= MAE
        self.assertGreaterEqual(rmse, mae)

        # Test with perfect predictions
        perfect_mae = self.metrics.calculate_mae(self.y_true, self.y_true)
        perfect_rmse = self.metrics.calculate_rmse(self.y_true, self.y_true)
        self.assertEqual(perfect_mae, 0.0)
        self.assertEqual(perfect_rmse, 0.0)

    def test_time_in_range_accuracy(self):
        """Test time-in-range accuracy calculation."""
        tir_metrics = self.metrics.calculate_time_in_range_accuracy(
            self.y_true, self.y_pred
        )

        # Check required keys
        required_keys = [
            "true_tir_percent",
            "predicted_tir_percent",
            "tir_prediction_accuracy",
            "tir_sensitivity",
            "tir_specificity",
            "tir_absolute_error",
        ]
        for key in required_keys:
            self.assertIn(key, tir_metrics)

        # Check value ranges
        for key in required_keys[:-1]:  # All except absolute_error
            self.assertGreaterEqual(tir_metrics[key], 0)
            self.assertLessEqual(tir_metrics[key], 100)

    def test_hypoglycemia_detection(self):
        """Test hypoglycemia detection metrics."""
        # Create data with known hypoglycemic events
        y_true_hypo = np.array([65, 120, 50, 150, 45, 180, 60, 90])
        y_pred_hypo = np.array([70, 115, 55, 145, 50, 175, 65, 95])

        hypo_metrics = self.metrics.detect_hypoglycemia_events(y_true_hypo, y_pred_hypo)

        # Check required keys
        required_keys = [
            "hypoglycemia_sensitivity",
            "hypoglycemia_specificity",
            "hypoglycemia_precision",
            "hypoglycemia_accuracy",
            "hypoglycemia_f1_score",
            "true_hypoglycemia_rate",
            "predicted_hypoglycemia_rate",
        ]
        for key in required_keys:
            self.assertIn(key, hypo_metrics)

        # Check value ranges
        for key in required_keys:
            self.assertGreaterEqual(hypo_metrics[key], 0)
            self.assertLessEqual(hypo_metrics[key], 100)

    def test_hyperglycemia_detection(self):
        """Test hyperglycemia detection metrics."""
        # Create data with known hyperglycemic events
        y_true_hyper = np.array([300, 120, 280, 150, 350, 180, 260, 90])
        y_pred_hyper = np.array([295, 115, 275, 145, 340, 175, 255, 95])

        hyper_metrics = self.metrics.detect_hyperglycemia_events(
            y_true_hyper, y_pred_hyper
        )

        # Check required keys
        required_keys = [
            "hyperglycemia_sensitivity",
            "hyperglycemia_specificity",
            "hyperglycemia_precision",
            "hyperglycemia_accuracy",
            "hyperglycemia_f1_score",
            "true_hyperglycemia_rate",
            "predicted_hyperglycemia_rate",
        ]
        for key in required_keys:
            self.assertIn(key, hyper_metrics)

        # Check value ranges
        for key in required_keys:
            self.assertGreaterEqual(hyper_metrics[key], 0)
            self.assertLessEqual(hyper_metrics[key], 100)

    def test_calculate_all_metrics(self):
        """Test comprehensive metrics calculation."""
        all_metrics = self.metrics.calculate_all_metrics(self.y_true, self.y_pred)

        # Check that all expected metrics are present
        expected_metrics = [
            "mard",
            "mae",
            "rmse",
            "true_tir_percent",
            "hypoglycemia_sensitivity",
            "hyperglycemia_sensitivity",
        ]
        for metric in expected_metrics:
            self.assertIn(metric, all_metrics)


class TestClarkeErrorGrid(unittest.TestCase):
    """Test cases for ClarkeErrorGrid class."""

    def setUp(self):
        """Set up test fixtures."""
        self.clarke = ClarkeErrorGrid()

        # Create test data with known zone classifications
        self.y_true = np.array([100, 120, 80, 150, 200, 60, 180, 90, 110, 140])
        self.y_pred = np.array([105, 115, 85, 145, 195, 65, 175, 95, 115, 135])

    def test_zone_classification(self):
        """Test individual zone classification."""
        # Test Zone A (clinically accurate)
        zone_a = self.clarke._detailed_zone_classification(100, 105)  # 5% error
        self.assertEqual(zone_a, "A")

        # Test hypoglycemic range
        zone_hypo = self.clarke._detailed_zone_classification(65, 68)
        self.assertEqual(zone_hypo, "A")

    def test_analyze(self):
        """Test Clarke Error Grid analysis."""
        results = self.clarke.analyze(self.y_true, self.y_pred)

        # Check required keys
        required_keys = [
            "clarke_zone_a_percent",
            "clarke_zone_b_percent",
            "clarke_zone_c_percent",
            "clarke_zone_d_percent",
            "clarke_zone_e_percent",
            "clarke_clinically_acceptable_percent",
            "clarke_dangerous_errors_percent",
            "clarke_total_points",
        ]
        for key in required_keys:
            self.assertIn(key, results)

        # Check that percentages sum to 100
        zone_sum = (
            results["clarke_zone_a_percent"]
            + results["clarke_zone_b_percent"]
            + results["clarke_zone_c_percent"]
            + results["clarke_zone_d_percent"]
            + results["clarke_zone_e_percent"]
        )
        self.assertAlmostEqual(zone_sum, 100.0, places=1)

    def test_analyze_edge_cases(self):
        """Test Clarke Error Grid analysis edge cases."""
        # Test with empty arrays
        with self.assertRaises(ValueError):
            self.clarke.analyze(np.array([]), np.array([]))

        # Test with mismatched lengths
        with self.assertRaises(ValueError):
            self.clarke.analyze(np.array([100]), np.array([100, 110]))

        # Test with invalid values
        with self.assertRaises(ValueError):
            self.clarke.analyze(np.array([0, -10]), np.array([100, 110]))

    def test_get_zone_boundaries(self):
        """Test zone boundary descriptions."""
        boundaries = self.clarke.get_zone_boundaries()

        # Check that all zones are described
        for zone in ["A", "B", "C", "D", "E"]:
            self.assertIn(zone, boundaries)
            self.assertIsInstance(boundaries[zone], str)

    def test_clinical_interpretation(self):
        """Test clinical interpretation generation."""
        results = self.clarke.analyze(self.y_true, self.y_pred)
        interpretation = self.clarke.get_clinical_interpretation(results)

        self.assertIsInstance(interpretation, str)
        self.assertGreater(len(interpretation), 0)


class TestParkesErrorGrid(unittest.TestCase):
    """Test cases for ParkesErrorGrid class."""

    def setUp(self):
        """Set up test fixtures."""
        self.parkes_type1 = ParkesErrorGrid(grid_type="type1")
        self.parkes_type2 = ParkesErrorGrid(grid_type="type2")

        # Create test data
        self.y_true = np.array([100, 120, 80, 150, 200, 60, 180, 90, 110, 140])
        self.y_pred = np.array([105, 115, 85, 145, 195, 65, 175, 95, 115, 135])

    def test_initialization(self):
        """Test Parkes Error Grid initialization."""
        # Test valid grid types
        parkes1 = ParkesErrorGrid("type1")
        parkes2 = ParkesErrorGrid("type2")
        self.assertEqual(parkes1.grid_type, "type1")
        self.assertEqual(parkes2.grid_type, "type2")

        # Test invalid grid type
        with self.assertRaises(ValueError):
            ParkesErrorGrid("invalid")

    def test_analyze_type1(self):
        """Test Parkes Error Grid analysis for Type 1 diabetes."""
        results = self.parkes_type1.analyze(self.y_true, self.y_pred)

        # Check required keys
        required_keys = [
            "parkes_zone_a_percent",
            "parkes_zone_b_percent",
            "parkes_zone_c_percent",
            "parkes_zone_d_percent",
            "parkes_zone_e_percent",
            "parkes_clinically_acceptable_percent",
            "parkes_dangerous_errors_percent",
            "parkes_total_points",
            "parkes_grid_type",
        ]
        for key in required_keys:
            self.assertIn(key, results)

        # Check grid type
        self.assertEqual(results["parkes_grid_type"], "type1")

        # Check that percentages sum to 100
        zone_sum = (
            results["parkes_zone_a_percent"]
            + results["parkes_zone_b_percent"]
            + results["parkes_zone_c_percent"]
            + results["parkes_zone_d_percent"]
            + results["parkes_zone_e_percent"]
        )
        self.assertAlmostEqual(zone_sum, 100.0, places=1)

    def test_analyze_type2(self):
        """Test Parkes Error Grid analysis for Type 2 diabetes."""
        results = self.parkes_type2.analyze(self.y_true, self.y_pred)

        # Check grid type
        self.assertEqual(results["parkes_grid_type"], "type2")

    def test_zone_descriptions(self):
        """Test zone descriptions."""
        descriptions = self.parkes_type1.get_zone_descriptions()

        # Check that all zones are described
        for zone in ["A", "B", "C", "D", "E"]:
            self.assertIn(zone, descriptions)
            self.assertIsInstance(descriptions[zone], str)

    def test_clinical_interpretation(self):
        """Test clinical interpretation generation."""
        results = self.parkes_type1.analyze(self.y_true, self.y_pred)
        interpretation = self.parkes_type1.get_clinical_interpretation(results)

        self.assertIsInstance(interpretation, str)
        self.assertGreater(len(interpretation), 0)
        self.assertIn("TYPE1", interpretation)

    def test_compare_with_clarke(self):
        """Test comparison with Clarke Error Grid."""
        parkes_results = self.parkes_type1.analyze(self.y_true, self.y_pred)

        # Mock Clarke results
        clarke_results = {
            "clarke_clinically_acceptable_percent": 85.0,
            "clarke_zone_a_percent": 70.0,
        }

        comparison = self.parkes_type1.compare_with_clarke(
            parkes_results, clarke_results
        )

        self.assertIsInstance(comparison, str)
        self.assertGreater(len(comparison), 0)


class TestVisualizationGenerator(unittest.TestCase):
    """Test cases for VisualizationGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.viz_gen = VisualizationGenerator(output_dir=self.temp_dir)

        # Create test data
        np.random.seed(42)
        self.y_true = np.random.uniform(70, 200, 100)
        self.y_pred = self.y_true + np.random.normal(0, 10, 100)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_plot_clarke_error_grid(self, mock_close, mock_savefig):
        """Test Clarke Error Grid plotting."""
        plot_path = self.viz_gen.plot_clarke_error_grid(self.y_true, self.y_pred)

        self.assertIsInstance(plot_path, str)
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_plot_parkes_error_grid(self, mock_close, mock_savefig):
        """Test Parkes Error Grid plotting."""
        plot_path = self.viz_gen.plot_parkes_error_grid(self.y_true, self.y_pred)

        self.assertIsInstance(plot_path, str)
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_plot_prediction_scatter(self, mock_close, mock_savefig):
        """Test prediction scatter plot."""
        plot_path = self.viz_gen.plot_prediction_scatter(self.y_true, self.y_pred)

        self.assertIsInstance(plot_path, str)
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_plot_residuals(self, mock_close, mock_savefig):
        """Test residual plot."""
        plot_path = self.viz_gen.plot_residuals(self.y_true, self.y_pred)

        self.assertIsInstance(plot_path, str)
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_plot_clinical_metrics_summary(self, mock_close, mock_savefig):
        """Test clinical metrics summary plot."""
        metrics = {
            "mard": 12.5,
            "mae": 15.2,
            "rmse": 18.7,
            "clarke_zone_a_percent": 75.0,
            "clarke_zone_b_percent": 20.0,
            "hypoglycemia_sensitivity": 85.0,
        }

        plot_path = self.viz_gen.plot_clinical_metrics_summary(metrics)

        self.assertIsInstance(plot_path, str)
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_generate_evaluation_report(self, mock_close, mock_savefig):
        """Test comprehensive evaluation report generation."""
        clinical_metrics = {"mard": 12.5, "mae": 15.2}
        clarke_results = {"clarke_zone_a_percent": 75.0}
        parkes_results = {"parkes_zone_a_percent": 70.0}

        report_path = self.viz_gen.generate_evaluation_report(
            self.y_true, self.y_pred, clinical_metrics, clarke_results, parkes_results
        )

        self.assertIsInstance(report_path, str)
        self.assertTrue(os.path.exists(report_path))

        # Check that report file contains expected content
        with open(report_path, "r") as f:
            content = f.read()
            self.assertIn("CLINICAL EVALUATION REPORT", content)
            self.assertIn("MARD", content)
            self.assertIn("CLARKE ERROR GRID", content)


class TestClinicalMetricsIntegration(unittest.TestCase):
    """Integration tests for clinical evaluation system."""

    def setUp(self):
        """Set up integration test fixtures."""
        # Create realistic test data
        np.random.seed(42)
        n_samples = 1000

        # Generate realistic glucose values
        self.y_true = np.random.normal(120, 30, n_samples)
        self.y_true = np.clip(self.y_true, 40, 400)  # Realistic glucose range

        # Add realistic prediction errors
        errors = np.random.normal(0, 15, n_samples)  # ~12-15% MARD typical
        self.y_pred = self.y_true + errors
        self.y_pred = np.clip(self.y_pred, 40, 400)

    def test_full_clinical_evaluation_pipeline(self):
        """Test complete clinical evaluation pipeline."""
        # Initialize all components
        clinical_metrics = ClinicalMetrics()
        clarke_grid = ClarkeErrorGrid()
        parkes_grid = ParkesErrorGrid("type1")

        # Calculate all metrics
        basic_metrics = clinical_metrics.calculate_all_metrics(self.y_true, self.y_pred)
        clarke_results = clarke_grid.analyze(self.y_true, self.y_pred)
        parkes_results = parkes_grid.analyze(self.y_true, self.y_pred)

        # Verify realistic results
        self.assertGreater(basic_metrics["mard"], 5)  # Should have some error
        self.assertLess(basic_metrics["mard"], 25)  # But not too much

        # Clarke grid should have majority in zones A+B
        clarke_acceptable = clarke_results["clarke_clinically_acceptable_percent"]
        self.assertGreater(clarke_acceptable, 70)

        # Parkes grid should be more stringent
        parkes_acceptable = parkes_results["parkes_clinically_acceptable_percent"]
        self.assertLessEqual(parkes_acceptable, clarke_acceptable)

    def test_reference_implementation_comparison(self):
        """Test against known reference values for clinical metrics."""
        # Create test case with known MARD
        y_true_ref = np.array([100, 150, 200, 80, 120])
        y_pred_ref = np.array([110, 140, 190, 85, 125])

        clinical_metrics = ClinicalMetrics()
        mard = clinical_metrics.calculate_mard(y_true_ref, y_pred_ref)

        # Calculate expected MARD manually
        expected_mard = (
            np.mean(
                [
                    abs(110 - 100) / 100,
                    abs(140 - 150) / 150,
                    abs(190 - 200) / 200,
                    abs(85 - 80) / 80,
                    abs(125 - 120) / 120,
                ]
            )
            * 100
        )

        self.assertAlmostEqual(mard, expected_mard, places=2)


if __name__ == "__main__":
    unittest.main()

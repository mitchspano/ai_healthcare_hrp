"""Parkes Error Grid Analysis for glucose prediction evaluation."""

import numpy as np
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


class ParkesErrorGrid:
    """
    Implements Parkes (Consensus) Error Grid Analysis for glucose prediction evaluation.

    The Parkes Error Grid is an updated version of the Clarke Error Grid that provides
    more stringent clinical risk assessment for glucose monitoring systems.
    """

    def __init__(self, grid_type: str = "type1"):
        """
        Initialize Parkes Error Grid analyzer.

        Args:
            grid_type: Type of diabetes ('type1' or 'type2')
        """
        self.grid_type = grid_type.lower()
        if self.grid_type not in ["type1", "type2"]:
            raise ValueError("grid_type must be 'type1' or 'type2'")

    def _get_zone_boundaries_type1(self) -> Dict[str, List[Tuple[float, float]]]:
        """
        Get zone boundary coordinates for Type 1 diabetes Parkes Error Grid.

        Returns:
            Dictionary with zone boundary coordinates
        """
        # These are the mathematical boundaries for the Parkes Error Grid zones
        # Based on the original Parkes et al. publication
        boundaries = {
            "A": [
                # Zone A boundaries (most restrictive, clinically accurate)
                (0, 0),
                (50, 50),
                (170, 170),
                (385, 385),
                (550, 550),
            ],
            "B": [
                # Zone B boundaries (benign errors)
                (0, 0),
                (120, 120),
                (260, 260),
                (550, 550),
            ],
            "C": [
                # Zone C boundaries (overcorrection errors)
                (0, 0),
                (80, 80),
                (200, 200),
                (550, 550),
            ],
            "D": [
                # Zone D boundaries (dangerous errors)
                (0, 0),
                (70, 70),
                (180, 180),
                (550, 550),
            ],
            "E": [
                # Zone E boundaries (erroneous treatment)
                (0, 0),
                (60, 60),
                (160, 160),
                (550, 550),
            ],
        }
        return boundaries

    def _classify_zone_type1(self, reference: float, predicted: float) -> str:
        """
        Classify a glucose prediction pair for Type 1 diabetes Parkes Error Grid.

        Args:
            reference: Reference glucose value (mg/dL)
            predicted: Predicted glucose value (mg/dL)

        Returns:
            Zone classification ('A', 'B', 'C', 'D', or 'E')
        """
        ref = float(reference)
        pred = float(predicted)

        # Zone A: Clinically accurate
        # More stringent than Clarke - within 15% or both < 70
        if ref <= 70:
            if pred <= 70:
                return "A"
            elif pred <= 110:
                return "B"
            elif pred <= 180:
                return "C"
            else:
                return "E"

        elif ref <= 180:
            if abs(pred - ref) <= 0.15 * ref:
                return "A"
            elif abs(pred - ref) <= 0.20 * ref:
                return "B"
            elif pred > ref:
                if pred <= 250:
                    return "C"
                else:
                    return "E"
            else:  # pred < ref
                if pred >= 70:
                    return "B"
                elif pred >= 56:
                    return "D"
                else:
                    return "E"

        elif ref <= 240:
            if abs(pred - ref) <= 0.15 * ref:
                return "A"
            elif pred >= 0.85 * ref and pred <= 1.15 * ref:
                return "B"
            elif pred > ref:
                return "C"
            else:  # pred < ref
                if pred >= 70:
                    return "B"
                else:
                    return "D"

        else:  # ref > 240
            if abs(pred - ref) <= 0.15 * ref:
                return "A"
            elif pred >= 0.85 * ref:
                return "B"
            elif pred < 0.85 * ref and pred >= 70:
                return "C"
            else:
                return "D"

    def _classify_zone_type2(self, reference: float, predicted: float) -> str:
        """
        Classify a glucose prediction pair for Type 2 diabetes Parkes Error Grid.

        Args:
            reference: Reference glucose value (mg/dL)
            predicted: Predicted glucose value (mg/dL)

        Returns:
            Zone classification ('A', 'B', 'C', 'D', or 'E')
        """
        ref = float(reference)
        pred = float(predicted)

        # Type 2 diabetes has slightly different thresholds
        # Generally more lenient than Type 1

        if ref <= 70:
            if pred <= 70:
                return "A"
            elif pred <= 120:
                return "B"
            elif pred <= 200:
                return "C"
            else:
                return "E"

        elif ref <= 180:
            if abs(pred - ref) <= 0.20 * ref:
                return "A"
            elif abs(pred - ref) <= 0.25 * ref:
                return "B"
            elif pred > ref:
                if pred <= 300:
                    return "C"
                else:
                    return "E"
            else:  # pred < ref
                if pred >= 70:
                    return "B"
                elif pred >= 50:
                    return "D"
                else:
                    return "E"

        elif ref <= 300:
            if abs(pred - ref) <= 0.20 * ref:
                return "A"
            elif pred >= 0.80 * ref and pred <= 1.20 * ref:
                return "B"
            elif pred > ref:
                return "C"
            else:  # pred < ref
                if pred >= 70:
                    return "B"
                else:
                    return "D"

        else:  # ref > 300
            if abs(pred - ref) <= 0.20 * ref:
                return "A"
            elif pred >= 0.80 * ref:
                return "B"
            elif pred < 0.80 * ref and pred >= 70:
                return "C"
            else:
                return "D"

    def analyze(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Perform Parkes Error Grid Analysis on glucose predictions.

        Args:
            y_true: True glucose values (mg/dL)
            y_pred: Predicted glucose values (mg/dL)

        Returns:
            Dictionary with zone percentages and clinical interpretation
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")

        if len(y_true) == 0:
            raise ValueError("Input arrays cannot be empty")

        # Remove invalid values
        valid_mask = (y_true > 0) & (y_pred > 0)
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]

        if len(y_true_valid) == 0:
            raise ValueError("No valid glucose value pairs found")

        # Choose classification method based on diabetes type
        classify_func = (
            self._classify_zone_type1
            if self.grid_type == "type1"
            else self._classify_zone_type2
        )

        # Classify each prediction pair
        zones = []
        for true_val, pred_val in zip(y_true_valid, y_pred_valid):
            zone = classify_func(true_val, pred_val)
            zones.append(zone)

        # Count occurrences of each zone
        zone_counts = {zone: zones.count(zone) for zone in ["A", "B", "C", "D", "E"]}
        total_points = len(zones)

        # Calculate percentages
        zone_percentages = {
            f"parkes_zone_{zone.lower()}_percent": (count / total_points) * 100
            for zone, count in zone_counts.items()
        }

        # Add clinical interpretation metrics
        clinically_acceptable = zone_counts["A"] + zone_counts["B"]
        clinically_acceptable_percent = (clinically_acceptable / total_points) * 100

        dangerous_errors = zone_counts["D"] + zone_counts["E"]
        dangerous_errors_percent = (dangerous_errors / total_points) * 100

        results = {
            **zone_percentages,
            "parkes_clinically_acceptable_percent": clinically_acceptable_percent,
            "parkes_dangerous_errors_percent": dangerous_errors_percent,
            "parkes_total_points": total_points,
            "parkes_grid_type": self.grid_type,
        }

        logger.info(f"Parkes Error Grid Analysis ({self.grid_type}) completed:")
        logger.info(
            f"  Zone A (Clinically Accurate): {zone_percentages['parkes_zone_a_percent']:.1f}%"
        )
        logger.info(
            f"  Zone B (Benign Errors): {zone_percentages['parkes_zone_b_percent']:.1f}%"
        )
        logger.info(
            f"  Zone C (Overcorrection): {zone_percentages['parkes_zone_c_percent']:.1f}%"
        )
        logger.info(
            f"  Zone D (Dangerous): {zone_percentages['parkes_zone_d_percent']:.1f}%"
        )
        logger.info(
            f"  Zone E (Erroneous): {zone_percentages['parkes_zone_e_percent']:.1f}%"
        )
        logger.info(
            f"  Clinically Acceptable (A+B): {clinically_acceptable_percent:.1f}%"
        )

        return results

    def get_zone_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of Parkes Error Grid zones.

        Returns:
            Dictionary with zone descriptions
        """
        return {
            "A": "Clinically accurate: No effect on clinical action",
            "B": "Benign errors: Little or no effect on clinical outcome",
            "C": "Overcorrection errors: Likely to affect clinical action but unlikely to affect clinical outcome",
            "D": "Dangerous errors: Likely to affect clinical action and likely to affect clinical outcome",
            "E": "Erroneous treatment: Likely to affect clinical action and could have significant medical risk",
        }

    def get_clinical_interpretation(self, results: Dict[str, float]) -> str:
        """
        Generate clinical interpretation of Parkes Error Grid results.

        Args:
            results: Results from analyze() method

        Returns:
            Clinical interpretation string
        """
        zone_a_percent = results["parkes_zone_a_percent"]
        clinically_acceptable = results["parkes_clinically_acceptable_percent"]
        dangerous_errors = results["parkes_dangerous_errors_percent"]
        grid_type = results["parkes_grid_type"]

        interpretation = []

        # Parkes Error Grid has more stringent requirements than Clarke
        if clinically_acceptable >= 99:
            interpretation.append("Excellent clinical accuracy (≥99% in zones A+B)")
        elif clinically_acceptable >= 95:
            interpretation.append("Very good clinical accuracy (≥95% in zones A+B)")
        elif clinically_acceptable >= 90:
            interpretation.append("Good clinical accuracy (≥90% in zones A+B)")
        elif clinically_acceptable >= 85:
            interpretation.append("Acceptable clinical accuracy (≥85% in zones A+B)")
        else:
            interpretation.append("Poor clinical accuracy (<85% in zones A+B)")

        # Zone A performance (more stringent than Clarke)
        if zone_a_percent >= 85:
            interpretation.append("Excellent precision (≥85% in zone A)")
        elif zone_a_percent >= 75:
            interpretation.append("Good precision (≥75% in zone A)")
        elif zone_a_percent >= 65:
            interpretation.append("Acceptable precision (≥65% in zone A)")
        else:
            interpretation.append("Poor precision (<65% in zone A)")

        # Safety assessment (more stringent than Clarke)
        if dangerous_errors <= 1:
            interpretation.append("Excellent safety profile (≤1% dangerous errors)")
        elif dangerous_errors <= 2:
            interpretation.append("Good safety profile (≤2% dangerous errors)")
        elif dangerous_errors <= 3:
            interpretation.append("Acceptable safety profile (≤3% dangerous errors)")
        else:
            interpretation.append("Poor safety profile (>3% dangerous errors)")

        interpretation.append(
            f"Analysis performed using {grid_type.upper()} diabetes criteria"
        )

        return "; ".join(interpretation)

    def compare_with_clarke(
        self, parkes_results: Dict[str, float], clarke_results: Dict[str, float]
    ) -> str:
        """
        Compare Parkes and Clarke Error Grid results.

        Args:
            parkes_results: Results from Parkes Error Grid analysis
            clarke_results: Results from Clarke Error Grid analysis

        Returns:
            Comparison summary string
        """
        parkes_acceptable = parkes_results["parkes_clinically_acceptable_percent"]
        clarke_acceptable = clarke_results["clarke_clinically_acceptable_percent"]

        parkes_zone_a = parkes_results["parkes_zone_a_percent"]
        clarke_zone_a = clarke_results["clarke_zone_a_percent"]

        comparison = []

        # Overall acceptability comparison
        diff_acceptable = parkes_acceptable - clarke_acceptable
        if abs(diff_acceptable) < 1:
            comparison.append(
                "Similar clinical acceptability between Parkes and Clarke grids"
            )
        elif diff_acceptable > 0:
            comparison.append(
                f"Parkes grid shows {diff_acceptable:.1f}% higher clinical acceptability"
            )
        else:
            comparison.append(
                f"Clarke grid shows {abs(diff_acceptable):.1f}% higher clinical acceptability"
            )

        # Zone A comparison
        diff_zone_a = parkes_zone_a - clarke_zone_a
        if abs(diff_zone_a) < 2:
            comparison.append("Similar precision (Zone A) between grids")
        elif diff_zone_a > 0:
            comparison.append(f"Parkes grid shows {diff_zone_a:.1f}% higher precision")
        else:
            comparison.append(
                f"Clarke grid shows {abs(diff_zone_a):.1f}% higher precision"
            )

        # General interpretation
        if parkes_acceptable < clarke_acceptable:
            comparison.append("Parkes grid provides more stringent clinical assessment")
        else:
            comparison.append(
                "Results consistent between both clinical assessment methods"
            )

        return "; ".join(comparison)

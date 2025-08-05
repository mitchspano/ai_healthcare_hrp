"""Clarke Error Grid Analysis for glucose prediction evaluation."""

import numpy as np
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


class ClarkeErrorGrid:
    """
    Implements Clarke Error Grid Analysis for glucose prediction evaluation.

    The Clarke Error Grid is a clinical accuracy assessment tool that categorizes
    glucose prediction errors into five zones (A-E) based on clinical risk.
    """

    def __init__(self):
        """Initialize Clarke Error Grid analyzer."""
        pass

    def _classify_zone(self, reference: float, predicted: float) -> str:
        """
        Classify a single glucose prediction pair into Clarke Error Grid zones.

        Args:
            reference: Reference glucose value (mg/dL)
            predicted: Predicted glucose value (mg/dL)

        Returns:
            Zone classification ('A', 'B', 'C', 'D', or 'E')
        """
        # Zone A: Clinically accurate
        # Within 20% of reference or both values < 70 mg/dL
        if reference <= 70:
            if predicted <= 70:
                return "A"
            elif predicted <= 180:
                return "B"
            else:
                return "C"

        elif reference >= 180:
            if predicted >= 70:
                if predicted <= reference + 0.2 * reference:
                    return "A"
                elif predicted >= reference + 0.2 * reference:
                    return "C"
                else:
                    return "B"
            else:
                return "E"

        else:  # 70 < reference < 180
            if abs(predicted - reference) <= 0.2 * reference:
                return "A"
            elif predicted > reference:
                if predicted <= 180:
                    return "B"
                elif predicted <= 240:
                    return "C"
                else:
                    return "C"
            else:  # predicted < reference
                if predicted >= 70:
                    return "B"
                elif predicted >= 56:
                    return "D"
                else:
                    return "E"

    def _detailed_zone_classification(self, reference: float, predicted: float) -> str:
        """
        Detailed Clarke Error Grid zone classification following the original algorithm.

        Args:
            reference: Reference glucose value (mg/dL)
            predicted: Predicted glucose value (mg/dL)

        Returns:
            Zone classification ('A', 'B', 'C', 'D', or 'E')
        """
        # Convert to ensure we're working with floats
        ref = float(reference)
        pred = float(predicted)

        # Zone A: Clinically accurate
        # Values within 20% of reference or both in hypoglycemic range
        if (abs(pred - ref) <= 0.2 * ref) or (ref < 70 and pred < 70):
            return "A"

        # Zone B: Benign errors
        if ref < 70:  # Reference is hypoglycemic
            if 70 <= pred <= 180:
                return "B"
        elif ref > 180:  # Reference is hyperglycemic
            if 70 <= pred < ref:
                return "B"
        else:  # Reference is in normal range (70-180)
            if pred > ref:
                if pred <= 180:
                    return "B"
            else:  # pred < ref
                if pred >= 70:
                    return "B"

        # Zone C: Overcorrection errors
        if ref < 70 and pred > 180:
            return "C"
        elif ref > 180 and pred < 70:
            return "C"
        elif 70 <= ref <= 180:
            if pred > 180:
                return "C"

        # Zone D: Dangerous failure to detect hypoglycemia
        if ref < 70 and pred > 70:
            return "D"
        elif 70 <= ref <= 180 and pred < 70:
            return "D"

        # Zone E: Erroneous treatment
        if ref > 180 and pred < 70:
            return "E"
        elif ref < 70 and pred > 180:
            return "E"

        # Default fallback (should not reach here with proper implementation)
        # Determine based on magnitude of error
        error_magnitude = abs(pred - ref)
        if error_magnitude <= 0.3 * ref:
            return "B"
        elif error_magnitude <= 0.5 * ref:
            return "C"
        else:
            return "D"

    def analyze(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Perform Clarke Error Grid Analysis on glucose predictions.

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

        # Remove invalid values (negative or zero glucose values)
        valid_mask = (y_true > 0) & (y_pred > 0)
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]

        if len(y_true_valid) == 0:
            raise ValueError("No valid glucose value pairs found")

        # Classify each prediction pair
        zones = []
        for true_val, pred_val in zip(y_true_valid, y_pred_valid):
            zone = self._detailed_zone_classification(true_val, pred_val)
            zones.append(zone)

        # Count occurrences of each zone
        zone_counts = {zone: zones.count(zone) for zone in ["A", "B", "C", "D", "E"]}
        total_points = len(zones)

        # Calculate percentages
        zone_percentages = {
            f"clarke_zone_{zone.lower()}_percent": (count / total_points) * 100
            for zone, count in zone_counts.items()
        }

        # Add clinical interpretation metrics
        clinically_acceptable = zone_counts["A"] + zone_counts["B"]
        clinically_acceptable_percent = (clinically_acceptable / total_points) * 100

        dangerous_errors = zone_counts["D"] + zone_counts["E"]
        dangerous_errors_percent = (dangerous_errors / total_points) * 100

        results = {
            **zone_percentages,
            "clarke_clinically_acceptable_percent": clinically_acceptable_percent,
            "clarke_dangerous_errors_percent": dangerous_errors_percent,
            "clarke_total_points": total_points,
        }

        logger.info(f"Clarke Error Grid Analysis completed:")
        logger.info(
            f"  Zone A (Clinically Accurate): {zone_percentages['clarke_zone_a_percent']:.1f}%"
        )
        logger.info(
            f"  Zone B (Benign Errors): {zone_percentages['clarke_zone_b_percent']:.1f}%"
        )
        logger.info(
            f"  Zone C (Overcorrection): {zone_percentages['clarke_zone_c_percent']:.1f}%"
        )
        logger.info(
            f"  Zone D (Dangerous): {zone_percentages['clarke_zone_d_percent']:.1f}%"
        )
        logger.info(
            f"  Zone E (Erroneous): {zone_percentages['clarke_zone_e_percent']:.1f}%"
        )
        logger.info(
            f"  Clinically Acceptable (A+B): {clinically_acceptable_percent:.1f}%"
        )

        return results

    def get_zone_boundaries(self) -> Dict[str, str]:
        """
        Get descriptions of Clarke Error Grid zone boundaries.

        Returns:
            Dictionary with zone descriptions
        """
        return {
            "A": "Clinically accurate: Within 20% of reference or both < 70 mg/dL",
            "B": "Benign errors: Deviations that would not lead to inappropriate treatment",
            "C": "Overcorrection errors: Deviations leading to unnecessary corrections",
            "D": "Dangerous errors: Failure to detect hypoglycemia or hyperglycemia",
            "E": "Erroneous treatment: Deviations that could lead to dangerous treatment decisions",
        }

    def get_clinical_interpretation(self, results: Dict[str, float]) -> str:
        """
        Generate clinical interpretation of Clarke Error Grid results.

        Args:
            results: Results from analyze() method

        Returns:
            Clinical interpretation string
        """
        zone_a_percent = results["clarke_zone_a_percent"]
        zone_b_percent = results["clarke_zone_b_percent"]
        clinically_acceptable = results["clarke_clinically_acceptable_percent"]
        dangerous_errors = results["clarke_dangerous_errors_percent"]

        interpretation = []

        # Overall clinical acceptability
        if clinically_acceptable >= 95:
            interpretation.append("Excellent clinical accuracy (≥95% in zones A+B)")
        elif clinically_acceptable >= 90:
            interpretation.append("Good clinical accuracy (≥90% in zones A+B)")
        elif clinically_acceptable >= 80:
            interpretation.append("Acceptable clinical accuracy (≥80% in zones A+B)")
        else:
            interpretation.append("Poor clinical accuracy (<80% in zones A+B)")

        # Zone A performance
        if zone_a_percent >= 75:
            interpretation.append("High precision (≥75% in zone A)")
        elif zone_a_percent >= 60:
            interpretation.append("Moderate precision (≥60% in zone A)")
        else:
            interpretation.append("Low precision (<60% in zone A)")

        # Safety assessment
        if dangerous_errors <= 2:
            interpretation.append("Low safety risk (≤2% dangerous errors)")
        elif dangerous_errors <= 5:
            interpretation.append("Moderate safety risk (≤5% dangerous errors)")
        else:
            interpretation.append("High safety risk (>5% dangerous errors)")

        return "; ".join(interpretation)

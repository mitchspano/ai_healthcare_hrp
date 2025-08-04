"""
Error recovery and restart capabilities for failed pipeline runs.
"""

import logging
import traceback
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
from datetime import datetime
import json
from enum import Enum

from .pipeline_status import PipelineStatus, StageStatus


class RecoveryStrategy(Enum):
    """Recovery strategies for different types of errors."""

    RETRY = "retry"
    SKIP = "skip"
    RESTART_FROM_STAGE = "restart_from_stage"
    RESTART_FROM_CHECKPOINT = "restart_from_checkpoint"
    MANUAL_INTERVENTION = "manual_intervention"


class ErrorRecoveryManager:
    """Manages error recovery and restart capabilities for pipeline failures."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize error recovery manager.

        Args:
            config: Configuration for error recovery settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Recovery settings
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 60)  # seconds
        self.checkpoint_dir = Path(self.config.get("checkpoint_dir", "checkpoints"))
        self.error_log_dir = Path(self.config.get("error_log_dir", "logs/errors"))

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.error_log_dir.mkdir(parents=True, exist_ok=True)

        # Error tracking
        self.error_history: List[Dict[str, Any]] = []
        self.retry_counts: Dict[str, int] = {}

        # Recovery strategies for different error types
        self.recovery_strategies = {
            "MemoryError": RecoveryStrategy.RESTART_FROM_CHECKPOINT,
            "FileNotFoundError": RecoveryStrategy.MANUAL_INTERVENTION,
            "ConnectionError": RecoveryStrategy.RETRY,
            "TimeoutError": RecoveryStrategy.RETRY,
            "ValueError": RecoveryStrategy.SKIP,
            "KeyError": RecoveryStrategy.SKIP,
            "ImportError": RecoveryStrategy.MANUAL_INTERVENTION,
            "ModuleNotFoundError": RecoveryStrategy.MANUAL_INTERVENTION,
            "default": RecoveryStrategy.RETRY,
        }

    def handle_error(
        self,
        stage_name: str,
        error: Exception,
        pipeline_status: PipelineStatus,
        context: Dict[str, Any] = None,
    ) -> RecoveryStrategy:
        """
        Handle an error that occurred during pipeline execution.

        Args:
            stage_name: Name of the stage where error occurred
            error: The exception that occurred
            pipeline_status: Current pipeline status
            context: Additional context information

        Returns:
            Recommended recovery strategy
        """
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "stage_name": stage_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": context or {},
        }

        # Log the error
        self.logger.error(
            f"Error in stage '{stage_name}': {error_info['error_type']}: {error_info['error_message']}"
        )

        # Save detailed error information
        self._save_error_details(error_info)

        # Add to error history
        self.error_history.append(error_info)

        # Update pipeline status
        pipeline_status.fail_stage(stage_name, str(error))

        # Determine recovery strategy
        strategy = self._determine_recovery_strategy(error_info, stage_name)

        self.logger.info(
            f"Recommended recovery strategy for '{stage_name}': {strategy.value}"
        )

        return strategy

    def _determine_recovery_strategy(
        self, error_info: Dict[str, Any], stage_name: str
    ) -> RecoveryStrategy:
        """
        Determine the appropriate recovery strategy based on error type and history.

        Args:
            error_info: Information about the error
            stage_name: Name of the stage where error occurred

        Returns:
            Recommended recovery strategy
        """
        error_type = error_info["error_type"]

        # Check retry count
        retry_count = self.retry_counts.get(stage_name, 0)

        # If we've exceeded max retries, escalate strategy
        if retry_count >= self.max_retries:
            if error_type in ["MemoryError", "ResourceExhaustedError"]:
                return RecoveryStrategy.RESTART_FROM_CHECKPOINT
            else:
                return RecoveryStrategy.MANUAL_INTERVENTION

        # Get strategy based on error type
        strategy = self.recovery_strategies.get(
            error_type, self.recovery_strategies["default"]
        )

        # Special handling for specific scenarios
        if "out of memory" in error_info["error_message"].lower():
            return RecoveryStrategy.RESTART_FROM_CHECKPOINT

        if (
            "disk" in error_info["error_message"].lower()
            and "space" in error_info["error_message"].lower()
        ):
            return RecoveryStrategy.MANUAL_INTERVENTION

        return strategy

    def can_retry(self, stage_name: str) -> bool:
        """
        Check if a stage can be retried.

        Args:
            stage_name: Name of the stage

        Returns:
            True if stage can be retried, False otherwise
        """
        retry_count = self.retry_counts.get(stage_name, 0)
        return retry_count < self.max_retries

    def increment_retry_count(self, stage_name: str) -> int:
        """
        Increment retry count for a stage.

        Args:
            stage_name: Name of the stage

        Returns:
            New retry count
        """
        self.retry_counts[stage_name] = self.retry_counts.get(stage_name, 0) + 1
        return self.retry_counts[stage_name]

    def reset_retry_count(self, stage_name: str) -> None:
        """
        Reset retry count for a stage.

        Args:
            stage_name: Name of the stage
        """
        if stage_name in self.retry_counts:
            del self.retry_counts[stage_name]

    def save_checkpoint(self, stage_name: str, data: Dict[str, Any]) -> Path:
        """
        Save a checkpoint for recovery purposes.

        Args:
            stage_name: Name of the stage
            data: Data to save in checkpoint

        Returns:
            Path to the saved checkpoint file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = (
            self.checkpoint_dir / f"{stage_name}_checkpoint_{timestamp}.json"
        )

        checkpoint_data = {
            "stage_name": stage_name,
            "timestamp": timestamp,
            "data": data,
        }

        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f, indent=2, default=str)

        self.logger.info(
            f"Checkpoint saved for stage '{stage_name}': {checkpoint_file}"
        )
        return checkpoint_file

    def load_checkpoint(
        self, stage_name: str, checkpoint_file: Path = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint for recovery.

        Args:
            stage_name: Name of the stage
            checkpoint_file: Specific checkpoint file to load (if None, loads latest)

        Returns:
            Checkpoint data or None if not found
        """
        if checkpoint_file is None:
            # Find the latest checkpoint for this stage
            pattern = f"{stage_name}_checkpoint_*.json"
            checkpoint_files = list(self.checkpoint_dir.glob(pattern))

            if not checkpoint_files:
                self.logger.warning(f"No checkpoints found for stage '{stage_name}'")
                return None

            # Sort by timestamp and get the latest
            checkpoint_file = sorted(checkpoint_files)[-1]

        try:
            with open(checkpoint_file, "r") as f:
                checkpoint_data = json.load(f)

            self.logger.info(
                f"Checkpoint loaded for stage '{stage_name}': {checkpoint_file}"
            )
            return checkpoint_data.get("data")

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint {checkpoint_file}: {e}")
            return None

    def get_restart_point(self, pipeline_status: PipelineStatus) -> Optional[str]:
        """
        Determine the appropriate restart point for a failed pipeline.

        Args:
            pipeline_status: Current pipeline status

        Returns:
            Name of the stage to restart from, or None if full restart needed
        """
        failed_stages = pipeline_status.get_failed_stages()

        if not failed_stages:
            return None

        # Find the first failed stage
        stage_names = list(pipeline_status.stages.keys())

        for stage_name in stage_names:
            stage = pipeline_status.stages[stage_name]
            if stage.status == StageStatus.FAILED:
                # Check if we have a checkpoint for this stage
                if self._has_checkpoint(stage_name):
                    return stage_name

                # Otherwise, restart from the previous completed stage
                stage_index = stage_names.index(stage_name)
                if stage_index > 0:
                    return stage_names[stage_index - 1]
                else:
                    return stage_name

        return None

    def _has_checkpoint(self, stage_name: str) -> bool:
        """
        Check if a checkpoint exists for a stage.

        Args:
            stage_name: Name of the stage

        Returns:
            True if checkpoint exists, False otherwise
        """
        pattern = f"{stage_name}_checkpoint_*.json"
        checkpoint_files = list(self.checkpoint_dir.glob(pattern))
        return len(checkpoint_files) > 0

    def _save_error_details(self, error_info: Dict[str, Any]) -> None:
        """
        Save detailed error information to file.

        Args:
            error_info: Error information to save
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_file = self.error_log_dir / f"error_{timestamp}.json"

        with open(error_file, "w") as f:
            json.dump(error_info, f, indent=2)

    def generate_recovery_report(self) -> str:
        """
        Generate a recovery report with error history and recommendations.

        Returns:
            Recovery report as string
        """
        report = []
        report.append("Pipeline Error Recovery Report")
        report.append("=" * 40)
        report.append(f"Total Errors: {len(self.error_history)}")
        report.append(f"Stages with Retries: {len(self.retry_counts)}")
        report.append("")

        if self.error_history:
            report.append("Error History:")
            report.append("-" * 20)

            for i, error in enumerate(
                self.error_history[-10:], 1
            ):  # Show last 10 errors
                report.append(f"{i}. {error['timestamp']}")
                report.append(f"   Stage: {error['stage_name']}")
                report.append(
                    f"   Error: {error['error_type']}: {error['error_message']}"
                )
                report.append("")

        if self.retry_counts:
            report.append("Retry Counts:")
            report.append("-" * 15)

            for stage, count in self.retry_counts.items():
                report.append(f"  {stage}: {count}/{self.max_retries}")
            report.append("")

        # Recommendations
        report.append("Recommendations:")
        report.append("-" * 15)

        if any(count >= self.max_retries for count in self.retry_counts.values()):
            report.append("• Some stages have exceeded maximum retry attempts")
            report.append("• Consider manual intervention or configuration changes")

        if any("MemoryError" in error["error_type"] for error in self.error_history):
            report.append(
                "• Memory errors detected - consider reducing batch size or using checkpoints"
            )

        if any(
            "ConnectionError" in error["error_type"] for error in self.error_history
        ):
            report.append("• Network errors detected - check internet connectivity")

        return "\n".join(report)

    def cleanup_old_checkpoints(self, days_to_keep: int = 7) -> None:
        """
        Clean up old checkpoint files.

        Args:
            days_to_keep: Number of days of checkpoints to keep
        """
        cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 3600)

        for checkpoint_file in self.checkpoint_dir.glob("*_checkpoint_*.json"):
            if checkpoint_file.stat().st_mtime < cutoff_time:
                checkpoint_file.unlink()
                self.logger.info(f"Cleaned up old checkpoint: {checkpoint_file}")

    def reset_error_history(self) -> None:
        """Reset error history and retry counts."""
        self.error_history.clear()
        self.retry_counts.clear()
        self.logger.info("Error history and retry counts reset")

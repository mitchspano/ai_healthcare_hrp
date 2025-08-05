"""
Pipeline status tracking and progress reporting system.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from pathlib import Path


class StageStatus(Enum):
    """Status of individual pipeline stages."""

    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageInfo:
    """Information about a pipeline stage."""

    name: str
    status: StageStatus = StageStatus.NOT_STARTED
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    error_message: Optional[str] = None
    progress: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def start(self) -> None:
        """Mark stage as started."""
        self.status = StageStatus.RUNNING
        self.start_time = datetime.now()
        self.progress = 0.0

    def complete(self, metadata: Dict[str, Any] = None) -> None:
        """Mark stage as completed."""
        self.status = StageStatus.COMPLETED
        self.end_time = datetime.now()
        self.progress = 100.0
        if self.start_time:
            self.duration = (self.end_time - self.start_time).total_seconds()
        if metadata:
            self.metadata.update(metadata)

    def fail(self, error_message: str) -> None:
        """Mark stage as failed."""
        self.status = StageStatus.FAILED
        self.end_time = datetime.now()
        self.error_message = error_message
        if self.start_time:
            self.duration = (self.end_time - self.start_time).total_seconds()

    def skip(self, reason: str = None) -> None:
        """Mark stage as skipped."""
        self.status = StageStatus.SKIPPED
        self.end_time = datetime.now()
        if reason:
            self.metadata["skip_reason"] = reason

    def update_progress(self, progress: float) -> None:
        """Update stage progress."""
        self.progress = max(0.0, min(100.0, progress))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "error_message": self.error_message,
            "progress": self.progress,
            "metadata": self.metadata,
        }


class PipelineStatus:
    """Tracks status and progress of the entire pipeline."""

    def __init__(self, pipeline_id: str = None):
        """
        Initialize pipeline status tracker.

        Args:
            pipeline_id: Unique identifier for this pipeline run
        """
        self.pipeline_id = (
            pipeline_id or f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.status: StageStatus = StageStatus.NOT_STARTED
        self.stages: Dict[str, StageInfo] = {}
        self.current_stage: Optional[str] = None
        self.metadata: Dict[str, Any] = {}

    def add_stage(self, stage_name: str) -> StageInfo:
        """
        Add a new stage to track.

        Args:
            stage_name: Name of the stage

        Returns:
            StageInfo object for the added stage
        """
        stage_info = StageInfo(name=stage_name)
        self.stages[stage_name] = stage_info
        return stage_info

    def start_pipeline(self) -> None:
        """Mark pipeline as started."""
        self.status = StageStatus.RUNNING
        self.start_time = datetime.now()

    def complete_pipeline(self, metadata: Dict[str, Any] = None) -> None:
        """Mark pipeline as completed."""
        self.status = StageStatus.COMPLETED
        self.end_time = datetime.now()
        self.current_stage = None
        if metadata:
            self.metadata.update(metadata)

    def fail_pipeline(self, error_message: str) -> None:
        """Mark pipeline as failed."""
        self.status = StageStatus.FAILED
        self.end_time = datetime.now()
        self.metadata["error_message"] = error_message

    def start_stage(self, stage_name: str) -> StageInfo:
        """
        Start a specific stage.

        Args:
            stage_name: Name of the stage to start

        Returns:
            StageInfo object for the started stage
        """
        if stage_name not in self.stages:
            self.add_stage(stage_name)

        stage = self.stages[stage_name]
        stage.start()
        self.current_stage = stage_name
        return stage

    def complete_stage(self, stage_name: str, metadata: Dict[str, Any] = None) -> None:
        """
        Complete a specific stage.

        Args:
            stage_name: Name of the stage to complete
            metadata: Additional metadata for the stage
        """
        if stage_name in self.stages:
            self.stages[stage_name].complete(metadata)

        if self.current_stage == stage_name:
            self.current_stage = None

    def fail_stage(self, stage_name: str, error_message: str) -> None:
        """
        Mark a specific stage as failed.

        Args:
            stage_name: Name of the stage that failed
            error_message: Error message describing the failure
        """
        if stage_name in self.stages:
            self.stages[stage_name].fail(error_message)

        if self.current_stage == stage_name:
            self.current_stage = None

    def skip_stage(self, stage_name: str, reason: str = None) -> None:
        """
        Mark a specific stage as skipped.

        Args:
            stage_name: Name of the stage to skip
            reason: Reason for skipping the stage
        """
        if stage_name not in self.stages:
            self.add_stage(stage_name)

        self.stages[stage_name].skip(reason)

    def update_stage_progress(self, stage_name: str, progress: float) -> None:
        """
        Update progress for a specific stage.

        Args:
            stage_name: Name of the stage
            progress: Progress percentage (0-100)
        """
        if stage_name in self.stages:
            self.stages[stage_name].update_progress(progress)

    def get_overall_progress(self) -> float:
        """
        Calculate overall pipeline progress.

        Returns:
            Overall progress percentage (0-100)
        """
        if not self.stages:
            return 0.0

        total_progress = sum(stage.progress for stage in self.stages.values())
        return total_progress / len(self.stages)

    def get_stage_summary(self) -> Dict[str, int]:
        """
        Get summary of stage statuses.

        Returns:
            Dictionary with counts for each status
        """
        summary = {status.value: 0 for status in StageStatus}

        for stage in self.stages.values():
            summary[stage.status.value] += 1

        return summary

    def get_failed_stages(self) -> List[StageInfo]:
        """
        Get list of failed stages.

        Returns:
            List of StageInfo objects for failed stages
        """
        return [
            stage
            for stage in self.stages.values()
            if stage.status == StageStatus.FAILED
        ]

    def get_duration(self) -> Optional[float]:
        """
        Get total pipeline duration.

        Returns:
            Duration in seconds, or None if not completed
        """
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def save_status(self, filepath: Path) -> None:
        """
        Save pipeline status to file.

        Args:
            filepath: Path to save the status file
        """
        status_data = {
            "pipeline_id": self.pipeline_id,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.get_duration(),
            "current_stage": self.current_stage,
            "overall_progress": self.get_overall_progress(),
            "stage_summary": self.get_stage_summary(),
            "stages": {name: stage.to_dict() for name, stage in self.stages.items()},
            "metadata": self.metadata,
        }

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(status_data, f, indent=2)

    def load_status(self, filepath: Path) -> None:
        """
        Load pipeline status from file.

        Args:
            filepath: Path to load the status file from
        """
        with open(filepath, "r") as f:
            status_data = json.load(f)

        self.pipeline_id = status_data["pipeline_id"]
        self.status = StageStatus(status_data["status"])
        self.start_time = (
            datetime.fromisoformat(status_data["start_time"])
            if status_data["start_time"]
            else None
        )
        self.end_time = (
            datetime.fromisoformat(status_data["end_time"])
            if status_data["end_time"]
            else None
        )
        self.current_stage = status_data["current_stage"]
        self.metadata = status_data.get("metadata", {})

        # Reconstruct stages
        self.stages = {}
        for stage_name, stage_data in status_data["stages"].items():
            stage = StageInfo(name=stage_name)
            stage.status = StageStatus(stage_data["status"])
            stage.start_time = (
                datetime.fromisoformat(stage_data["start_time"])
                if stage_data["start_time"]
                else None
            )
            stage.end_time = (
                datetime.fromisoformat(stage_data["end_time"])
                if stage_data["end_time"]
                else None
            )
            stage.duration = stage_data["duration"]
            stage.error_message = stage_data["error_message"]
            stage.progress = stage_data["progress"]
            stage.metadata = stage_data["metadata"]
            self.stages[stage_name] = stage

    def print_status(self) -> None:
        """Print current pipeline status to console."""
        print(f"\n{'='*60}")
        print(f"Pipeline Status: {self.pipeline_id}")
        print(f"{'='*60}")
        print(f"Overall Status: {self.status.value.upper()}")
        print(f"Overall Progress: {self.get_overall_progress():.1f}%")

        if self.current_stage:
            print(f"Current Stage: {self.current_stage}")

        if self.start_time:
            print(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        if self.end_time:
            print(f"Ended: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            duration = self.get_duration()
            if duration:
                print(f"Duration: {duration:.2f} seconds")

        print(f"\nStage Summary:")
        summary = self.get_stage_summary()
        for status, count in summary.items():
            if count > 0:
                print(f"  {status.replace('_', ' ').title()}: {count}")

        print(f"\nDetailed Stage Status:")
        for stage_name, stage in self.stages.items():
            status_icon = {
                StageStatus.NOT_STARTED: "⏸️",
                StageStatus.RUNNING: "🔄",
                StageStatus.COMPLETED: "✅",
                StageStatus.FAILED: "❌",
                StageStatus.SKIPPED: "⏭️",
            }.get(stage.status, "❓")

            print(
                f"  {status_icon} {stage_name}: {stage.status.value} ({stage.progress:.1f}%)"
            )

            if stage.duration:
                print(f"    Duration: {stage.duration:.2f}s")

            if stage.error_message:
                print(f"    Error: {stage.error_message}")

        print(f"{'='*60}\n")

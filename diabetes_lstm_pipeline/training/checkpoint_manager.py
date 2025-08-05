"""Model checkpointing and resume functionality for training."""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
from pathlib import Path
import json
import pickle
import shutil
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages model checkpoints with advanced save/load and resume functionality."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize checkpoint manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.checkpoint_config = config.get("checkpointing", {})

        # Checkpoint directories
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        self.best_models_dir = self.checkpoint_dir / "best_models"
        self.periodic_dir = self.checkpoint_dir / "periodic"
        self.resume_dir = self.checkpoint_dir / "resume"

        # Create directories
        for dir_path in [
            self.checkpoint_dir,
            self.best_models_dir,
            self.periodic_dir,
            self.resume_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Checkpoint parameters
        self.save_best_only = self.checkpoint_config.get("save_best_only", True)
        self.save_frequency = self.checkpoint_config.get("save_frequency", 10)
        self.max_checkpoints = self.checkpoint_config.get("max_checkpoints", 5)
        self.monitor_metric = self.checkpoint_config.get("monitor_metric", "val_loss")
        self.mode = self.checkpoint_config.get("mode", "min")

        # State tracking
        self.best_metric_value = None
        self.checkpoint_history = []
        self.current_run_id = None

    def create_checkpoint(
        self,
        model: keras.Model,
        epoch: int,
        logs: Dict[str, float],
        run_id: str,
        optimizer_state: Optional[Dict] = None,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        """
        Create a checkpoint of the current training state.

        Args:
            model: Keras model to checkpoint
            epoch: Current epoch number
            logs: Training logs for this epoch
            run_id: Unique run identifier
            optimizer_state: Optional optimizer state
            additional_data: Optional additional data to save

        Returns:
            Path to created checkpoint or None
        """
        self.current_run_id = run_id

        # Determine if this is the best model
        is_best = self._is_best_model(logs)

        # Create checkpoint data
        checkpoint_data = {
            "epoch": epoch,
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "logs": logs.copy(),
            "config": self.config.copy(),
            "is_best": is_best,
            "monitor_metric": self.monitor_metric,
            "monitor_value": logs.get(self.monitor_metric, None),
        }

        if additional_data:
            checkpoint_data.update(additional_data)

        # Save checkpoint
        checkpoint_path = None

        if is_best:
            checkpoint_path = self._save_best_checkpoint(
                model, checkpoint_data, optimizer_state
            )

        # Save periodic checkpoint
        if epoch % self.save_frequency == 0:
            periodic_path = self._save_periodic_checkpoint(
                model, checkpoint_data, optimizer_state
            )
            if checkpoint_path is None:
                checkpoint_path = periodic_path

        # Save resume checkpoint (always save latest for resuming)
        resume_path = self._save_resume_checkpoint(
            model, checkpoint_data, optimizer_state
        )

        # Update checkpoint history
        self.checkpoint_history.append(checkpoint_data)

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

        return checkpoint_path

    def _is_best_model(self, logs: Dict[str, float]) -> bool:
        """
        Determine if current model is the best so far.

        Args:
            logs: Training logs

        Returns:
            True if this is the best model
        """
        if self.monitor_metric not in logs:
            return False

        current_value = logs[self.monitor_metric]

        if self.best_metric_value is None:
            self.best_metric_value = current_value
            return True

        if self.mode == "min":
            is_better = current_value < self.best_metric_value
        else:  # mode == "max"
            is_better = current_value > self.best_metric_value

        if is_better:
            self.best_metric_value = current_value
            return True

        return False

    def _save_best_checkpoint(
        self,
        model: keras.Model,
        checkpoint_data: Dict[str, Any],
        optimizer_state: Optional[Dict] = None,
    ) -> Path:
        """
        Save the best model checkpoint.

        Args:
            model: Keras model
            checkpoint_data: Checkpoint metadata
            optimizer_state: Optional optimizer state

        Returns:
            Path to saved checkpoint
        """
        run_id = checkpoint_data["run_id"]
        epoch = checkpoint_data["epoch"]

        # Model path
        model_path = self.best_models_dir / f"best_model_{run_id}.keras"

        # Save model
        model.save(model_path)

        # Save metadata
        metadata_path = self.best_models_dir / f"best_model_{run_id}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2, default=str)

        # Save optimizer state if provided
        if optimizer_state:
            optimizer_path = self.best_models_dir / f"best_model_{run_id}_optimizer.pkl"
            with open(optimizer_path, "wb") as f:
                pickle.dump(optimizer_state, f)

        logger.info(f"Saved best model checkpoint at epoch {epoch}: {model_path}")
        return model_path

    def _save_periodic_checkpoint(
        self,
        model: keras.Model,
        checkpoint_data: Dict[str, Any],
        optimizer_state: Optional[Dict] = None,
    ) -> Path:
        """
        Save a periodic checkpoint.

        Args:
            model: Keras model
            checkpoint_data: Checkpoint metadata
            optimizer_state: Optional optimizer state

        Returns:
            Path to saved checkpoint
        """
        run_id = checkpoint_data["run_id"]
        epoch = checkpoint_data["epoch"]

        # Model path
        model_path = self.periodic_dir / f"model_{run_id}_epoch_{epoch}.keras"

        # Save model
        model.save(model_path)

        # Save metadata
        metadata_path = (
            self.periodic_dir / f"model_{run_id}_epoch_{epoch}_metadata.json"
        )
        with open(metadata_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2, default=str)

        # Save optimizer state if provided
        if optimizer_state:
            optimizer_path = (
                self.periodic_dir / f"model_{run_id}_epoch_{epoch}_optimizer.pkl"
            )
            with open(optimizer_path, "wb") as f:
                pickle.dump(optimizer_state, f)

        logger.debug(f"Saved periodic checkpoint at epoch {epoch}: {model_path}")
        return model_path

    def _save_resume_checkpoint(
        self,
        model: keras.Model,
        checkpoint_data: Dict[str, Any],
        optimizer_state: Optional[Dict] = None,
    ) -> Path:
        """
        Save a resume checkpoint (latest state for resuming training).

        Args:
            model: Keras model
            checkpoint_data: Checkpoint metadata
            optimizer_state: Optional optimizer state

        Returns:
            Path to saved checkpoint
        """
        run_id = checkpoint_data["run_id"]

        # Model path (overwrite previous resume checkpoint)
        model_path = self.resume_dir / f"resume_{run_id}.keras"

        # Save model
        model.save(model_path)

        # Save metadata
        metadata_path = self.resume_dir / f"resume_{run_id}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2, default=str)

        # Save optimizer state if provided
        if optimizer_state:
            optimizer_path = self.resume_dir / f"resume_{run_id}_optimizer.pkl"
            with open(optimizer_path, "wb") as f:
                pickle.dump(optimizer_state, f)

        logger.debug(f"Saved resume checkpoint: {model_path}")
        return model_path

    def _cleanup_old_checkpoints(self) -> None:
        """Clean up old periodic checkpoints to save disk space."""
        if self.max_checkpoints <= 0:
            return

        # Get all periodic checkpoints for current run
        if not self.current_run_id:
            return

        pattern = f"model_{self.current_run_id}_epoch_*.keras"
        checkpoint_files = list(self.periodic_dir.glob(pattern))

        if len(checkpoint_files) <= self.max_checkpoints:
            return

        # Sort by epoch number and remove oldest
        def extract_epoch(path):
            try:
                return int(path.stem.split("_epoch_")[-1])
            except:
                return 0

        checkpoint_files.sort(key=extract_epoch)
        files_to_remove = checkpoint_files[: -self.max_checkpoints]

        for file_path in files_to_remove:
            try:
                # Remove model file
                file_path.unlink()

                # Remove associated metadata and optimizer files
                base_name = file_path.stem
                metadata_file = file_path.parent / f"{base_name}_metadata.json"
                optimizer_file = file_path.parent / f"{base_name}_optimizer.pkl"

                if metadata_file.exists():
                    metadata_file.unlink()
                if optimizer_file.exists():
                    optimizer_file.unlink()

                logger.debug(f"Removed old checkpoint: {file_path}")

            except Exception as e:
                logger.warning(f"Failed to remove old checkpoint {file_path}: {str(e)}")

    def load_checkpoint(
        self, checkpoint_path: Union[str, Path], load_optimizer: bool = True
    ) -> Tuple[keras.Model, Dict[str, Any], Optional[Dict]]:
        """
        Load a model checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint
            load_optimizer: Whether to load optimizer state

        Returns:
            Tuple of (model, metadata, optimizer_state)
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        # Load model
        model = keras.models.load_model(checkpoint_path)

        # Load metadata
        metadata_path = checkpoint_path.parent / f"{checkpoint_path.stem}_metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

        # Load optimizer state
        optimizer_state = None
        if load_optimizer:
            optimizer_path = (
                checkpoint_path.parent / f"{checkpoint_path.stem}_optimizer.pkl"
            )
            if optimizer_path.exists():
                with open(optimizer_path, "rb") as f:
                    optimizer_state = pickle.load(f)

        logger.info(f"Loaded checkpoint from epoch {metadata.get('epoch', 'unknown')}")
        return model, metadata, optimizer_state

    def get_best_checkpoint(self, run_id: str) -> Optional[Path]:
        """
        Get the path to the best checkpoint for a run.

        Args:
            run_id: Run identifier

        Returns:
            Path to best checkpoint or None
        """
        best_model_path = self.best_models_dir / f"best_model_{run_id}.keras"

        if best_model_path.exists():
            return best_model_path

        return None

    def get_resume_checkpoint(self, run_id: str) -> Optional[Path]:
        """
        Get the path to the resume checkpoint for a run.

        Args:
            run_id: Run identifier

        Returns:
            Path to resume checkpoint or None
        """
        resume_path = self.resume_dir / f"resume_{run_id}.keras"

        if resume_path.exists():
            return resume_path

        return None

    def list_checkpoints(self, run_id: Optional[str] = None) -> Dict[str, List[Path]]:
        """
        List available checkpoints.

        Args:
            run_id: Optional run ID to filter by

        Returns:
            Dictionary of checkpoint types and their paths
        """
        checkpoints = {"best": [], "periodic": [], "resume": []}

        # Best checkpoints
        pattern = f"best_model_{run_id}.keras" if run_id else "best_model_*.keras"
        checkpoints["best"] = list(self.best_models_dir.glob(pattern))

        # Periodic checkpoints
        pattern = f"model_{run_id}_epoch_*.keras" if run_id else "model_*_epoch_*.keras"
        checkpoints["periodic"] = list(self.periodic_dir.glob(pattern))

        # Resume checkpoints
        pattern = f"resume_{run_id}.keras" if run_id else "resume_*.keras"
        checkpoints["resume"] = list(self.resume_dir.glob(pattern))

        return checkpoints

    def create_keras_callback(self, run_id: str) -> keras.callbacks.Callback:
        """
        Create a Keras callback for automatic checkpointing.

        Args:
            run_id: Run identifier

        Returns:
            Keras callback
        """
        return CheckpointCallback(self, run_id)

    def export_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        export_dir: Union[str, Path],
        include_optimizer: bool = False,
    ) -> Path:
        """
        Export a checkpoint to a different location with all associated files.

        Args:
            checkpoint_path: Source checkpoint path
            export_dir: Destination directory
            include_optimizer: Whether to include optimizer state

        Returns:
            Path to exported checkpoint
        """
        checkpoint_path = Path(checkpoint_path)
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Copy model file
        exported_model_path = export_dir / checkpoint_path.name
        shutil.copy2(checkpoint_path, exported_model_path)

        # Copy metadata
        metadata_path = checkpoint_path.parent / f"{checkpoint_path.stem}_metadata.json"
        if metadata_path.exists():
            exported_metadata_path = (
                export_dir / f"{checkpoint_path.stem}_metadata.json"
            )
            shutil.copy2(metadata_path, exported_metadata_path)

        # Copy optimizer state if requested
        if include_optimizer:
            optimizer_path = (
                checkpoint_path.parent / f"{checkpoint_path.stem}_optimizer.pkl"
            )
            if optimizer_path.exists():
                exported_optimizer_path = (
                    export_dir / f"{checkpoint_path.stem}_optimizer.pkl"
                )
                shutil.copy2(optimizer_path, exported_optimizer_path)

        logger.info(f"Exported checkpoint to {exported_model_path}")
        return exported_model_path

    def get_checkpoint_info(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get information about a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint

        Returns:
            Dictionary with checkpoint information
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        info = {
            "path": str(checkpoint_path),
            "size_mb": checkpoint_path.stat().st_size / (1024 * 1024),
            "created": datetime.fromtimestamp(
                checkpoint_path.stat().st_mtime
            ).isoformat(),
        }

        # Load metadata if available
        metadata_path = checkpoint_path.parent / f"{checkpoint_path.stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            info.update(metadata)

        return info


class CheckpointCallback(keras.callbacks.Callback):
    """Keras callback that integrates with CheckpointManager."""

    def __init__(self, checkpoint_manager: CheckpointManager, run_id: str):
        """
        Initialize callback.

        Args:
            checkpoint_manager: CheckpointManager instance
            run_id: Run identifier
        """
        super().__init__()
        self.checkpoint_manager = checkpoint_manager
        self.run_id = run_id

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch."""
        if logs is None:
            logs = {}

        # Get optimizer state
        optimizer_state = None
        if hasattr(self.model, "optimizer"):
            try:
                optimizer_state = {
                    "weights": self.model.optimizer.get_weights(),
                    "config": self.model.optimizer.get_config(),
                }
            except Exception as e:
                logger.warning(f"Failed to get optimizer state: {str(e)}")

        # Create checkpoint
        self.checkpoint_manager.create_checkpoint(
            self.model, epoch, logs, self.run_id, optimizer_state
        )

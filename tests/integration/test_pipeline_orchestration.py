"""
Integration tests for complete end-to-end pipeline execution.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from diabetes_lstm_pipeline.orchestration import (
    PipelineOrchestrator,
    PipelineStatus,
    StageStatus,
    ErrorRecoveryManager,
    ParallelProcessor,
)
from diabetes_lstm_pipeline.orchestration.error_recovery import RecoveryStrategy
from diabetes_lstm_pipeline.utils import ConfigManager


class TestPipelineOrchestration:
    """Test suite for pipeline orchestration functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_config(self, temp_dir):
        """Create mock configuration for testing."""
        config_data = {
            "data": {
                "raw_data_path": str(temp_dir / "raw"),
                "processed_data_path": str(temp_dir / "processed"),
                "dataset_url": "https://example.com/dataset.zip",
            },
            "logging": {"level": "INFO", "file": str(temp_dir / "logs" / "test.log")},
            "model": {
                "sequence_length": 60,
                "lstm_units": [64, 32],
                "dropout_rate": 0.2,
            },
            "training": {"batch_size": 32, "epochs": 5},
            "evaluation": {"metrics_output_dir": str(temp_dir / "metrics")},
            "pipeline": {"skip_stages": [], "parallel_stages": []},
            "parallel_processing": {"max_workers": 2, "memory_limit_gb": 1.0},
            "error_recovery": {
                "max_retries": 2,
                "checkpoint_dir": str(temp_dir / "checkpoints"),
            },
        }

        # Create config file
        config_file = temp_dir / "test_config.yaml"
        import yaml

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        return ConfigManager(config_file)

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing."""
        dates = pd.date_range(start="2023-01-01", periods=1000, freq="5min")

        return pd.DataFrame(
            {
                "EventDateTime": dates,
                "CGM": np.random.normal(120, 30, 1000),
                "Basal": np.random.normal(1.0, 0.2, 1000),
                "TotalBolusInsulinDelivered": np.random.exponential(0.5, 1000),
                "CorrectionDelivered": np.random.exponential(0.3, 1000),
                "FoodDelivered": np.random.exponential(0.2, 1000),
                "CarbSize": np.random.exponential(10, 1000),
                "DeviceMode": ["Auto"] * 1000,
                "BolusType": ["Normal"] * 1000,
            }
        )

    def test_pipeline_orchestrator_initialization(self, mock_config):
        """Test pipeline orchestrator initialization."""
        orchestrator = PipelineOrchestrator(mock_config)

        assert orchestrator.config == mock_config
        assert isinstance(orchestrator.status, PipelineStatus)
        assert isinstance(orchestrator.error_recovery, ErrorRecoveryManager)
        assert isinstance(orchestrator.parallel_processor, ParallelProcessor)
        assert len(orchestrator.stages) == 9

    @patch(
        "diabetes_lstm_pipeline.orchestration.pipeline_orchestrator.DataAcquisitionPipeline"
    )
    @patch("diabetes_lstm_pipeline.orchestration.pipeline_orchestrator.DataValidator")
    @patch(
        "diabetes_lstm_pipeline.orchestration.pipeline_orchestrator.DataPreprocessor"
    )
    def test_component_initialization(
        self, mock_preprocessor, mock_validator, mock_acquisition, mock_config
    ):
        """Test that all pipeline components are initialized correctly."""
        orchestrator = PipelineOrchestrator(mock_config)

        # Verify components were initialized
        mock_acquisition.assert_called_once()
        mock_validator.assert_called_once()
        mock_preprocessor.assert_called_once()

    def test_pipeline_status_tracking(self, mock_config):
        """Test pipeline status tracking functionality."""
        orchestrator = PipelineOrchestrator(mock_config)

        # Test status initialization
        assert orchestrator.status.status == StageStatus.NOT_STARTED
        assert len(orchestrator.status.stages) == 0

        # Test adding stages
        for stage in orchestrator.stages:
            orchestrator.status.add_stage(stage)

        assert len(orchestrator.status.stages) == len(orchestrator.stages)

    def test_stage_execution_success(self, mock_config, sample_dataframe):
        """Test successful stage execution."""
        orchestrator = PipelineOrchestrator(mock_config)

        # Mock stage method
        def mock_stage_method(progress_callback):
            progress_callback(50)
            progress_callback(100)
            return sample_dataframe

        orchestrator._execute_data_acquisition = mock_stage_method

        # Execute stage
        success = orchestrator._execute_stage("data_acquisition")

        assert success is True
        assert "data_acquisition" in orchestrator.pipeline_data
        assert (
            orchestrator.status.stages["data_acquisition"].status
            == StageStatus.COMPLETED
        )

    def test_stage_execution_failure(self, mock_config):
        """Test stage execution failure handling."""
        orchestrator = PipelineOrchestrator(mock_config)

        # Mock stage method that raises exception
        def mock_stage_method(progress_callback):
            raise ValueError("Test error")

        orchestrator._execute_data_acquisition = mock_stage_method

        # Execute stage
        success = orchestrator._execute_stage("data_acquisition")

        assert success is False
        assert (
            orchestrator.status.stages["data_acquisition"].status == StageStatus.FAILED
        )
        assert (
            "Test error" in orchestrator.status.stages["data_acquisition"].error_message
        )

    @patch(
        "diabetes_lstm_pipeline.orchestration.pipeline_orchestrator.DataAcquisitionPipeline"
    )
    def test_data_acquisition_stage(
        self, mock_acquisition_class, mock_config, sample_dataframe
    ):
        """Test data acquisition stage execution."""
        # Setup mock
        mock_acquisition = Mock()
        mock_acquisition.download_dataset.return_value = Path("dataset.zip")
        mock_acquisition.extract_dataset.return_value = Path("extracted")
        mock_acquisition.load_dataset.return_value = sample_dataframe
        mock_acquisition_class.return_value = mock_acquisition

        orchestrator = PipelineOrchestrator(mock_config)

        # Execute stage
        result = orchestrator._execute_data_acquisition(lambda x: None)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_dataframe)
        mock_acquisition.download_dataset.assert_called_once()
        mock_acquisition.extract_dataset.assert_called_once()
        mock_acquisition.load_dataset.assert_called_once()

    def test_parallel_stage_execution(self, mock_config, sample_dataframe):
        """Test parallel stage execution."""
        orchestrator = PipelineOrchestrator(mock_config)
        orchestrator.parallel_stages.add("data_acquisition")

        # Mock stage method
        def mock_stage_method(progress_callback):
            return sample_dataframe

        orchestrator._execute_data_acquisition = mock_stage_method

        # Mock parallel processor
        orchestrator.parallel_processor.check_resource_availability = Mock(
            return_value=True
        )
        orchestrator.parallel_processor.adjust_workers_based_on_resources = Mock()

        # Execute stage
        success = orchestrator._execute_stage("data_acquisition")

        assert success is True
        orchestrator.parallel_processor.check_resource_availability.assert_called_once()
        orchestrator.parallel_processor.adjust_workers_based_on_resources.assert_called_once()

    def test_error_recovery_integration(self, mock_config):
        """Test error recovery integration."""
        orchestrator = PipelineOrchestrator(mock_config)

        # Mock stage method that fails
        def mock_stage_method(progress_callback):
            raise ConnectionError("Network error")

        orchestrator._execute_data_acquisition = mock_stage_method

        # Execute stage
        success = orchestrator._execute_stage("data_acquisition")

        assert success is False
        assert len(orchestrator.error_recovery.error_history) > 0
        assert (
            orchestrator.error_recovery.error_history[0]["error_type"]
            == "ConnectionError"
        )

    def test_skip_stages_functionality(self, mock_config):
        """Test stage skipping functionality."""
        orchestrator = PipelineOrchestrator(mock_config)
        orchestrator.skip_stages.add("data_validation")

        # Mock successful data acquisition
        orchestrator.pipeline_data["data_acquisition"] = pd.DataFrame()

        # Run pipeline with limited stages
        with patch.object(orchestrator, "_execute_stage") as mock_execute:
            mock_execute.return_value = True

            result = orchestrator.run_pipeline(
                stages_to_run=["data_acquisition", "data_validation"]
            )

            # Verify data_validation was skipped
            assert (
                orchestrator.status.stages["data_validation"].status
                == StageStatus.SKIPPED
            )

    def test_resume_functionality(self, mock_config, temp_dir):
        """Test pipeline resume functionality."""
        orchestrator = PipelineOrchestrator(mock_config)

        # Create mock status file
        status_file = temp_dir / "test_status.json"

        # Create and save status
        status = PipelineStatus("test_pipeline")
        status.add_stage("data_acquisition")
        status.add_stage("data_validation")
        status.start_stage("data_acquisition")
        status.complete_stage("data_acquisition")
        status.start_stage("data_validation")
        status.fail_stage("data_validation", "Test failure")
        status.save_status(status_file)

        # Test resume
        with patch.object(orchestrator, "run_pipeline") as mock_run:
            mock_run.return_value = {"status": "completed"}

            result = orchestrator.resume_pipeline(status_file)

            mock_run.assert_called_once()

    def test_pipeline_results_generation(self, mock_config):
        """Test pipeline results generation."""
        orchestrator = PipelineOrchestrator(mock_config)

        # Setup mock pipeline data
        orchestrator.pipeline_data = {
            "data_acquisition": pd.DataFrame(),
            "training": {"model": Mock(), "metadata": {}},
        }

        # Setup status
        orchestrator.status.start_pipeline()
        orchestrator.status.add_stage("data_acquisition")
        orchestrator.status.complete_stage("data_acquisition")
        orchestrator.status.complete_pipeline()

        # Generate results
        results = orchestrator._generate_pipeline_results()

        assert "pipeline_id" in results
        assert "status" in results
        assert "duration" in results
        assert "stages_completed" in results
        assert "stage_results" in results
        assert results["stages_completed"] == 1

    def test_cleanup_resources(self, mock_config):
        """Test resource cleanup functionality."""
        orchestrator = PipelineOrchestrator(mock_config)

        # Mock cleanup methods
        orchestrator.error_recovery.cleanup_old_checkpoints = Mock()
        orchestrator.error_recovery.reset_error_history = Mock()

        # Set status to completed
        orchestrator.status.status = StageStatus.COMPLETED

        # Test cleanup
        orchestrator.cleanup_resources()

        orchestrator.error_recovery.cleanup_old_checkpoints.assert_called_once()
        orchestrator.error_recovery.reset_error_history.assert_called_once()

    @patch(
        "diabetes_lstm_pipeline.orchestration.pipeline_orchestrator.DataAcquisitionPipeline"
    )
    @patch("diabetes_lstm_pipeline.orchestration.pipeline_orchestrator.DataValidator")
    def test_full_pipeline_mock_execution(
        self,
        mock_validator_class,
        mock_acquisition_class,
        mock_config,
        sample_dataframe,
    ):
        """Test full pipeline execution with mocked components."""
        # Setup mocks
        mock_acquisition = Mock()
        mock_acquisition.download_dataset.return_value = Path("dataset.zip")
        mock_acquisition.extract_dataset.return_value = Path("extracted")
        mock_acquisition.load_dataset.return_value = sample_dataframe
        mock_acquisition_class.return_value = mock_acquisition

        mock_validator = Mock()
        mock_validator.validate_schema.return_value = Mock()
        mock_validator.assess_quality.return_value = Mock()
        mock_validator.detect_outliers.return_value = Mock()
        mock_validator_class.return_value = mock_validator

        orchestrator = PipelineOrchestrator(mock_config)

        # Mock all other stage methods to return simple results
        orchestrator._execute_preprocessing = Mock(return_value=sample_dataframe)
        orchestrator._execute_feature_engineering = Mock(return_value=sample_dataframe)
        orchestrator._execute_sequence_generation = Mock(
            return_value={"sequences": np.array([]), "splits": {}, "validation": True}
        )
        orchestrator._execute_model_building = Mock(return_value=Mock())
        orchestrator._execute_training = Mock(
            return_value={"model": Mock(), "metadata": {}}
        )
        orchestrator._execute_evaluation = Mock(
            return_value={"metrics": {}, "visualizations": {}}
        )
        orchestrator._execute_model_persistence = Mock(
            return_value={
                "model_path": "model.h5",
                "preprocessing_path": "preprocessing.pkl",
            }
        )

        # Run pipeline with first two stages only
        result = orchestrator.run_pipeline(
            stages_to_run=["data_acquisition", "data_validation"]
        )

        # Verify execution
        assert result["status"] == "completed"
        assert result["stages_completed"] == 2
        assert result["stages_failed"] == 0

    def test_progress_callback_functionality(self, mock_config):
        """Test progress callback functionality."""
        orchestrator = PipelineOrchestrator(mock_config)

        progress_values = []

        def mock_stage_method(progress_callback):
            progress_callback(25)
            progress_callback(50)
            progress_callback(75)
            progress_callback(100)
            return "result"

        orchestrator._execute_data_acquisition = mock_stage_method

        # Execute stage and capture progress
        success = orchestrator._execute_stage("data_acquisition")

        assert success is True
        stage = orchestrator.status.stages["data_acquisition"]
        assert stage.progress == 100.0

    def test_system_resource_monitoring(self, mock_config):
        """Test system resource monitoring integration."""
        orchestrator = PipelineOrchestrator(mock_config)

        # Test resource information
        resources = orchestrator.parallel_processor.get_system_resources()

        assert "cpu_count" in resources
        assert "memory_total_gb" in resources
        assert "max_workers" in resources

        # Test resource availability check
        availability = orchestrator.parallel_processor.check_resource_availability()
        assert isinstance(availability, bool)


class TestPipelineStatus:
    """Test suite for pipeline status tracking."""

    def test_pipeline_status_initialization(self):
        """Test pipeline status initialization."""
        status = PipelineStatus("test_pipeline")

        assert status.pipeline_id == "test_pipeline"
        assert status.status == StageStatus.NOT_STARTED
        assert len(status.stages) == 0

    def test_stage_lifecycle(self):
        """Test complete stage lifecycle."""
        status = PipelineStatus()

        # Add stage
        stage = status.add_stage("test_stage")
        assert stage.name == "test_stage"
        assert stage.status == StageStatus.NOT_STARTED

        # Start stage
        status.start_stage("test_stage")
        assert status.stages["test_stage"].status == StageStatus.RUNNING
        assert status.current_stage == "test_stage"

        # Update progress
        status.update_stage_progress("test_stage", 50.0)
        assert status.stages["test_stage"].progress == 50.0

        # Complete stage
        status.complete_stage("test_stage", {"result": "success"})
        assert status.stages["test_stage"].status == StageStatus.COMPLETED
        assert status.stages["test_stage"].progress == 100.0
        assert status.current_stage is None

    def test_pipeline_progress_calculation(self):
        """Test overall pipeline progress calculation."""
        status = PipelineStatus()

        # Add multiple stages
        status.add_stage("stage1")
        status.add_stage("stage2")
        status.add_stage("stage3")

        # Set different progress levels
        status.update_stage_progress("stage1", 100.0)  # Completed
        status.update_stage_progress("stage2", 50.0)  # Half done
        status.update_stage_progress("stage3", 0.0)  # Not started

        # Calculate overall progress
        overall_progress = status.get_overall_progress()
        expected_progress = (100.0 + 50.0 + 0.0) / 3
        assert overall_progress == expected_progress

    def test_status_persistence(self, tmp_path):
        """Test saving and loading pipeline status."""
        status = PipelineStatus("test_pipeline")

        # Setup status
        status.start_pipeline()
        status.add_stage("test_stage")
        status.start_stage("test_stage")
        status.complete_stage("test_stage")
        status.complete_pipeline()

        # Save status
        status_file = tmp_path / "test_status.json"
        status.save_status(status_file)

        assert status_file.exists()

        # Load status
        new_status = PipelineStatus()
        new_status.load_status(status_file)

        assert new_status.pipeline_id == "test_pipeline"
        assert new_status.status == StageStatus.COMPLETED
        assert "test_stage" in new_status.stages
        assert new_status.stages["test_stage"].status == StageStatus.COMPLETED


class TestErrorRecovery:
    """Test suite for error recovery functionality."""

    def test_error_recovery_initialization(self):
        """Test error recovery manager initialization."""
        config = {"max_retries": 3, "retry_delay": 30}
        recovery = ErrorRecoveryManager(config)

        assert recovery.max_retries == 3
        assert recovery.retry_delay == 30
        assert len(recovery.error_history) == 0

    def test_error_handling(self):
        """Test error handling and strategy determination."""
        recovery = ErrorRecoveryManager()
        status = PipelineStatus()

        # Test different error types
        memory_error = MemoryError("Out of memory")
        strategy = recovery.handle_error("test_stage", memory_error, status)
        assert strategy == RecoveryStrategy.RESTART_FROM_CHECKPOINT

        connection_error = ConnectionError("Network timeout")
        strategy = recovery.handle_error("test_stage", connection_error, status)
        assert strategy == RecoveryStrategy.RETRY

    def test_retry_count_management(self):
        """Test retry count management."""
        recovery = ErrorRecoveryManager()

        # Test retry count operations
        assert recovery.can_retry("test_stage") is True

        count = recovery.increment_retry_count("test_stage")
        assert count == 1

        recovery.reset_retry_count("test_stage")
        assert recovery.can_retry("test_stage") is True

    def test_checkpoint_operations(self, tmp_path):
        """Test checkpoint save and load operations."""
        config = {"checkpoint_dir": str(tmp_path)}
        recovery = ErrorRecoveryManager(config)

        # Save checkpoint
        test_data = {"key": "value", "number": 42}
        checkpoint_file = recovery.save_checkpoint("test_stage", test_data)

        assert checkpoint_file.exists()

        # Load checkpoint
        loaded_data = recovery.load_checkpoint("test_stage")
        assert loaded_data == test_data


class TestParallelProcessor:
    """Test suite for parallel processing functionality."""

    def test_parallel_processor_initialization(self):
        """Test parallel processor initialization."""
        config = {"max_workers": 4, "memory_limit_gb": 2.0}
        processor = ParallelProcessor(config)

        assert processor.max_workers == 4
        assert processor.memory_limit_gb == 2.0

    def test_parallel_processing(self):
        """Test basic parallel processing functionality."""
        processor = ParallelProcessor({"max_workers": 2, "use_processes": False})

        # Simple function to process
        def square_function(x):
            return x**2

        # Test data
        data = [1, 2, 3, 4, 5]

        # Process in parallel (using threads to avoid pickling issues)
        results = processor.process_in_parallel(square_function, data)

        expected = [1, 4, 9, 16, 25]
        assert results == expected

    def test_resource_monitoring(self):
        """Test system resource monitoring."""
        processor = ParallelProcessor()

        resources = processor.get_system_resources()

        assert "cpu_count" in resources
        assert "memory_total_gb" in resources
        assert "cpu_percent" in resources
        assert resources["cpu_count"] > 0
        assert resources["memory_total_gb"] > 0

    def test_processing_strategy_creation(self):
        """Test processing strategy creation."""
        processor = ParallelProcessor()

        # Test strategy for different data sizes and complexities
        strategy_small = processor.create_processing_strategy(100, "low")
        strategy_large = processor.create_processing_strategy(10000, "high")

        assert strategy_small["chunk_size"] > 0
        assert strategy_large["chunk_size"] > 0
        assert (
            strategy_small["chunk_size"] <= strategy_large["chunk_size"]
        )  # Larger chunks for simple tasks


if __name__ == "__main__":
    pytest.main([__file__])

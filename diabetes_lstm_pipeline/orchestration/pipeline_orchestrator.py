"""
Main pipeline orchestrator that coordinates all pipeline stages.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from datetime import datetime
import traceback

from .pipeline_status import PipelineStatus, StageStatus
from .error_recovery import ErrorRecoveryManager, RecoveryStrategy
from .parallel_processor import ParallelProcessor

# Import pipeline components
from ..data_acquisition.data_acquisition import DataAcquisitionPipeline
from ..data_validation.data_validation import DataValidator
from ..preprocessing.preprocessing import DataPreprocessor
from ..feature_engineering.feature_engineering import FeatureEngineer
from ..sequence_generation.sequence_generation import SequenceGenerationPipeline
from ..model_architecture.model_builder import LSTMModelBuilder
from ..training.model_trainer import ModelTrainer
from ..evaluation.clinical_metrics import ClinicalMetrics
from ..evaluation.visualization_generator import VisualizationGenerator
from ..model_persistence.model_persistence import ModelPersistence
from ..utils.config_manager import ConfigManager
from ..utils.logger import get_logger


class PipelineOrchestrator:
    """Orchestrates the complete diabetes LSTM pipeline execution."""

    def __init__(self, config: ConfigManager):
        """
        Initialize the pipeline orchestrator.

        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Initialize pipeline components
        self.status = PipelineStatus()
        self.error_recovery = ErrorRecoveryManager(config.get("error_recovery", {}))
        self.parallel_processor = ParallelProcessor(
            config.get("parallel_processing", {})
        )

        # Pipeline stages configuration
        self.stages = [
            "data_acquisition",
            "data_validation",
            "preprocessing",
            "feature_engineering",
            "sequence_generation",
            "model_building",
            "training",
            "evaluation",
            "model_persistence",
        ]

        # Initialize stage components
        self._initialize_components()

        # Pipeline state
        self.pipeline_data = {}
        self.skip_stages = set(config.get("pipeline.skip_stages", []))
        self.parallel_stages = set(config.get("pipeline.parallel_stages", []))

        self.logger.info("Pipeline orchestrator initialized")

    def _initialize_components(self) -> None:
        """Initialize all pipeline components."""
        try:
            # Data acquisition
            self.data_acquisition = DataAcquisitionPipeline(self.config.get("data", {}))

            # Data validation
            validation_config = self.config.get("validation", {})
            self.data_validator = DataValidator(validation_config)

            # Preprocessing
            preprocessing_config = self.config.get("preprocessing", {})
            self.preprocessor = DataPreprocessor(preprocessing_config)

            # Feature engineering
            feature_config = self.config.get("feature_engineering", {})
            self.feature_engineer = FeatureEngineer(feature_config)

            # Sequence generation
            sequence_config = self.config.get("sequence_generation", {})
            self.sequence_generator = SequenceGenerationPipeline(sequence_config)

            # Model building
            self.model_builder = LSTMModelBuilder(self.config.get_all())

            # Training
            self.trainer = ModelTrainer(self.config.get_all())

            # Evaluation
            evaluation_config = self.config.get("evaluation", {})
            self.clinical_metrics = ClinicalMetrics(
                target_range=tuple(
                    evaluation_config.get("target_glucose_range", [70.0, 180.0])
                ),
                hypoglycemia_threshold=evaluation_config.get(
                    "hypoglycemia_threshold", 70.0
                ),
                hyperglycemia_threshold=evaluation_config.get(
                    "hyperglycemia_threshold", 250.0
                ),
            )
            self.visualization_generator = VisualizationGenerator(
                evaluation_config.get("metrics_output_dir", "reports")
            )

            # Model persistence
            self.model_persistence = ModelPersistence(self.config.get_all())

            self.logger.info("All pipeline components initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline components: {e}")
            import traceback

            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def run_pipeline(
        self,
        resume_from: Optional[str] = None,
        stages_to_run: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline or resume from a specific stage.

        Args:
            resume_from: Stage name to resume from (if None, starts from beginning)
            stages_to_run: Specific stages to run (if None, runs all stages)

        Returns:
            Dictionary with pipeline results and metadata
        """
        self.logger.info("Starting diabetes LSTM pipeline execution")

        # Initialize pipeline status
        self.status.start_pipeline()

        # Determine which stages to run
        if stages_to_run:
            stages = [stage for stage in self.stages if stage in stages_to_run]
        else:
            stages = self.stages.copy()

        # Handle resume functionality
        if resume_from:
            try:
                resume_index = stages.index(resume_from)
                stages = stages[resume_index:]
                self.logger.info(f"Resuming pipeline from stage: {resume_from}")
            except ValueError:
                self.logger.error(f"Invalid resume stage: {resume_from}")
                raise ValueError(f"Stage '{resume_from}' not found in pipeline")

        # Add stages to status tracker
        for stage in stages:
            self.status.add_stage(stage)

        try:
            # Execute pipeline stages
            for stage_name in stages:
                if stage_name in self.skip_stages:
                    self.status.skip_stage(stage_name, "Configured to skip")
                    continue

                success = self._execute_stage(stage_name)

                if not success:
                    # Handle stage failure
                    recovery_strategy = self._handle_stage_failure(stage_name)

                    if recovery_strategy == RecoveryStrategy.MANUAL_INTERVENTION:
                        self.logger.error(
                            f"Manual intervention required for stage: {stage_name}"
                        )
                        break
                    elif recovery_strategy == RecoveryStrategy.SKIP:
                        self.status.skip_stage(stage_name, "Skipped due to error")
                        continue
                    elif recovery_strategy == RecoveryStrategy.RETRY:
                        # Retry the stage
                        if self.error_recovery.can_retry(stage_name):
                            self.error_recovery.increment_retry_count(stage_name)
                            self.logger.info(f"Retrying stage: {stage_name}")
                            success = self._execute_stage(stage_name)
                            if not success:
                                break
                        else:
                            self.logger.error(
                                f"Max retries exceeded for stage: {stage_name}"
                            )
                            break

            # Check if pipeline completed successfully
            failed_stages = self.status.get_failed_stages()
            if not failed_stages:
                self.status.complete_pipeline(
                    {
                        "total_duration": self.status.get_duration(),
                        "stages_completed": len(
                            [
                                s
                                for s in self.status.stages.values()
                                if s.status == StageStatus.COMPLETED
                            ]
                        ),
                    }
                )
                self.logger.info("Pipeline completed successfully")
            else:
                self.status.fail_pipeline(
                    f"Pipeline failed with {len(failed_stages)} failed stages"
                )
                self.logger.error(
                    f"Pipeline failed with {len(failed_stages)} failed stages"
                )

        except Exception as e:
            self.logger.error(f"Unexpected error in pipeline execution: {e}")
            self.status.fail_pipeline(str(e))
            raise

        finally:
            # Save pipeline status
            self._save_pipeline_status()

            # Print final status
            self.status.print_status()

        return self._generate_pipeline_results()

    def _execute_stage(self, stage_name: str) -> bool:
        """
        Execute a single pipeline stage.

        Args:
            stage_name: Name of the stage to execute

        Returns:
            True if stage completed successfully, False otherwise
        """
        self.logger.info(f"Starting stage: {stage_name}")
        stage = self.status.start_stage(stage_name)

        try:
            # Create progress callback
            def progress_callback(progress: float):
                self.status.update_stage_progress(stage_name, progress)

            # Execute the appropriate stage method
            stage_method = getattr(self, f"_execute_{stage_name}", None)
            if stage_method is None:
                raise NotImplementedError(f"Stage method not implemented: {stage_name}")

            # Check if stage should run in parallel
            if stage_name in self.parallel_stages:
                result = self._execute_stage_parallel(
                    stage_name, stage_method, progress_callback
                )
            else:
                result = stage_method(progress_callback)

            # Store stage result
            self.pipeline_data[stage_name] = result

            # Complete the stage
            metadata = {"result_type": type(result).__name__}
            if hasattr(result, "__len__"):
                metadata["result_size"] = len(result)

            self.status.complete_stage(stage_name, metadata)
            self.logger.info(f"Completed stage: {stage_name}")

            return True

        except Exception as e:
            self.logger.error(f"Error in stage {stage_name}: {e}")
            self.status.fail_stage(stage_name, str(e))

            # Handle error through recovery manager
            self.error_recovery.handle_error(stage_name, e, self.status)

            return False

    def _execute_stage_parallel(
        self,
        stage_name: str,
        stage_method: Callable,
        progress_callback: Callable[[float], None],
    ) -> Any:
        """
        Execute a stage with parallel processing support.

        Args:
            stage_name: Name of the stage
            stage_method: Method to execute
            progress_callback: Progress callback function

        Returns:
            Stage execution result
        """
        self.logger.info(f"Executing stage {stage_name} with parallel processing")

        # Check resource availability
        if not self.parallel_processor.check_resource_availability():
            self.logger.warning(
                f"Insufficient resources for parallel processing, running {stage_name} sequentially"
            )
            return stage_method(progress_callback)

        # Adjust workers based on current resources
        self.parallel_processor.adjust_workers_based_on_resources()

        return stage_method(progress_callback)

    def _handle_stage_failure(self, stage_name: str) -> RecoveryStrategy:
        """
        Handle failure of a pipeline stage.

        Args:
            stage_name: Name of the failed stage

        Returns:
            Recovery strategy to apply
        """
        # Get the last error for this stage
        stage = self.status.stages[stage_name]
        if stage.error_message:
            # Create a dummy exception for recovery analysis
            error = Exception(stage.error_message)
            return self.error_recovery.handle_error(stage_name, error, self.status)

        return RecoveryStrategy.MANUAL_INTERVENTION

    # Stage execution methods
    def _execute_data_acquisition(
        self, progress_callback: Callable[[float], None]
    ) -> Any:
        """Execute data acquisition stage."""
        progress_callback(10)

        # Check if force download is configured
        force_download = self.config.get("data.force_download", False)

        # Run the complete data acquisition pipeline
        dataset = self.data_acquisition.run_pipeline(force_download=force_download)
        progress_callback(100)

        return dataset

    def _execute_data_validation(
        self, progress_callback: Callable[[float], None]
    ) -> Any:
        """Execute data validation stage."""
        dataset = self.pipeline_data["data_acquisition"]

        progress_callback(20)

        # Run comprehensive validation
        validation_result, quality_report, outlier_report = (
            self.data_validator.validate_dataset(dataset)
        )
        progress_callback(100)

        return {
            "validation_result": validation_result,
            "quality_report": quality_report,
            "outlier_report": outlier_report,
        }

    def _execute_preprocessing(self, progress_callback: Callable[[float], None]) -> Any:
        """Execute preprocessing stage."""
        dataset = self.pipeline_data["data_acquisition"]

        progress_callback(10)

        # Run the complete preprocessing pipeline
        dataset, preprocessing_stats = self.preprocessor.preprocess(dataset)
        progress_callback(100)

        return dataset

    def _execute_feature_engineering(
        self, progress_callback: Callable[[float], None]
    ) -> Any:
        """Execute feature engineering stage."""
        dataset = self.pipeline_data["preprocessing"]

        progress_callback(10)

        # Run the complete feature engineering pipeline
        dataset = self.feature_engineer.engineer_features(dataset, fit_scaler=True)
        progress_callback(100)

        return dataset

    def _execute_sequence_generation(
        self, progress_callback: Callable[[float], None]
    ) -> Any:
        """Execute sequence generation stage."""
        dataset = self.pipeline_data["feature_engineering"]

        progress_callback(20)

        # Generate sequences using the correct method
        data_splits, metadata = (
            self.sequence_generator.generate_sequences_from_dataframe(dataset)
        )
        progress_callback(100)

        return {
            "splits": data_splits,
            "metadata": metadata,
        }

    def _execute_model_building(
        self, progress_callback: Callable[[float], None]
    ) -> Any:
        """Execute model building stage."""
        sequence_data = self.pipeline_data["sequence_generation"]

        progress_callback(30)

        # Get input shape from sequence metadata
        sequence_info = sequence_data["metadata"]["sequence_info"]
        input_shape = (
            sequence_info["sequence_length_points"],
            sequence_info["n_features"],
        )

        # Build model architecture
        model = self.model_builder.build_model(input_shape)
        progress_callback(70)

        # Compile model
        compiled_model = self.model_builder.compile_model(model)
        progress_callback(100)

        return compiled_model

    def _execute_training(self, progress_callback: Callable[[float], None]) -> Any:
        """Execute training stage."""
        model = self.pipeline_data["model_building"]
        sequence_data = self.pipeline_data["sequence_generation"]

        progress_callback(20)

        # Extract training data from sequence splits
        X_train, y_train = sequence_data["splits"]["train"]
        X_val, y_val = sequence_data["splits"]["val"]

        progress_callback(40)

        # Train model
        training_result = self.trainer.train(
            X_train, y_train, validation_data=(X_val, y_val)
        )

        progress_callback(100)

        return training_result

    def _execute_evaluation(self, progress_callback: Callable[[float], None]) -> Any:
        """Execute evaluation stage."""
        print("DEBUG: _execute_evaluation called")

        model = self.pipeline_data["model_building"]
        sequence_data = self.pipeline_data["sequence_generation"]

        print("DEBUG: model type:", type(model))
        print("DEBUG: sequence_data type:", type(sequence_data))
        print(
            "DEBUG: sequence_data keys:",
            (
                list(sequence_data.keys())
                if isinstance(sequence_data, dict)
                else "Not a dict"
            ),
        )

        if "splits" in sequence_data:
            print(
                "DEBUG: splits keys:",
                (
                    list(sequence_data["splits"].keys())
                    if isinstance(sequence_data["splits"], dict)
                    else "splits not a dict"
                ),
            )
            if "test" in sequence_data["splits"]:
                print("DEBUG: test data type:", type(sequence_data["splits"]["test"]))
                print(
                    "DEBUG: test data keys:",
                    (
                        list(sequence_data["splits"]["test"].keys())
                        if isinstance(sequence_data["splits"]["test"], dict)
                        else "test not a dict"
                    ),
                )

        progress_callback(20)

        # Convert tuple to dictionary format expected by evaluate_model
        X_test, y_test = sequence_data["splits"]["test"]
        test_data = {"X": X_test, "y": y_test}

        # Calculate clinical metrics
        print("DEBUG: About to call clinical_metrics.evaluate_model")
        metrics = self.clinical_metrics.evaluate_model(model, test_data)
        print("DEBUG: clinical_metrics.evaluate_model completed")
        progress_callback(60)

        # Generate visualizations
        print("DEBUG: About to call visualization_generator.generate_evaluation_plots")
        visualizations = self.visualization_generator.generate_evaluation_plots(
            model, test_data, metrics
        )
        print("DEBUG: visualization_generator.generate_evaluation_plots completed")
        progress_callback(100)

        return {"metrics": metrics, "visualizations": visualizations}

    def _execute_model_persistence(
        self, progress_callback: Callable[[float], None]
    ) -> Any:
        """Execute model persistence stage."""
        model = self.pipeline_data[
            "training"
        ]  # The training stage returns the model directly
        training_metadata = (
            self.trainer.get_training_summary()
        )  # Get metadata from trainer
        evaluation_result = self.pipeline_data["evaluation"]

        progress_callback(30)

        # Save model (includes preprocessing components)
        result = self.model_persistence.save_model(
            model,
            self.pipeline_data["feature_engineering"],  # preprocessing_components
            training_metadata,
            evaluation_result["metrics"],  # performance_metrics
        )
        progress_callback(100)

        return result

    def _save_pipeline_status(self) -> None:
        """Save pipeline status to file."""
        try:
            status_dir = Path("logs/pipeline_status")
            status_dir.mkdir(parents=True, exist_ok=True)

            status_file = status_dir / f"pipeline_status_{self.status.pipeline_id}.json"
            self.status.save_status(status_file)

            self.logger.info(f"Pipeline status saved to: {status_file}")

        except Exception as e:
            self.logger.error(f"Failed to save pipeline status: {e}")

    def _generate_pipeline_results(self) -> Dict[str, Any]:
        """
        Generate comprehensive pipeline results.

        Returns:
            Dictionary with pipeline results and metadata
        """
        return {
            "pipeline_id": self.status.pipeline_id,
            "status": self.status.status.value,
            "duration": self.status.get_duration(),
            "stages_completed": len(
                [
                    s
                    for s in self.status.stages.values()
                    if s.status == StageStatus.COMPLETED
                ]
            ),
            "stages_failed": len(self.status.get_failed_stages()),
            "overall_progress": self.status.get_overall_progress(),
            "stage_results": self.pipeline_data,
            "error_recovery_report": self.error_recovery.generate_recovery_report(),
            "system_resources": self.parallel_processor.get_system_resources(),
        }

    def get_pipeline_status(self) -> PipelineStatus:
        """
        Get current pipeline status.

        Returns:
            PipelineStatus object
        """
        return self.status

    def resume_pipeline(self, status_file: Path) -> Dict[str, Any]:
        """
        Resume pipeline from saved status.

        Args:
            status_file: Path to saved pipeline status file

        Returns:
            Pipeline results
        """
        self.logger.info(f"Resuming pipeline from status file: {status_file}")

        # Load previous status
        self.status.load_status(status_file)

        # Determine restart point
        restart_stage = self.error_recovery.get_restart_point(self.status)

        if restart_stage:
            self.logger.info(f"Resuming from stage: {restart_stage}")
            return self.run_pipeline(resume_from=restart_stage)
        else:
            self.logger.info("No valid restart point found, starting from beginning")
            return self.run_pipeline()

    def cleanup_resources(self) -> None:
        """Clean up pipeline resources."""
        try:
            # Clean up old checkpoints
            self.error_recovery.cleanup_old_checkpoints()

            # Reset error history if pipeline completed successfully
            if self.status.status == StageStatus.COMPLETED:
                self.error_recovery.reset_error_history()

            self.logger.info("Pipeline resources cleaned up")

        except Exception as e:
            self.logger.error(f"Error during resource cleanup: {e}")

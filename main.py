#!/usr/bin/env python3
"""
Main entry point for the Diabetes LSTM Pipeline.
This script provides a command-line interface for pipeline execution with configurable parameters.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, List

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from diabetes_lstm_pipeline.utils import get_config, setup_logging, get_logger
from diabetes_lstm_pipeline.orchestration import PipelineOrchestrator


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Diabetes LSTM Pipeline - Blood glucose prediction system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --run                                    # Run complete pipeline
  %(prog)s --run --stages data_acquisition training # Run specific stages
  %(prog)s --resume logs/pipeline_status/status.json # Resume from saved status
  %(prog)s --config configs/custom_config.yaml --run # Use custom configuration
  %(prog)s --status                                 # Show pipeline status
  %(prog)s --run --force-download                   # Force re-download of dataset
        """,
    )

    # Main actions
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--run", action="store_true", help="Run the complete pipeline"
    )
    action_group.add_argument(
        "--resume",
        type=str,
        metavar="STATUS_FILE",
        help="Resume pipeline from saved status file",
    )
    action_group.add_argument(
        "--status", action="store_true", help="Show current pipeline status"
    )
    action_group.add_argument(
        "--init", action="store_true", help="Initialize pipeline and show configuration"
    )

    # Configuration options
    parser.add_argument(
        "--config",
        type=str,
        metavar="CONFIG_FILE",
        help="Path to configuration file (default: configs/default_config.yaml)",
    )

    # Pipeline execution options
    parser.add_argument(
        "--stages",
        nargs="+",
        metavar="STAGE",
        choices=[
            "data_acquisition",
            "data_validation",
            "preprocessing",
            "feature_engineering",
            "sequence_generation",
            "model_building",
            "training",
            "evaluation",
            "model_persistence",
        ],
        help="Specific stages to run (default: all stages)",
    )

    parser.add_argument(
        "--resume-from", type=str, metavar="STAGE", help="Stage to resume from"
    )

    parser.add_argument(
        "--skip-stages",
        nargs="+",
        metavar="STAGE",
        help="Stages to skip during execution",
    )

    parser.add_argument(
        "--parallel",
        nargs="+",
        metavar="STAGE",
        help="Stages to run with parallel processing",
    )

    # Output options
    parser.add_argument(
        "--output-dir", type=str, metavar="DIR", help="Output directory for results"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )

    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    # Resource management
    parser.add_argument(
        "--max-workers",
        type=int,
        metavar="N",
        help="Maximum number of parallel workers",
    )

    parser.add_argument(
        "--memory-limit",
        type=float,
        metavar="GB",
        help="Memory limit in GB for parallel processing",
    )

    # Data acquisition options
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of dataset even if already available",
    )

    return parser


def setup_pipeline_config(args: argparse.Namespace) -> None:
    """Setup pipeline configuration based on command-line arguments."""
    # Load base configuration
    config = get_config(args.config)

    # Override configuration with command-line arguments
    if args.skip_stages:
        config.set("pipeline.skip_stages", args.skip_stages)

    if args.parallel:
        config.set("pipeline.parallel_stages", args.parallel)

    if args.output_dir:
        config.set("output.base_dir", args.output_dir)

    if args.log_level:
        config.set("logging.level", args.log_level)

    if args.max_workers:
        config.set("parallel_processing.max_workers", args.max_workers)

    if args.memory_limit:
        config.set("parallel_processing.memory_limit_gb", args.memory_limit)

    # Data acquisition options
    if args.force_download:
        config.set("data.force_download", True)

    # Setup logging
    logging_config = config.get("logging", {})
    if args.quiet:
        logging_config["level"] = "ERROR"
    elif args.verbose:
        logging_config["level"] = "DEBUG"

    setup_logging(logging_config)


def run_pipeline(args: argparse.Namespace) -> int:
    """Run the pipeline with specified arguments."""
    logger = get_logger("main")

    try:
        # Setup configuration
        setup_pipeline_config(args)
        config = get_config()

        # Initialize orchestrator
        orchestrator = PipelineOrchestrator(config)

        logger.info("Starting diabetes LSTM pipeline execution")

        if not args.quiet:
            print("🚀 Starting Diabetes LSTM Pipeline")
            print("=" * 50)

        # Run pipeline
        results = orchestrator.run_pipeline(
            resume_from=args.resume_from, stages_to_run=args.stages
        )

        # Display results
        if not args.quiet:
            print("\n📊 Pipeline Results:")
            print(f"  Status: {results['status']}")
            print(f"  Duration: {results.get('duration', 0):.2f} seconds")
            print(f"  Stages Completed: {results['stages_completed']}")
            print(f"  Stages Failed: {results['stages_failed']}")
            print(f"  Overall Progress: {results['overall_progress']:.1f}%")

        # Cleanup resources
        orchestrator.cleanup_resources()

        if results["status"] == "completed":
            logger.info("Pipeline completed successfully")
            return 0
        else:
            logger.error("Pipeline failed")
            return 1

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        if args.verbose:
            import traceback

            logger.error(traceback.format_exc())
        return 1


def resume_pipeline(args: argparse.Namespace) -> int:
    """Resume pipeline from saved status."""
    logger = get_logger("main")

    try:
        # Setup configuration
        setup_pipeline_config(args)
        config = get_config()

        # Initialize orchestrator
        orchestrator = PipelineOrchestrator(config)

        # Resume pipeline
        status_file = Path(args.resume)
        if not status_file.exists():
            logger.error(f"Status file not found: {status_file}")
            return 1

        logger.info(f"Resuming pipeline from: {status_file}")

        if not args.quiet:
            print(f"🔄 Resuming Pipeline from: {status_file}")
            print("=" * 50)

        results = orchestrator.resume_pipeline(status_file)

        # Display results
        if not args.quiet:
            print("\n📊 Pipeline Results:")
            print(f"  Status: {results['status']}")
            print(f"  Duration: {results.get('duration', 0):.2f} seconds")
            print(f"  Stages Completed: {results['stages_completed']}")
            print(f"  Stages Failed: {results['stages_failed']}")

        # Cleanup resources
        orchestrator.cleanup_resources()

        return 0 if results["status"] == "completed" else 1

    except Exception as e:
        logger.error(f"Pipeline resume failed: {e}")
        return 1


def show_status(args: argparse.Namespace) -> int:
    """Show current pipeline status."""
    try:
        # Look for recent status files
        status_dir = Path("logs/pipeline_status")

        if not status_dir.exists():
            print("No pipeline status files found.")
            return 0

        status_files = list(status_dir.glob("pipeline_status_*.json"))

        if not status_files:
            print("No pipeline status files found.")
            return 0

        # Show the most recent status file
        latest_status = sorted(status_files, key=lambda x: x.stat().st_mtime)[-1]

        print(f"📋 Latest Pipeline Status: {latest_status.name}")
        print("=" * 50)

        # Load and display status
        from diabetes_lstm_pipeline.orchestration import PipelineStatus

        status = PipelineStatus()
        status.load_status(latest_status)
        status.print_status()

        return 0

    except Exception as e:
        print(f"Error showing status: {e}")
        return 1


def initialize_pipeline(args: argparse.Namespace) -> int:
    """Initialize pipeline and show configuration."""
    try:
        # Setup configuration
        setup_pipeline_config(args)
        config = get_config()

        logger = get_logger("main")
        logger.info("Diabetes LSTM Pipeline initialized")

        print("🔧 Diabetes LSTM Pipeline Configuration")
        print("=" * 50)

        # Display key configuration values
        print("\n📁 Data Configuration:")
        print(f"  Raw data path: {config.get('data.raw_data_path')}")
        print(f"  Processed data path: {config.get('data.processed_data_path')}")
        print(f"  Dataset URL: {config.get('data.dataset_url')}")

        print("\n🧠 Model Configuration:")
        print(f"  Sequence length: {config.get('model.sequence_length')}")
        print(f"  LSTM units: {config.get('model.lstm_units')}")
        print(f"  Dropout rate: {config.get('model.dropout_rate')}")
        print(f"  Learning rate: {config.get('model.learning_rate')}")

        print("\n🏋️ Training Configuration:")
        print(f"  Batch size: {config.get('training.batch_size')}")
        print(f"  Epochs: {config.get('training.epochs')}")
        print(f"  Validation split: {config.get('training.validation_split')}")

        print("\n📊 Logging Configuration:")
        print(f"  Log level: {config.get('logging.level')}")
        print(f"  Log file: {config.get('logging.file')}")

        print("\n🔧 System Configuration:")
        print(f"  Random seed: {config.get('random_seed')}")

        print("\n✅ Pipeline initialized successfully!")
        print("Use --run to execute the complete pipeline.")

        return 0

    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        return 1


def main():
    """Main entry point for the diabetes LSTM pipeline."""
    parser = create_argument_parser()
    args = parser.parse_args()

    try:
        if args.run:
            return run_pipeline(args)
        elif args.resume:
            return resume_pipeline(args)
        elif args.status:
            return show_status(args)
        elif args.init:
            return initialize_pipeline(args)
        else:
            parser.print_help()
            return 1

    except KeyboardInterrupt:
        print("\n⏹️ Pipeline execution interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

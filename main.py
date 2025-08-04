#!/usr/bin/env python3
"""
Main entry point for the Diabetes LSTM Pipeline.
This script demonstrates the complete pipeline setup and initialization.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from diabetes_lstm_pipeline.utils import get_config, setup_logging, get_logger


def main():
    """Main entry point for the diabetes LSTM pipeline."""
    print("Diabetes LSTM Pipeline")
    print("=" * 30)

    try:
        # Load configuration
        config = get_config()
        print(f"✓ Configuration loaded from: {config.config_path}")

        # Setup logging
        logging_config = config.get("logging", {})
        setup_logging(logging_config)

        # Get logger for main module
        logger = get_logger("main")
        logger.info("Diabetes LSTM Pipeline started")

        # Display key configuration values
        print("\nKey Configuration Values:")
        print(f"  - Data path: {config.get('data.raw_data_path')}")
        print(f"  - Processed data path: {config.get('data.processed_data_path')}")
        print(f"  - Log level: {config.get('logging.level')}")
        print(f"  - Sequence length: {config.get('model.sequence_length')}")
        print(f"  - LSTM units: {config.get('model.lstm_units')}")
        print(f"  - Random seed: {config.get('random_seed')}")

        logger.info("Pipeline initialization completed successfully")
        print("\n✓ Pipeline setup completed successfully!")
        print("Ready for data processing and model training.")

        return 0

    except Exception as e:
        print(f"✗ Pipeline initialization failed: {e}")
        if "logger" in locals():
            logger.error(f"Pipeline initialization failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

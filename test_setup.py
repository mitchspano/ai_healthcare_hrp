#!/usr/bin/env python3
"""
Test script to verify the project setup is working correctly.
This script tests the configuration management and logging systems.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from diabetes_lstm_pipeline.utils import get_config, setup_logging, get_logger


def test_configuration_system():
    """Test the configuration management system."""
    print("Testing configuration system...")

    try:
        # Load default configuration
        config = get_config()

        # Test getting configuration values
        data_path = config.get("data.raw_data_path")
        log_level = config.get("logging.level")
        model_units = config.get("model.lstm_units")

        print(f"✓ Configuration loaded successfully")
        print(f"  - Data path: {data_path}")
        print(f"  - Log level: {log_level}")
        print(f"  - LSTM units: {model_units}")

        # Test setting a value
        config.set("test.value", "test_success")
        test_value = config.get("test.value")

        if test_value == "test_success":
            print("✓ Configuration set/get working correctly")
        else:
            print("✗ Configuration set/get failed")
            return False

        return True

    except Exception as e:
        print(f"✗ Configuration system failed: {e}")
        return False


def test_logging_system():
    """Test the logging infrastructure."""
    print("\nTesting logging system...")

    try:
        # Get configuration for logging
        config = get_config()
        logging_config = config.get("logging", {})

        # Setup logging
        setup_logging(logging_config)

        # Get a logger and test it
        logger = get_logger("test_module")

        logger.info("This is a test info message")
        logger.warning("This is a test warning message")
        logger.debug("This is a test debug message")

        print("✓ Logging system initialized successfully")
        print("✓ Test messages logged (check logs/pipeline.log)")

        return True

    except Exception as e:
        print(f"✗ Logging system failed: {e}")
        return False


def test_directory_structure():
    """Test that all required directories exist."""
    print("\nTesting directory structure...")

    required_dirs = [
        "diabetes_lstm_pipeline",
        "diabetes_lstm_pipeline/data_acquisition",
        "diabetes_lstm_pipeline/data_validation",
        "diabetes_lstm_pipeline/preprocessing",
        "diabetes_lstm_pipeline/feature_engineering",
        "diabetes_lstm_pipeline/sequence_generation",
        "diabetes_lstm_pipeline/model_architecture",
        "diabetes_lstm_pipeline/training",
        "diabetes_lstm_pipeline/evaluation",
        "diabetes_lstm_pipeline/utils",
        "tests",
        "tests/unit",
        "tests/integration",
        "configs",
        "data",
        "data/raw",
        "data/processed",
        "logs",
        "models",
    ]

    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)

    if missing_dirs:
        print(f"✗ Missing directories: {missing_dirs}")
        return False
    else:
        print("✓ All required directories exist")
        return True


def main():
    """Run all setup tests."""
    print("Diabetes LSTM Pipeline - Setup Verification")
    print("=" * 50)

    tests = [test_directory_structure, test_configuration_system, test_logging_system]

    results = []
    for test in tests:
        results.append(test())

    print("\n" + "=" * 50)
    if all(results):
        print("✓ All setup tests passed! The project is ready for development.")
        return 0
    else:
        print("✗ Some setup tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

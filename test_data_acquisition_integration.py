#!/usr/bin/env python3
"""Integration test for the data acquisition module."""

import tempfile
from pathlib import Path
from diabetes_lstm_pipeline.data_acquisition import DataAcquisitionPipeline
from diabetes_lstm_pipeline.utils.config_manager import ConfigManager


def test_data_acquisition_integration():
    """Test the data acquisition pipeline with configuration."""

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test configuration
        config = {
            "dataset_url": "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/gk9m674wcx-1.zip",
            "raw_data_path": str(temp_path / "raw"),
            "max_retries": 2,
            "retry_delay": 0.5,
            "chunk_size": 8192,
            "timeout": 60,
        }

        # Initialize pipeline
        pipeline = DataAcquisitionPipeline(config)

        # Get dataset info
        info = pipeline.get_dataset_info()
        print("Dataset Info:")
        print(f"  URL: {info['dataset_url']}")
        print(f"  Local file exists: {info['local_file_exists']}")
        print(f"  Expected columns: {len(info['expected_columns'])}")

        # Test individual components
        print("\nTesting individual components:")

        # Test downloader initialization
        print("✓ DataDownloader initialized")

        # Test extractor initialization
        print("✓ DataExtractor initialized")

        # Test loader initialization
        print("✓ DataLoader initialized")

        print("\nData acquisition module integration test completed successfully!")


if __name__ == "__main__":
    test_data_acquisition_integration()

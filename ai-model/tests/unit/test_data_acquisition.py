"""Unit tests for the data acquisition module."""

import hashlib
import json
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import pytest
import requests

from diabetes_lstm_pipeline.data_acquisition import (
    DataDownloader,
    DataExtractor,
    DataLoader,
    DataAcquisitionPipeline,
)


class TestDataDownloader:
    """Test cases for DataDownloader class."""

    @pytest.fixture
    def config(self):
        """Test configuration."""
        return {
            "dataset_url": "https://example.com/test-dataset.zip",
            "raw_data_path": "test_data/raw",
            "max_retries": 2,
            "retry_delay": 0.1,
            "chunk_size": 1024,
            "timeout": 30,
        }

    @pytest.fixture
    def downloader(self, config, tmp_path):
        """Create DataDownloader instance with temporary directory."""
        config["raw_data_path"] = str(tmp_path / "raw")
        return DataDownloader(config)

    def test_init(self, downloader, tmp_path):
        """Test DataDownloader initialization."""
        assert downloader.dataset_url == "https://example.com/test-dataset.zip"
        assert downloader.download_path == tmp_path / "raw"
        assert downloader.max_retries == 2
        assert downloader.retry_delay == 0.1
        assert downloader.chunk_size == 1024
        assert downloader.timeout == 30
        assert downloader.download_path.exists()

    def test_calculate_file_hash(self, downloader, tmp_path):
        """Test file hash calculation."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_content = b"Hello, World!"
        test_file.write_bytes(test_content)

        # Calculate expected hash
        expected_hash = hashlib.md5(test_content).hexdigest()

        # Test hash calculation
        actual_hash = downloader._calculate_file_hash(test_file)
        assert actual_hash == expected_hash

    @patch("requests.Session.head")
    def test_get_remote_file_info(self, mock_head, downloader):
        """Test getting remote file information."""
        # Mock response
        mock_response = Mock()
        mock_response.headers = {
            "content-length": "12345",
            "last-modified": "Wed, 21 Oct 2015 07:28:00 GMT",
            "etag": '"abc123"',
        }
        mock_head.return_value = mock_response

        # Test
        info = downloader._get_remote_file_info("https://example.com/file.zip")

        assert info["content_length"] == 12345
        assert info["last_modified"] == "Wed, 21 Oct 2015 07:28:00 GMT"
        assert info["etag"] == "abc123"

    @patch("requests.Session.head")
    def test_get_remote_file_info_error(self, mock_head, downloader):
        """Test getting remote file info with error."""
        mock_head.side_effect = requests.RequestException("Network error")

        info = downloader._get_remote_file_info("https://example.com/file.zip")
        assert info == {}

    def test_should_download_file_not_exists(self, downloader, tmp_path):
        """Test should_download when file doesn't exist."""
        non_existent_file = tmp_path / "nonexistent.zip"
        remote_info = {"content_length": 12345}

        assert downloader._should_download(non_existent_file, remote_info) is True

    def test_should_download_size_mismatch(self, downloader, tmp_path):
        """Test should_download with size mismatch."""
        test_file = tmp_path / "test.zip"
        test_file.write_bytes(b"small content")

        remote_info = {"content_length": 99999}

        assert downloader._should_download(test_file, remote_info) is True

    def test_should_download_size_match(self, downloader, tmp_path):
        """Test should_download with matching size."""
        test_content = b"test content"
        test_file = tmp_path / "test.zip"
        test_file.write_bytes(test_content)

        remote_info = {"content_length": len(test_content)}

        assert downloader._should_download(test_file, remote_info) is False

    @patch("requests.Session.get")
    @patch("requests.Session.head")
    def test_download_dataset_success(self, mock_head, mock_get, downloader):
        """Test successful dataset download."""
        # Mock head response
        mock_head_response = Mock()
        mock_head_response.headers = {"content-length": "13"}
        mock_head.return_value = mock_head_response

        # Mock get response
        test_content = b"Hello, World!"
        mock_response = Mock()
        mock_response.headers = {"content-length": str(len(test_content))}
        mock_response.iter_content.return_value = [test_content]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Test download
        result_path = downloader.download_dataset(force_download=True)

        assert result_path.exists()
        assert result_path.read_bytes() == test_content
        assert result_path.name == "test-dataset.zip"

    def test_download_dataset_no_url(self, tmp_path):
        """Test download with no URL configured."""
        config = {"raw_data_path": str(tmp_path)}
        downloader = DataDownloader(config)

        with pytest.raises(ValueError, match="Dataset URL not configured"):
            downloader.download_dataset()

    @patch("requests.Session.get")
    def test_download_dataset_retry_logic(self, mock_get, downloader):
        """Test download retry logic."""
        # First call fails, second succeeds
        mock_get.side_effect = [
            requests.RequestException("Network error"),
            Mock(
                headers={"content-length": "5"},
                iter_content=lambda chunk_size: [b"hello"],
                raise_for_status=lambda: None,
            ),
        ]

        with patch("time.sleep"):  # Speed up test
            result_path = downloader.download_dataset(force_download=True)

        assert result_path.exists()
        assert mock_get.call_count == 2

    @patch("requests.Session.get")
    def test_download_dataset_max_retries_exceeded(self, mock_get, downloader):
        """Test download failure after max retries."""
        mock_get.side_effect = requests.RequestException("Persistent error")

        with patch("time.sleep"):  # Speed up test
            with pytest.raises(requests.RequestException):
                downloader.download_dataset(force_download=True)

    def test_verify_download_integrity_file_not_exists(self, downloader, tmp_path):
        """Test integrity verification with non-existent file."""
        non_existent_file = tmp_path / "nonexistent.zip"

        assert downloader.verify_download_integrity(non_existent_file) is False

    def test_verify_download_integrity_empty_file(self, downloader, tmp_path):
        """Test integrity verification with empty file."""
        empty_file = tmp_path / "empty.zip"
        empty_file.touch()

        assert downloader.verify_download_integrity(empty_file) is False

    def test_verify_download_integrity_valid_file(self, downloader, tmp_path):
        """Test integrity verification with valid file."""
        test_file = tmp_path / "test.zip"
        test_content = b"Valid content"
        test_file.write_bytes(test_content)

        assert downloader.verify_download_integrity(test_file) is True

    def test_verify_download_integrity_with_hash(self, downloader, tmp_path):
        """Test integrity verification with hash check."""
        test_file = tmp_path / "test.zip"
        test_content = b"Test content for hash"
        test_file.write_bytes(test_content)

        expected_hash = hashlib.md5(test_content).hexdigest()

        assert downloader.verify_download_integrity(test_file, expected_hash) is True

    def test_verify_download_integrity_hash_mismatch(self, downloader, tmp_path):
        """Test integrity verification with hash mismatch."""
        test_file = tmp_path / "test.zip"
        test_file.write_bytes(b"Test content")

        wrong_hash = "wrong_hash_value"

        assert downloader.verify_download_integrity(test_file, wrong_hash) is False


class TestDataExtractor:
    """Test cases for DataExtractor class."""

    @pytest.fixture
    def config(self):
        """Test configuration."""
        return {"raw_data_path": "test_data/raw"}

    @pytest.fixture
    def extractor(self, config, tmp_path):
        """Create DataExtractor instance with temporary directory."""
        config["raw_data_path"] = str(tmp_path / "raw")
        return DataExtractor(config)

    @pytest.fixture
    def sample_zip(self, tmp_path):
        """Create a sample ZIP file for testing."""
        zip_path = tmp_path / "sample.zip"

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("file1.txt", "Content of file 1")
            zf.writestr("file2.csv", "col1,col2\nval1,val2")
            zf.writestr("subdir/file3.txt", "Content of file 3")

        return zip_path

    def test_init(self, extractor, tmp_path):
        """Test DataExtractor initialization."""
        assert extractor.extract_path == tmp_path / "raw"
        assert extractor.extract_path.exists()

    def test_extract_dataset_success(self, extractor, sample_zip, tmp_path):
        """Test successful dataset extraction."""
        extract_dir = extractor.extract_dataset(sample_zip)

        assert extract_dir.exists()
        assert (extract_dir / "file1.txt").exists()
        assert (extract_dir / "file2.csv").exists()
        assert (extract_dir / "subdir" / "file3.txt").exists()

        # Check file contents
        assert (extract_dir / "file1.txt").read_text() == "Content of file 1"
        assert (extract_dir / "file2.csv").read_text() == "col1,col2\nval1,val2"

    def test_extract_dataset_file_not_found(self, extractor, tmp_path):
        """Test extraction with non-existent ZIP file."""
        non_existent_zip = tmp_path / "nonexistent.zip"

        with pytest.raises(FileNotFoundError):
            extractor.extract_dataset(non_existent_zip)

    def test_extract_dataset_corrupted_zip(self, extractor, tmp_path):
        """Test extraction with corrupted ZIP file."""
        corrupted_zip = tmp_path / "corrupted.zip"
        corrupted_zip.write_bytes(b"This is not a valid ZIP file")

        with pytest.raises(zipfile.BadZipFile):
            extractor.extract_dataset(corrupted_zip)

    def test_extract_dataset_custom_location(self, extractor, sample_zip, tmp_path):
        """Test extraction to custom location."""
        custom_dir = tmp_path / "custom_extract"

        extract_dir = extractor.extract_dataset(sample_zip, custom_dir)

        assert extract_dir == custom_dir
        assert (custom_dir / "file1.txt").exists()

    def test_list_archive_contents(self, extractor, sample_zip):
        """Test listing archive contents."""
        contents = extractor.list_archive_contents(sample_zip)

        expected_files = ["file1.txt", "file2.csv", "subdir/file3.txt"]
        assert sorted(contents) == sorted(expected_files)

    def test_list_archive_contents_file_not_found(self, extractor, tmp_path):
        """Test listing contents of non-existent archive."""
        non_existent_zip = tmp_path / "nonexistent.zip"

        with pytest.raises(FileNotFoundError):
            extractor.list_archive_contents(non_existent_zip)

    def test_validate_archive_valid(self, extractor, sample_zip):
        """Test validation of valid archive."""
        assert extractor.validate_archive(sample_zip) is True

    def test_validate_archive_corrupted(self, extractor, tmp_path):
        """Test validation of corrupted archive."""
        corrupted_zip = tmp_path / "corrupted.zip"
        corrupted_zip.write_bytes(b"Not a ZIP file")

        assert extractor.validate_archive(corrupted_zip) is False


class TestDataLoader:
    """Test cases for DataLoader class."""

    @pytest.fixture
    def config(self):
        """Test configuration."""
        return {"raw_data_path": "test_data/raw"}

    @pytest.fixture
    def loader(self, config, tmp_path):
        """Create DataLoader instance with temporary directory."""
        config["raw_data_path"] = str(tmp_path / "raw")
        return DataLoader(config)

    @pytest.fixture
    def sample_csv_data(self):
        """Sample CSV data matching AZT1D schema."""
        return {
            "EventDateTime": [
                "2023-01-01 10:00:00",
                "2023-01-01 10:05:00",
                "2023-01-01 10:10:00",
            ],
            "DeviceMode": ["Manual", "Auto", "Manual"],
            "BolusType": ["Normal", "Extended", "Normal"],
            "Basal": [1.2, 1.3, 1.1],
            "CorrectionDelivered": [0.0, 2.5, 0.0],
            "TotalBolusInsulinDelivered": [3.5, 0.0, 4.2],
            "FoodDelivered": [0.0, 1.0, 0.0],
            "CarbSize": [45.0, 0.0, 60.0],
            "CGM": [120.5, 145.2, 110.8],
        }

    @pytest.fixture
    def sample_csv_file(self, tmp_path, sample_csv_data):
        """Create sample CSV file."""
        csv_path = tmp_path / "sample.csv"
        df = pd.DataFrame(sample_csv_data)
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_init(self, loader, tmp_path):
        """Test DataLoader initialization."""
        assert loader.data_path == tmp_path / "raw"
        assert len(loader.expected_columns) == 9
        assert "EventDateTime" in loader.expected_columns
        assert "CGM" in loader.expected_columns

    def test_infer_csv_files(self, loader, tmp_path):
        """Test CSV file discovery."""
        # Create test CSV files
        (tmp_path / "file1.csv").touch()
        (tmp_path / "file2.csv").touch()
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "file3.csv").touch()
        (tmp_path / "not_csv.txt").touch()

        csv_files = loader._infer_csv_files(tmp_path)

        assert len(csv_files) == 3
        csv_names = [f.name for f in csv_files]
        assert "file1.csv" in csv_names
        assert "file2.csv" in csv_names
        assert "file3.csv" in csv_names

    def test_parse_datetime(self, loader, sample_csv_data):
        """Test datetime parsing."""
        df = pd.DataFrame(sample_csv_data)

        # Initially, EventDateTime is string
        assert df["EventDateTime"].dtype == "object"

        # Parse datetime
        df_parsed = loader._parse_datetime(df)

        # Should be datetime now
        assert pd.api.types.is_datetime64_any_dtype(df_parsed["EventDateTime"])

    def test_parse_datetime_missing_column(self, loader):
        """Test datetime parsing with missing column."""
        df = pd.DataFrame({"other_col": [1, 2, 3]})

        # Should not raise error
        df_result = loader._parse_datetime(df)
        assert df_result.equals(df)

    def test_load_csv_file_success(self, loader, sample_csv_file):
        """Test successful CSV file loading."""
        df = loader.load_csv_file(sample_csv_file)

        assert len(df) == 3
        assert len(df.columns) == 10  # 9 original columns + participant_id
        assert "EventDateTime" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["EventDateTime"])
        assert df["DeviceMode"].dtype.name == "category"

    def test_load_csv_file_not_found(self, loader, tmp_path):
        """Test loading non-existent CSV file."""
        non_existent_file = tmp_path / "nonexistent.csv"

        with pytest.raises(FileNotFoundError):
            loader.load_csv_file(non_existent_file)

    def test_load_csv_file_empty(self, loader, tmp_path):
        """Test loading empty CSV file."""
        empty_csv = tmp_path / "empty.csv"
        empty_csv.touch()

        with pytest.raises(pd.errors.EmptyDataError):
            loader.load_csv_file(empty_csv)

    def test_load_dataset_success(self, loader, tmp_path, sample_csv_data):
        """Test successful dataset loading."""
        # Create multiple CSV files
        for i in range(3):
            csv_path = tmp_path / f"file{i}.csv"
            df = pd.DataFrame(sample_csv_data)
            df.to_csv(csv_path, index=False)

        # Load dataset
        combined_df = loader.load_dataset(tmp_path)

        assert len(combined_df) == 9  # 3 files × 3 rows each
        assert len(combined_df.columns) == 10  # 9 original columns + participant_id

    def test_load_dataset_no_csv_files(self, loader, tmp_path):
        """Test loading dataset with no CSV files."""
        with pytest.raises(ValueError, match="No CSV files found"):
            loader.load_dataset(tmp_path)

    def test_load_dataset_directory_not_found(self, loader, tmp_path):
        """Test loading dataset from non-existent directory."""
        non_existent_dir = tmp_path / "nonexistent"

        with pytest.raises(FileNotFoundError):
            loader.load_dataset(non_existent_dir)

    def test_validate_dataset_schema_valid(self, loader, sample_csv_data):
        """Test schema validation with valid dataset."""
        df = pd.DataFrame(sample_csv_data)
        df = loader._parse_datetime(df)

        results = loader.validate_dataset_schema(df)

        assert results["is_valid"] is True
        assert len(results["missing_columns"]) == 0
        assert results["summary"]["total_rows"] == 3

    def test_validate_dataset_schema_missing_columns(self, loader):
        """Test schema validation with missing columns."""
        df = pd.DataFrame(
            {
                "EventDateTime": ["2023-01-01 10:00:00"],
                "CGM": [120.5],
                # Missing other required columns
            }
        )

        results = loader.validate_dataset_schema(df)

        assert results["is_valid"] is False
        assert len(results["missing_columns"]) > 0
        assert "Basal" in results["missing_columns"]

    def test_validate_dataset_schema_extra_columns(self, loader, sample_csv_data):
        """Test schema validation with extra columns."""
        sample_csv_data["ExtraColumn"] = [1, 2, 3]
        df = pd.DataFrame(sample_csv_data)

        results = loader.validate_dataset_schema(df)

        assert "ExtraColumn" in results["extra_columns"]


class TestDataAcquisitionPipeline:
    """Test cases for DataAcquisitionPipeline class."""

    @pytest.fixture
    def config(self):
        """Test configuration."""
        return {
            "dataset_url": "https://example.com/test-dataset.zip",
            "raw_data_path": "test_data/raw",
            "max_retries": 1,
            "retry_delay": 0.1,
        }

    @pytest.fixture
    def pipeline(self, config, tmp_path):
        """Create DataAcquisitionPipeline instance."""
        config["raw_data_path"] = str(tmp_path / "raw")
        return DataAcquisitionPipeline(config)

    @pytest.fixture
    def sample_dataset_zip(self, tmp_path):
        """Create sample dataset ZIP file."""
        zip_path = tmp_path / "dataset.zip"

        # Create sample CSV content
        csv_content = """EventDateTime,DeviceMode,BolusType,Basal,CorrectionDelivered,TotalBolusInsulinDelivered,FoodDelivered,CarbSize,CGM
2023-01-01 10:00:00,Manual,Normal,1.2,0.0,3.5,0.0,45.0,120.5
2023-01-01 10:05:00,Auto,Extended,1.3,2.5,0.0,1.0,0.0,145.2
2023-01-01 10:10:00,Manual,Normal,1.1,0.0,4.2,0.0,60.0,110.8"""

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("data.csv", csv_content)

        return zip_path

    def test_init(self, pipeline):
        """Test DataAcquisitionPipeline initialization."""
        assert isinstance(pipeline.downloader, DataDownloader)
        assert isinstance(pipeline.extractor, DataExtractor)
        assert isinstance(pipeline.loader, DataLoader)

    @patch.object(DataDownloader, "download_dataset")
    @patch.object(DataDownloader, "verify_download_integrity")
    @patch.object(DataExtractor, "extract_dataset")
    @patch.object(DataLoader, "load_dataset")
    @patch.object(DataLoader, "validate_dataset_schema")
    def test_run_pipeline_success(
        self,
        mock_validate,
        mock_load,
        mock_extract,
        mock_verify,
        mock_download,
        pipeline,
        tmp_path,
    ):
        """Test successful pipeline execution."""
        # Setup mocks
        zip_path = tmp_path / "dataset.zip"
        extract_dir = tmp_path / "extracted"
        sample_df = pd.DataFrame({"col1": [1, 2, 3]})

        mock_download.return_value = zip_path
        mock_verify.return_value = True
        mock_extract.return_value = extract_dir
        mock_load.return_value = sample_df
        mock_validate.return_value = {"is_valid": True}

        # Run pipeline
        result_df = pipeline.run_pipeline()

        # Verify calls
        mock_download.assert_called_once_with(force_download=False)
        mock_verify.assert_called_once_with(zip_path)
        mock_extract.assert_called_once_with(zip_path)
        mock_load.assert_called_once_with(extract_dir)
        mock_validate.assert_called_once_with(sample_df)

        # Verify result
        assert result_df.equals(sample_df)

    @patch.object(DataDownloader, "download_dataset")
    @patch.object(DataDownloader, "verify_download_integrity")
    def test_run_pipeline_integrity_check_fails(
        self, mock_verify, mock_download, pipeline, tmp_path
    ):
        """Test pipeline failure when integrity check fails."""
        zip_path = tmp_path / "dataset.zip"
        mock_download.return_value = zip_path
        mock_verify.return_value = False

        with pytest.raises(ValueError, match="integrity check"):
            pipeline.run_pipeline()

    @patch.object(DataDownloader, "download_dataset")
    def test_run_pipeline_download_fails(self, mock_download, pipeline):
        """Test pipeline failure when download fails."""
        mock_download.side_effect = Exception("Download failed")

        with pytest.raises(Exception, match="Download failed"):
            pipeline.run_pipeline()

    def test_get_dataset_info(self, pipeline):
        """Test getting dataset information."""
        info = pipeline.get_dataset_info()

        assert "dataset_url" in info
        assert "raw_data_path" in info
        assert "expected_columns" in info
        assert "dtype_mapping" in info
        assert "local_file_exists" in info

        assert info["dataset_url"] == "https://example.com/test-dataset.zip"
        assert len(info["expected_columns"]) == 9

    def test_get_dataset_info_with_local_file(self, pipeline, tmp_path):
        """Test getting dataset info when local file exists."""
        # Create local ZIP file
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(exist_ok=True)
        local_zip = raw_dir / "test-dataset.zip"
        local_zip.write_bytes(b"dummy content")

        info = pipeline.get_dataset_info()

        assert info["local_file_exists"] is True
        assert info["local_file_size"] == len(b"dummy content")
        assert "local_file_path" in info

    @patch.object(DataLoader, "load_dataset")
    @patch.object(DataLoader, "validate_dataset_schema")
    @patch.object(DataAcquisitionPipeline, "_is_dataset_available")
    def test_run_pipeline_no_op_when_data_available(
        self, mock_is_available, mock_validate, mock_load, pipeline, tmp_path
    ):
        """Test that pipeline performs no-op when data is already available."""
        # Mock that dataset is available
        mock_is_available.return_value = True

        # Create mock data
        mock_df = pd.DataFrame(
            {
                "EventDateTime": ["2023-01-01 00:00:00"],
                "DeviceMode": ["Manual"],
                "BolusType": ["Normal"],
                "Basal": [1.0],
                "CorrectionDelivered": [0.0],
                "TotalBolusInsulinDelivered": [0.0],
                "FoodDelivered": [0.0],
                "CarbSize": [0.0],
                "CGM": [120.0],
            }
        )

        mock_load.return_value = mock_df
        mock_validate.return_value = {"is_valid": True}

        # Run pipeline with force_download=False (default)
        result = pipeline.run_pipeline(force_download=False)

        # Verify that _is_dataset_available was called
        mock_is_available.assert_called_once()

        # Verify that load_dataset was called (for existing data)
        mock_load.assert_called_once()

        # Verify that validation was called
        mock_validate.assert_called_once_with(mock_df)

        # Verify the result
        assert result.equals(mock_df)

    @patch.object(DataDownloader, "download_dataset")
    @patch.object(DataDownloader, "verify_download_integrity")
    @patch.object(DataExtractor, "extract_dataset")
    @patch.object(DataLoader, "load_dataset")
    @patch.object(DataLoader, "validate_dataset_schema")
    @patch.object(DataAcquisitionPipeline, "_is_dataset_available")
    def test_run_pipeline_force_download_ignores_existing_data(
        self,
        mock_is_available,
        mock_validate,
        mock_load,
        mock_extract,
        mock_verify,
        mock_download,
        pipeline,
        tmp_path,
    ):
        """Test that force_download=True ignores existing data and downloads again."""
        # Mock that dataset is available (but force_download=True should ignore this)
        mock_is_available.return_value = True

        # Create mock data
        mock_df = pd.DataFrame(
            {
                "EventDateTime": ["2023-01-01 00:00:00"],
                "DeviceMode": ["Manual"],
                "BolusType": ["Normal"],
                "Basal": [1.0],
                "CorrectionDelivered": [0.0],
                "TotalBolusInsulinDelivered": [0.0],
                "FoodDelivered": [0.0],
                "CarbSize": [0.0],
                "CGM": [120.0],
            }
        )

        mock_download.return_value = tmp_path / "test.zip"
        mock_verify.return_value = True
        mock_extract.return_value = tmp_path / "extracted"
        mock_load.return_value = mock_df
        mock_validate.return_value = {"is_valid": True}

        # Run pipeline with force_download=True
        result = pipeline.run_pipeline(force_download=True)

        # Verify that _is_dataset_available was NOT called (force_download=True bypasses it)
        mock_is_available.assert_not_called()

        # Verify that download was called despite existing data
        mock_download.assert_called_once_with(force_download=True)

        # Verify that other steps were also called
        mock_verify.assert_called_once()
        mock_extract.assert_called_once()
        mock_load.assert_called_once()
        mock_validate.assert_called_once()

        # Verify the result
        assert result.equals(mock_df)


if __name__ == "__main__":
    pytest.main([__file__])

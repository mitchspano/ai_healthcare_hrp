"""for downloading, extracting, and loading the AZT1D dataset."""

import hashlib
import logging
import time
import zipfile
from pathlib import Path
from typing import Dict, Any, Optional, Union
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class DataDownloader:
    """Handles dataset retrieval from S3 with retry logic and integrity checks."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DataDownloader.

        Args:
            config: Configuration dictionary containing download settings
        """
        self.config = config
        self.dataset_url = config.get("dataset_url")
        self.download_path = Path(config.get("raw_data_path", "data/raw"))
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1.0)
        self.chunk_size = config.get("chunk_size", 8192)
        self.timeout = config.get("timeout", 300)  # 5 minutes

        # Create download directory if it doesn't exist
        self.download_path.mkdir(parents=True, exist_ok=True)

        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=self.max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _calculate_file_hash(self, file_path: Path, algorithm: str = "md5") -> str:
        """
        Calculate hash of a file.

        Args:
            file_path: Path to the file
            algorithm: Hash algorithm to use

        Returns:
            Hexadecimal hash string
        """
        hash_func = hashlib.new(algorithm)

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(self.chunk_size), b""):
                hash_func.update(chunk)

        return hash_func.hexdigest()

    def _get_remote_file_info(self, url: str) -> Dict[str, Any]:
        """
        Get information about the remote file.

        Args:
            url: URL of the remote file

        Returns:
            Dictionary containing file information
        """
        try:
            response = self.session.head(url, timeout=self.timeout)
            response.raise_for_status()

            return {
                "content_length": int(response.headers.get("content-length", 0)),
                "last_modified": response.headers.get("last-modified"),
                "etag": response.headers.get("etag", "").strip('"'),
            }
        except Exception as e:
            logger.warning(f"Could not get remote file info: {e}")
            return {}

    def _should_download(self, local_path: Path, remote_info: Dict[str, Any]) -> bool:
        """
        Determine if file should be downloaded based on local and remote information.

        Args:
            local_path: Path to local file
            remote_info: Remote file information

        Returns:
            True if file should be downloaded
        """
        if not local_path.exists():
            return True

        # Check file size if available
        if remote_info.get("content_length"):
            local_size = local_path.stat().st_size
            remote_size = remote_info["content_length"]
            if local_size != remote_size:
                logger.info(
                    f"File size mismatch: local={local_size}, remote={remote_size}"
                )
                return True

        return False

    def download_dataset(self, force_download: bool = False) -> Path:
        """
        Download the dataset from the configured URL.

        Args:
            force_download: Force download even if file exists

        Returns:
            Path to the downloaded file

        Raises:
            ValueError: If dataset URL is not configured
            requests.RequestException: If download fails after retries
        """
        if not self.dataset_url:
            raise ValueError("Dataset URL not configured")

        # Determine local file path
        filename = Path(self.dataset_url).name
        if not filename or filename == "/":
            filename = "dataset.zip"

        local_path = self.download_path / filename

        # Check if we need to download
        if not force_download:
            remote_info = self._get_remote_file_info(self.dataset_url)
            if not self._should_download(local_path, remote_info):
                logger.info(f"Dataset already exists and is up to date: {local_path}")
                return local_path

        logger.info(f"Downloading dataset from {self.dataset_url}")

        # Download with retry logic
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(
                    self.dataset_url, stream=True, timeout=self.timeout
                )
                response.raise_for_status()

                # Get total file size for progress tracking
                total_size = int(response.headers.get("content-length", 0))
                downloaded_size = 0

                # Download file in chunks
                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)

                            # Log progress every 10MB
                            if (
                                downloaded_size % (10 * 1024 * 1024) == 0
                                and total_size > 0
                            ):
                                progress = (downloaded_size / total_size) * 100
                                logger.info(f"Download progress: {progress:.1f}%")

                logger.info(f"Dataset downloaded successfully: {local_path}")
                logger.info(f"File size: {local_path.stat().st_size:,} bytes")

                return local_path

            except Exception as e:
                if attempt < self.max_retries:
                    wait_time = self.retry_delay * (2**attempt)
                    logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"Download failed after {self.max_retries + 1} attempts: {e}"
                    )
                    # Clean up partial download
                    if local_path.exists():
                        local_path.unlink()
                    raise

    def verify_download_integrity(
        self, file_path: Path, expected_hash: Optional[str] = None
    ) -> bool:
        """
        Verify the integrity of a downloaded file.

        Args:
            file_path: Path to the downloaded file
            expected_hash: Expected hash value (if known)

        Returns:
            True if file integrity is verified
        """
        if not file_path.exists():
            logger.error(f"File does not exist: {file_path}")
            return False

        # Basic checks
        file_size = file_path.stat().st_size
        if file_size == 0:
            logger.error(f"Downloaded file is empty: {file_path}")
            return False

        # Hash verification if expected hash is provided
        if expected_hash:
            actual_hash = self._calculate_file_hash(file_path)
            if actual_hash != expected_hash:
                logger.error(
                    f"Hash mismatch: expected={expected_hash}, actual={actual_hash}"
                )
                return False
            logger.info("File hash verification passed")

        logger.info(f"File integrity verified: {file_path} ({file_size:,} bytes)")
        return True


class DataExtractor:
    """Handles ZIP file extraction and directory management."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DataExtractor.

        Args:
            config: Configuration dictionary containing extraction settings
        """
        self.config = config
        self.extract_path = Path(config.get("raw_data_path", "data/raw"))
        self.extract_path.mkdir(parents=True, exist_ok=True)

    def extract_dataset(
        self, zip_path: Path, extract_to: Optional[Path] = None
    ) -> Path:
        """
        Extract ZIP file contents to the specified directory.

        Args:
            zip_path: Path to the ZIP file
            extract_to: Directory to extract to (defaults to configured path)

        Returns:
            Path to the extraction directory

        Raises:
            FileNotFoundError: If ZIP file doesn't exist
            zipfile.BadZipFile: If ZIP file is corrupted
        """
        if not zip_path.exists():
            raise FileNotFoundError(f"ZIP file not found: {zip_path}")

        extract_dir = extract_to or self.extract_path
        extract_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Extracting {zip_path} to {extract_dir}")

        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                # Get list of files in the archive
                file_list = zip_ref.namelist()
                logger.info(f"Archive contains {len(file_list)} files")

                # Extract all files
                zip_ref.extractall(extract_dir)

                # Log extracted files
                for file_name in file_list:
                    extracted_path = extract_dir / file_name
                    if extracted_path.exists():
                        size = extracted_path.stat().st_size
                        logger.debug(f"Extracted: {file_name} ({size:,} bytes)")

            logger.info(f"Extraction completed successfully to {extract_dir}")
            return extract_dir

        except zipfile.BadZipFile as e:
            logger.error(f"Corrupted ZIP file: {zip_path}")
            raise
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            raise

    def list_archive_contents(self, zip_path: Path) -> list:
        """
        List contents of a ZIP archive without extracting.

        Args:
            zip_path: Path to the ZIP file

        Returns:
            List of file names in the archive
        """
        if not zip_path.exists():
            raise FileNotFoundError(f"ZIP file not found: {zip_path}")

        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                return zip_ref.namelist()
        except zipfile.BadZipFile:
            logger.error(f"Corrupted ZIP file: {zip_path}")
            raise

    def validate_archive(self, zip_path: Path) -> bool:
        """
        Validate ZIP archive integrity.

        Args:
            zip_path: Path to the ZIP file

        Returns:
            True if archive is valid
        """
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                # Test the archive
                bad_file = zip_ref.testzip()
                if bad_file:
                    logger.error(f"Corrupted file in archive: {bad_file}")
                    return False

                logger.info("ZIP archive validation passed")
                return True

        except zipfile.BadZipFile:
            logger.error(f"Invalid ZIP file: {zip_path}")
            return False
        except Exception as e:
            logger.error(f"Archive validation failed: {e}")
            return False


class DataLoader:
    """Loads CSV files into pandas DataFrames with proper data type inference."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DataLoader.

        Args:
            config: Configuration dictionary containing loading settings
        """
        self.config = config
        self.data_path = Path(config.get("raw_data_path", "data/raw"))

        # Expected data schema for the AZT1D dataset
        self.expected_columns = [
            "EventDateTime",
            "DeviceMode",
            "BolusType",
            "Basal",
            "CorrectionDelivered",
            "TotalBolusInsulinDelivered",
            "FoodDelivered",
            "CarbSize",
            "CGM",
        ]

        # Data type mappings
        self.dtype_mapping = {
            "DeviceMode": "category",
            "BolusType": "category",
            "Basal": "float64",
            "CorrectionDelivered": "float64",
            "TotalBolusInsulinDelivered": "float64",
            "FoodDelivered": "float64",
            "CarbSize": "float64",
            "CGM": "float64",
        }

    def _infer_csv_files(self, data_dir: Path) -> list:
        """
        Find CSV files in the data directory.

        Args:
            data_dir: Directory to search for CSV files

        Returns:
            List of CSV file paths
        """
        csv_files = []

        # Search for CSV files recursively
        for csv_file in data_dir.rglob("*.csv"):
            csv_files.append(csv_file)

        logger.info(f"Found {len(csv_files)} CSV files in {data_dir}")
        return csv_files

    def _parse_datetime(
        self, df: pd.DataFrame, datetime_column: str = "EventDateTime"
    ) -> pd.DataFrame:
        """
        Parse datetime column with proper handling of various formats.

        Args:
            df: DataFrame to process
            datetime_column: Name of the datetime column

        Returns:
            DataFrame with parsed datetime column
        """
        if datetime_column not in df.columns:
            logger.warning(f"Datetime column '{datetime_column}' not found")
            return df

        try:
            # Try to parse datetime with automatic format detection
            df[datetime_column] = pd.to_datetime(df[datetime_column])
            logger.info(f"Successfully parsed {datetime_column} column")
        except Exception as e:
            logger.warning(f"Could not parse datetime column: {e}")

        return df

    def load_csv_file(self, csv_path: Path) -> pd.DataFrame:
        """
        Load a single CSV file with proper data type inference.

        Args:
            csv_path: Path to the CSV file

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            pd.errors.EmptyDataError: If CSV file is empty
        """
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        logger.info(f"Loading CSV file: {csv_path}")

        try:
            # Load CSV with initial data type inference
            df = pd.read_csv(
                csv_path,
                dtype=self.dtype_mapping,
                low_memory=False,
                na_values=["", "NA", "N/A", "null", "NULL", "nan", "NaN"],
            )

            # Parse datetime column
            df = self._parse_datetime(df)

            # Log basic information
            logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            logger.info(f"Columns: {list(df.columns)}")
            logger.info(
                f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
            )

            return df

        except pd.errors.EmptyDataError:
            logger.error(f"CSV file is empty: {csv_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load CSV file {csv_path}: {e}")
            raise

    def load_dataset(self, data_dir: Optional[Path] = None) -> pd.DataFrame:
        """
        Load the complete dataset from CSV files in the specified directory.

        Args:
            data_dir: Directory containing CSV files (defaults to configured path)

        Returns:
            Combined DataFrame from all CSV files

        Raises:
            ValueError: If no CSV files are found
        """
        search_dir = data_dir or self.data_path

        if not search_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {search_dir}")

        # Find CSV files
        csv_files = self._infer_csv_files(search_dir)

        if not csv_files:
            raise ValueError(f"No CSV files found in {search_dir}")

        # Load and combine all CSV files
        dataframes = []
        total_rows = 0

        for csv_file in csv_files:
            try:
                df = self.load_csv_file(csv_file)
                dataframes.append(df)
                total_rows += len(df)
                logger.info(f"Loaded {csv_file.name}: {len(df)} rows")
            except Exception as e:
                logger.error(f"Failed to load {csv_file}: {e}")
                continue

        if not dataframes:
            raise ValueError("No CSV files could be loaded successfully")

        # Combine all dataframes
        logger.info("Combining all CSV files...")
        combined_df = pd.concat(dataframes, ignore_index=True)

        logger.info(
            f"Combined dataset: {len(combined_df)} rows, {len(combined_df.columns)} columns"
        )
        logger.info(
            f"Total memory usage: {combined_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        )

        return combined_df

    def validate_dataset_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the dataset schema against expected structure.

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            "is_valid": True,
            "missing_columns": [],
            "extra_columns": [],
            "data_type_issues": [],
            "summary": {},
        }

        # Check for missing columns
        missing_cols = set(self.expected_columns) - set(df.columns)
        if missing_cols:
            validation_results["missing_columns"] = list(missing_cols)
            validation_results["is_valid"] = False

        # Check for extra columns
        extra_cols = set(df.columns) - set(self.expected_columns)
        if extra_cols:
            validation_results["extra_columns"] = list(extra_cols)

        # Check data types
        for col, expected_dtype in self.dtype_mapping.items():
            if col in df.columns:
                actual_dtype = str(df[col].dtype)
                if (
                    expected_dtype not in actual_dtype
                    and actual_dtype != expected_dtype
                ):
                    validation_results["data_type_issues"].append(
                        {
                            "column": col,
                            "expected": expected_dtype,
                            "actual": actual_dtype,
                        }
                    )

        # Generate summary
        validation_results["summary"] = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.astype(str).to_dict(),
        }

        return validation_results


class DataAcquisitionPipeline:
    """Orchestrates the complete data acquisition process."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DataAcquisitionPipeline.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.downloader = DataDownloader(config)
        self.extractor = DataExtractor(config)
        self.loader = DataLoader(config)

    def run_pipeline(self, force_download: bool = False) -> pd.DataFrame:
        """
        Run the complete data acquisition pipeline.

        Args:
            force_download: Force re-download of dataset

        Returns:
            Loaded and validated DataFrame

        Raises:
            Exception: If any step in the pipeline fails
        """
        logger.info("Starting data acquisition pipeline")

        try:
            # Step 1: Download dataset
            zip_path = self.downloader.download_dataset(force_download=force_download)

            # Step 2: Verify download integrity
            if not self.downloader.verify_download_integrity(zip_path):
                raise ValueError("Downloaded file failed integrity check")

            # Step 3: Extract dataset
            extract_dir = self.extractor.extract_dataset(zip_path)

            # Step 4: Load dataset
            df = self.loader.load_dataset(extract_dir)

            # Step 5: Validate schema
            validation_results = self.loader.validate_dataset_schema(df)

            if not validation_results["is_valid"]:
                logger.warning("Dataset schema validation failed:")
                logger.warning(
                    f"Missing columns: {validation_results['missing_columns']}"
                )
                logger.warning(
                    f"Data type issues: {validation_results['data_type_issues']}"
                )
            else:
                logger.info("Dataset schema validation passed")

            logger.info("Data acquisition pipeline completed successfully")
            return df

        except Exception as e:
            logger.error(f"Data acquisition pipeline failed: {e}")
            raise

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the dataset without loading it.

        Returns:
            Dictionary containing dataset information
        """
        info = {
            "dataset_url": self.config.get("dataset_url"),
            "raw_data_path": str(self.config.get("raw_data_path")),
            "expected_columns": self.loader.expected_columns,
            "dtype_mapping": self.loader.dtype_mapping,
        }

        # Check if dataset exists locally
        zip_filename = Path(self.config.get("dataset_url", "")).name or "dataset.zip"
        local_zip_path = (
            Path(self.config.get("raw_data_path", "data/raw")) / zip_filename
        )

        if local_zip_path.exists():
            info["local_file_exists"] = True
            info["local_file_size"] = local_zip_path.stat().st_size
            info["local_file_path"] = str(local_zip_path)
        else:
            info["local_file_exists"] = False

        return info

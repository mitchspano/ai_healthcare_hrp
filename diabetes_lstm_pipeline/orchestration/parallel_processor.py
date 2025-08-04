"""
Parallel processing options for computationally intensive stages.
"""

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Callable, Optional, Union
import time
from functools import partial
import psutil


class ParallelProcessor:
    """Manages parallel processing for computationally intensive pipeline stages."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize parallel processor.

        Args:
            config: Configuration for parallel processing settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Processing settings
        self.max_workers = self.config.get("max_workers", self._get_optimal_workers())
        self.use_processes = self.config.get("use_processes", True)
        self.chunk_size = self.config.get("chunk_size", 1000)
        self.memory_limit_gb = self.config.get(
            "memory_limit_gb", self._get_memory_limit()
        )

        self.logger.info(
            f"Parallel processor initialized with {self.max_workers} workers"
        )

    def _get_optimal_workers(self) -> int:
        """
        Determine optimal number of workers based on system resources.

        Returns:
            Optimal number of workers
        """
        cpu_count = mp.cpu_count()

        # For CPU-intensive tasks, use CPU count
        # For I/O-intensive tasks, can use more workers
        optimal_workers = min(cpu_count, 8)  # Cap at 8 to avoid resource exhaustion

        return optimal_workers

    def _get_memory_limit(self) -> float:
        """
        Get memory limit based on available system memory.

        Returns:
            Memory limit in GB
        """
        available_memory_gb = psutil.virtual_memory().available / (1024**3)

        # Use 80% of available memory as limit
        return available_memory_gb * 0.8

    def process_in_parallel(
        self,
        func: Callable,
        data_chunks: List[Any],
        stage_name: str = "parallel_processing",
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[Any]:
        """
        Process data chunks in parallel.

        Args:
            func: Function to apply to each chunk
            data_chunks: List of data chunks to process
            stage_name: Name of the stage for logging
            progress_callback: Optional callback for progress updates

        Returns:
            List of results from processing each chunk
        """
        if not data_chunks:
            return []

        self.logger.info(
            f"Starting parallel processing for {stage_name} with {len(data_chunks)} chunks"
        )

        start_time = time.time()
        results = []

        # Choose executor type based on configuration
        executor_class = (
            ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        )

        try:
            with executor_class(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_chunk = {
                    executor.submit(func, chunk): i
                    for i, chunk in enumerate(data_chunks)
                }

                # Collect results as they complete
                completed_count = 0
                for future in as_completed(future_to_chunk):
                    chunk_index = future_to_chunk[future]

                    try:
                        result = future.result()
                        results.append((chunk_index, result))
                        completed_count += 1

                        # Update progress
                        if progress_callback:
                            progress = (completed_count / len(data_chunks)) * 100
                            progress_callback(progress)

                        self.logger.debug(
                            f"Completed chunk {chunk_index + 1}/{len(data_chunks)}"
                        )

                    except Exception as e:
                        self.logger.error(f"Error processing chunk {chunk_index}: {e}")
                        results.append((chunk_index, None))
                        completed_count += 1

        except Exception as e:
            self.logger.error(f"Error in parallel processing: {e}")
            raise

        # Sort results by original chunk order
        results.sort(key=lambda x: x[0])
        final_results = [result for _, result in results]

        duration = time.time() - start_time
        self.logger.info(f"Parallel processing completed in {duration:.2f} seconds")

        return final_results

    def process_dataframe_parallel(
        self,
        df,
        func: Callable,
        stage_name: str = "dataframe_processing",
        progress_callback: Optional[Callable[[float], None]] = None,
    ):
        """
        Process a pandas DataFrame in parallel chunks.

        Args:
            df: Pandas DataFrame to process
            func: Function to apply to each chunk
            stage_name: Name of the stage for logging
            progress_callback: Optional callback for progress updates

        Returns:
            Concatenated result DataFrame
        """
        import pandas as pd

        if df.empty:
            return df

        # Calculate chunk size based on memory constraints
        memory_per_row = df.memory_usage(deep=True).sum() / len(df)
        max_rows_per_chunk = int(
            (self.memory_limit_gb * 1024**3) / (memory_per_row * self.max_workers)
        )
        chunk_size = min(self.chunk_size, max_rows_per_chunk, len(df))

        self.logger.info(
            f"Processing DataFrame with {len(df)} rows in chunks of {chunk_size}"
        )

        # Split DataFrame into chunks
        chunks = [df.iloc[i : i + chunk_size] for i in range(0, len(df), chunk_size)]

        # Process chunks in parallel
        results = self.process_in_parallel(func, chunks, stage_name, progress_callback)

        # Filter out None results and concatenate
        valid_results = [result for result in results if result is not None]

        if valid_results:
            return pd.concat(valid_results, ignore_index=True)
        else:
            return pd.DataFrame()

    def process_files_parallel(
        self,
        file_paths: List[str],
        func: Callable,
        stage_name: str = "file_processing",
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[Any]:
        """
        Process multiple files in parallel.

        Args:
            file_paths: List of file paths to process
            func: Function to apply to each file
            stage_name: Name of the stage for logging
            progress_callback: Optional callback for progress updates

        Returns:
            List of results from processing each file
        """
        self.logger.info(f"Processing {len(file_paths)} files in parallel")

        return self.process_in_parallel(func, file_paths, stage_name, progress_callback)

    def map_reduce_parallel(
        self,
        data: List[Any],
        map_func: Callable,
        reduce_func: Callable,
        stage_name: str = "map_reduce",
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Any:
        """
        Perform parallel map-reduce operation.

        Args:
            data: Input data to process
            map_func: Function to apply to each data item (map phase)
            reduce_func: Function to combine results (reduce phase)
            stage_name: Name of the stage for logging
            progress_callback: Optional callback for progress updates

        Returns:
            Final reduced result
        """
        self.logger.info(f"Starting map-reduce operation for {stage_name}")

        # Map phase - process data in parallel
        map_results = self.process_in_parallel(
            map_func, data, f"{stage_name}_map", progress_callback
        )

        # Reduce phase - combine results
        self.logger.info(f"Starting reduce phase for {stage_name}")

        # Filter out None results
        valid_results = [result for result in map_results if result is not None]

        if not valid_results:
            return None

        # Apply reduce function
        final_result = valid_results[0]
        for result in valid_results[1:]:
            final_result = reduce_func(final_result, result)

        self.logger.info(f"Map-reduce operation completed for {stage_name}")
        return final_result

    def get_system_resources(self) -> Dict[str, Any]:
        """
        Get current system resource usage.

        Returns:
            Dictionary with system resource information
        """
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)

        return {
            "cpu_count": mp.cpu_count(),
            "cpu_percent": cpu_percent,
            "memory_total_gb": memory.total / (1024**3),
            "memory_available_gb": memory.available / (1024**3),
            "memory_percent": memory.percent,
            "max_workers": self.max_workers,
            "memory_limit_gb": self.memory_limit_gb,
        }

    def check_resource_availability(self) -> bool:
        """
        Check if system has sufficient resources for parallel processing.

        Returns:
            True if resources are available, False otherwise
        """
        resources = self.get_system_resources()

        # Check memory availability
        if resources["memory_percent"] > 85:
            self.logger.warning(
                "High memory usage detected, parallel processing may be limited"
            )
            return False

        # Check CPU availability
        if resources["cpu_percent"] > 90:
            self.logger.warning(
                "High CPU usage detected, parallel processing may be limited"
            )
            return False

        return True

    def adjust_workers_based_on_resources(self) -> None:
        """Dynamically adjust number of workers based on current resource usage."""
        resources = self.get_system_resources()

        # Reduce workers if memory usage is high
        if resources["memory_percent"] > 80:
            new_workers = max(1, self.max_workers // 2)
            self.logger.info(
                f"Reducing workers from {self.max_workers} to {new_workers} due to high memory usage"
            )
            self.max_workers = new_workers

        # Reduce workers if CPU usage is high
        elif resources["cpu_percent"] > 85:
            new_workers = max(1, self.max_workers // 2)
            self.logger.info(
                f"Reducing workers from {self.max_workers} to {new_workers} due to high CPU usage"
            )
            self.max_workers = new_workers

    def create_processing_strategy(
        self, data_size: int, complexity: str = "medium"
    ) -> Dict[str, Any]:
        """
        Create an optimal processing strategy based on data size and complexity.

        Args:
            data_size: Size of data to process
            complexity: Complexity level ("low", "medium", "high")

        Returns:
            Dictionary with processing strategy recommendations
        """
        resources = self.get_system_resources()

        # Determine optimal chunk size based on data size and available memory
        if complexity == "low":
            chunk_size = min(10000, data_size // self.max_workers)
        elif complexity == "medium":
            chunk_size = min(5000, data_size // self.max_workers)
        else:  # high complexity
            chunk_size = min(1000, data_size // self.max_workers)

        # Ensure minimum chunk size
        chunk_size = max(100, chunk_size)

        # Determine if we should use processes or threads
        use_processes = complexity in ["medium", "high"] and data_size > 1000

        strategy = {
            "chunk_size": chunk_size,
            "use_processes": use_processes,
            "max_workers": self.max_workers,
            "estimated_chunks": (data_size + chunk_size - 1) // chunk_size,
            "memory_per_worker_gb": resources["memory_available_gb"] / self.max_workers,
            "recommended": True,
        }

        # Check if strategy is feasible
        if strategy["memory_per_worker_gb"] < 0.5:  # Less than 500MB per worker
            strategy["recommended"] = False
            strategy["reason"] = "Insufficient memory for parallel processing"

        return strategy

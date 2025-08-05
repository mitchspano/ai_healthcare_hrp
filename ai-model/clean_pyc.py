#!/usr/bin/env python3
"""
Script to remove all .pyc files and __pycache__ directories from the project.
This helps clean up compiled Python bytecode files that are automatically generated.
"""

import os
import shutil
import sys
from pathlib import Path


def clean_pyc_files():
    """
    Remove all .pyc files and __pycache__ directories from the current directory and subdirectories.
    """
    current_dir = Path.cwd()
    removed_files = 0
    removed_dirs = 0

    print(f"Cleaning .pyc files and __pycache__ directories in: {current_dir}")
    print("-" * 60)

    # Find and remove .pyc files
    for pyc_file in current_dir.rglob("*.pyc"):
        try:
            os.remove(pyc_file)
            print(f"Removed file: {pyc_file}")
            removed_files += 1
        except OSError as e:
            print(f"Error removing {pyc_file}: {e}")

    # Find and remove __pycache__ directories
    for pycache_dir in current_dir.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache_dir)
            print(f"Removed directory: {pycache_dir}")
            removed_dirs += 1
        except OSError as e:
            print(f"Error removing {pycache_dir}: {e}")

    print("-" * 60)
    print(f"Cleanup complete!")
    print(f"Removed {removed_files} .pyc files")
    print(f"Removed {removed_dirs} __pycache__ directories")

    if removed_files == 0 and removed_dirs == 0:
        print("No .pyc files or __pycache__ directories found to clean.")

    return removed_files + removed_dirs


def main():
    """Main function to run the cleanup."""
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        print("Usage: python clean_pyc.py")
        print(
            "Removes all .pyc files and __pycache__ directories from the current project."
        )
        return

    try:
        total_removed = clean_pyc_files()
        if total_removed > 0:
            print(f"\nTotal items removed: {total_removed}")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\nCleanup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during cleanup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

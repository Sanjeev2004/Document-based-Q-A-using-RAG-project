"""
Helper utility functions
"""

import os
from pathlib import Path


def ensure_directory_exists(directory_path: str) -> None:
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        directory_path: Path to the directory
    """
    os.makedirs(directory_path, exist_ok=True)


def get_file_extension(file_path: str) -> str:
    """
    Get file extension from file path.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File extension (e.g., '.pdf')
    """
    return Path(file_path).suffix.lower()


def is_valid_pdf(file_path: str) -> bool:
    """
    Check if file is a valid PDF.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file is a PDF, False otherwise
    """
    return get_file_extension(file_path) == '.pdf'

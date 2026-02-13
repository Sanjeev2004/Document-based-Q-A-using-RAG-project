"""
Document loaders for various file formats
Currently supports PDF files
"""

from typing import List
import PyPDF2
from pathlib import Path


def load_pdf(file_path: str) -> str:
    """
    Load text content from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text content as a string
    """
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        raise Exception(f"Error loading PDF: {str(e)}")
    
    return text


def load_document(file_path: str) -> str:
    """
    Universal document loader that routes to appropriate loader based on file extension.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        Extracted text content as a string
    """
    path = Path(file_path)
    extension = path.suffix.lower()
    
    if extension == '.pdf':
        return load_pdf(file_path)
    else:
        raise ValueError(f"Unsupported file format: {extension}. Currently only PDF is supported.")

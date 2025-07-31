from PyPDF2 import PdfReader
import io

def extract_text_from_pdf(content: bytes) -> str:
    """
    Extract text from PDF content (bytes)
    
    Args:
        content (bytes): PDF file content as bytes
        
    Returns:
        str: Extracted text from the PDF
    """
    try:
        # Convert bytes to a file-like object
        file_like_object = io.BytesIO(content)
        
        # Create PdfReader from the file-like object
        reader = PdfReader(file_like_object)
        
        text = ""
        # Extract text from all pages
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        return text.strip()
    
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    """
    Split text into overlapping chunks
    
    Args:
        text (str): Text to be chunked
        chunk_size (int): Size of each chunk
        overlap (int): Overlap between chunks
        
    Returns:
        list: List of text chunks
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # If this isn't the last chunk and we're in the middle of a word,
        # try to end at a word boundary
        if end < len(text) and not text[end].isspace():
            # Look backwards for a space
            last_space = chunk.rfind(' ')
            if last_space > chunk_size * 0.8:  # Only if we don't cut too much
                chunk = chunk[:last_space]
                end = start + last_space
        
        chunks.append(chunk.strip())
        
        # Move start position considering overlap
        start = end - overlap
        
        # Avoid infinite loop if overlap is too large
        if start <= 0:
            start = end
    
    return [chunk for chunk in chunks if chunk]  # Remove empty chunks
# pdf_utils.py
from PyPDF2 import PdfReader
import pdfplumber
import io
import re
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(content: bytes) -> str:
    """
    Enhanced PDF text extraction using multiple methods for better accuracy
    
    Args:
        content (bytes): PDF file content as bytes
        
    Returns:
        str: Extracted text from the PDF
    """
    try:
        text = ""
        extraction_method = "unknown"
        
        # Method 1: Try pdfplumber first (better for complex layouts, tables, etc.)
        try:
            logger.info("Attempting text extraction with pdfplumber...")
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                pdfplumber_text = ""
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        pdfplumber_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                
                if pdfplumber_text.strip():
                    text = pdfplumber_text
                    extraction_method = "pdfplumber"
                    logger.info(f"Successfully extracted {len(text)} characters using pdfplumber")
                    
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
        
        # Method 2: Fallback to PyPDF2 if pdfplumber fails or returns insufficient text
        if len(text.strip()) < 100:
            try:
                logger.info("Attempting text extraction with PyPDF2...")
                file_like_object = io.BytesIO(content)
                reader = PdfReader(file_like_object)
                
                pypdf2_text = ""
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        pypdf2_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                
                # Use PyPDF2 result if it extracted more text
                if len(pypdf2_text.strip()) > len(text.strip()):
                    text = pypdf2_text
                    extraction_method = "PyPDF2"
                    logger.info(f"PyPDF2 extracted more text ({len(pypdf2_text)} chars), using PyPDF2 result")
                    
            except Exception as e:
                logger.warning(f"PyPDF2 extraction failed: {e}")
        
        # Clean and preprocess the extracted text
        text = clean_extracted_text(text)
        
        if not text.strip():
            raise Exception("No text could be extracted from the PDF using any method")
        
        logger.info(f"Final extraction result: {len(text)} characters using {extraction_method}")
        return text
        
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise Exception(f"Error extracting text from PDF: {str(e)}")

def clean_extracted_text(text: str) -> str:
    """
    Clean and preprocess extracted text for better chunking and embedding
    
    Args:
        text (str): Raw extracted text
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace while preserving paragraph structure
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
    text = re.sub(r'\n[ ]+', '\n', text)  # Remove spaces after newlines
    text = re.sub(r'[ ]+\n', '\n', text)  # Remove spaces before newlines
    
    # Normalize different types of line breaks
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove excessive line breaks but keep paragraph structure
    text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines to double newline
    
    # Remove page headers/footers patterns (common in insurance documents)
    text = re.sub(r'\n--- Page \d+ ---\n', '\n', text)
    
    # Fix common OCR issues
    text = re.sub(r'(\w)\s+(\W)', r'\1\2', text)  # Remove space before punctuation
    text = re.sub(r'(\W)\s+(\w)', r'\1 \2', text)  # Ensure space after punctuation
    
    # Remove form feed characters
    text = text.replace('\f', '\n')
    
    # Fix broken words (common in PDF extraction)
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)  # Fix hyphenated words across lines
    
    # Clean up bullet points and numbering
    text = re.sub(r'\n\s*[•·▪▫◦]\s*', '\n• ', text)  # Standardize bullet points
    text = re.sub(r'\n\s*(\d+\.)\s*', r'\n\1 ', text)  # Clean numbered lists
    
    # Remove excessive punctuation
    text = re.sub(r'\.{3,}', '...', text)  # Multiple dots to ellipsis
    
    return text.strip()

def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 300) -> List[str]:
    """
    Enhanced text chunking with better boundary detection and context preservation
    
    Args:
        text (str): Text to be chunked
        chunk_size (int): Target size of each chunk (increased default)
        overlap (int): Overlap between chunks (increased default)
        
    Returns:
        List[str]: List of text chunks
    """
    if not text or not text.strip():
        return []
    
    logger.info(f"Chunking text of length {len(text)} with chunk_size={chunk_size}, overlap={overlap}")
    
    # First, try to split by major sections (useful for policy documents)
    chunks = []
    sections = split_by_sections(text)
    
    for section in sections:
        if len(section) <= chunk_size:
            # Section fits in one chunk
            chunks.append(section.strip())
        else:
            # Section needs to be split further
            section_chunks = chunk_by_sentences(section, chunk_size, overlap)
            chunks.extend(section_chunks)
    
    # Remove empty chunks and very short chunks
    final_chunks = []
    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) > 50:  # Minimum chunk size
            final_chunks.append(chunk)
    
    logger.info(f"Created {len(final_chunks)} chunks")
    return final_chunks

def split_by_sections(text: str) -> List[str]:
    """
    Split text by major sections (useful for policy documents)
    
    Args:
        text (str): Input text
        
    Returns:
        List[str]: List of sections
    """
    # Common section headers in insurance documents
    section_patterns = [
        r'\n\s*(?:SECTION|Section)\s+\d+',
        r'\n\s*(?:CLAUSE|Clause)\s+\d+',
        r'\n\s*(?:ARTICLE|Article)\s+\d+',
        r'\n\s*\d+\.\s+[A-Z][A-Z\s]{10,}',  # Numbered major headings
        r'\n\s*[A-Z]{3,}(?:\s+[A-Z]{3,})*\s*\n',  # ALL CAPS headings
    ]
    
    # Try to find section breaks
    import re
    split_points = [0]
    
    for pattern in section_patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        for match in matches:
            if match.start() not in split_points:
                split_points.append(match.start())
    
    split_points.append(len(text))
    split_points.sort()
    
    sections = []
    for i in range(len(split_points) - 1):
        section = text[split_points[i]:split_points[i + 1]]
        if section.strip():
            sections.append(section)
    
    # If no major sections found, return the whole text
    if len(sections) <= 1:
        return [text]
    
    return sections

def chunk_by_sentences(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Chunk text by sentences for better context preservation
    
    Args:
        text (str): Text to chunk
        chunk_size (int): Target chunk size
        overlap (int): Overlap between chunks
        
    Returns:
        List[str]: List of chunks
    """
    # Split into sentences using multiple delimiters
    sentence_endings = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(sentence_endings, text)
    
    if len(sentences) <= 1:
        # Fallback to paragraph splitting
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            sentences = paragraphs
        else:
            # Final fallback to word-based chunking
            return chunk_by_words(text, chunk_size, overlap)
    
    chunks = []
    current_chunk = ""
    current_size = 0
    sentence_buffer = []  # For overlap
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_size = len(sentence)
        
        # If adding this sentence exceeds chunk size, finalize current chunk
        if current_size + sentence_size > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            
            # Create overlap from recent sentences
            overlap_text = ""
            overlap_size = 0
            
            # Add sentences from buffer for overlap
            for buffered_sentence in reversed(sentence_buffer[-5:]):  # Last 5 sentences max
                if overlap_size + len(buffered_sentence) <= overlap:
                    overlap_text = buffered_sentence + " " + overlap_text
                    overlap_size += len(buffered_sentence)
                else:
                    break
            
            current_chunk = overlap_text + sentence
            current_size = len(current_chunk)
            sentence_buffer = [sentence]
        else:
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
            current_size += sentence_size
            sentence_buffer.append(sentence)
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def chunk_by_words(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Fallback word-based chunking when sentence splitting fails
    
    Args:
        text (str): Text to chunk
        chunk_size (int): Target chunk size
        overlap (int): Overlap between chunks
        
    Returns:
        List[str]: List of chunks
    """
    words = text.split()
    chunks = []
    
    i = 0
    while i < len(words):
        chunk_words = []
        chunk_length = 0
        
        # Build chunk word by word
        while i < len(words) and chunk_length < chunk_size:
            word = words[i]
            if chunk_length + len(word) + 1 <= chunk_size:  # +1 for space
                chunk_words.append(word)
                chunk_length += len(word) + 1
                i += 1
            else:
                break
        
        if chunk_words:
            chunks.append(' '.join(chunk_words))
        
        # Calculate overlap for next chunk
        if overlap > 0 and len(chunk_words) > 1:
            overlap_words = 0
            overlap_length = 0
            
            # Count words from the end for overlap
            for j in range(len(chunk_words) - 1, -1, -1):
                word_len = len(chunk_words[j]) + 1
                if overlap_length + word_len <= overlap:
                    overlap_length += word_len
                    overlap_words += 1
                else:
                    break
            
            # Move back by overlap amount
            i -= overlap_words
    
    return chunks

def validate_pdf_content(content: bytes) -> Dict[str, Any]:
    """
    Validate PDF content and return diagnostic information
    
    Args:
        content (bytes): PDF file content
        
    Returns:
        Dict[str, Any]: Validation results and diagnostics
    """
    try:
        # Basic file validation
        if not content or len(content) < 100:
            return {"valid": False, "error": "File too small or empty"}
        
        # Check PDF magic bytes
        if not content.startswith(b'%PDF-'):
            return {"valid": False, "error": "Not a valid PDF file (missing PDF header)"}
        
        # Try to get basic PDF info
        diagnostics = {
            "valid": True,
            "file_size": len(content),
            "has_pdf_header": True
        }
        
        # Try to extract metadata and page count
        try:
            file_obj = io.BytesIO(content)
            reader = PdfReader(file_obj)
            
            diagnostics.update({
                "page_count": len(reader.pages),
                "is_encrypted": reader.is_encrypted,
                "metadata": reader.metadata if hasattr(reader, 'metadata') else {}
            })
            
            # Sample text extraction from first page
            if len(reader.pages) > 0:
                first_page_text = reader.pages[0].extract_text()
                diagnostics.update({
                    "first_page_text_length": len(first_page_text),
                    "first_page_preview": first_page_text[:200] if first_page_text else "No text found"
                })
            
        except Exception as e:
            diagnostics.update({
                "extraction_error": str(e),
                "can_extract_metadata": False
            })
        
        return diagnostics
        
    except Exception as e:
        return {"valid": False, "error": f"Validation failed: {str(e)}"}

def get_text_statistics(text: str) -> Dict[str, Any]:
    """
    Get comprehensive statistics about extracted text
    
    Args:
        text (str): Extracted text
        
    Returns:
        Dict[str, Any]: Text statistics
    """
    if not text:
        return {"error": "No text provided"}
    
    # Basic statistics
    stats = {
        "character_count": len(text),
        "word_count": len(text.split()),
        "line_count": text.count('\n') + 1,
        "paragraph_count": len([p for p in text.split('\n\n') if p.strip()]),
    }
    
    # Content analysis
    sentences = re.split(r'[.!?]+', text)
    stats["sentence_count"] = len([s for s in sentences if s.strip()])
    
    # Look for insurance-specific terms
    insurance_terms = [
        'policy', 'premium', 'coverage', 'claim', 'deductible', 'benefit',
        'exclusion', 'waiting period', 'pre-existing', 'maternity', 'hospital',
        'medical', 'treatment', 'insured', 'policyholder', 'renewal'
    ]
    
    found_terms = []
    text_lower = text.lower()
    for term in insurance_terms:
        if term in text_lower:
            count = text_lower.count(term)
            found_terms.append({"term": term, "count": count})
    
    stats["insurance_terms_found"] = found_terms
    stats["estimated_policy_document"] = len(found_terms) > 5
    
    return stats
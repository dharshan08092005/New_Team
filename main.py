from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils.pdf_utils import extract_text_from_pdf, chunk_text
from utils.embed_utils import insert_into_pinecone, search_similar_chunks
import google.generativeai as genai
import os
from pathlib import Path
from urllib.parse import urlparse
import requests
import re

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
chat_model = genai.GenerativeModel('gemini-2.5-flash')

app = FastAPI(title="PDF Bot API")

class HackRXRequest(BaseModel):
    documents: str  # URL or local path
    questions: list[str]

def is_url(string: str) -> bool:
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except:
        return False

def is_local_path(string: str) -> bool:
    return Path(string).exists()

def download_file_from_url(url: str) -> bytes:
    headers = {
        'User-Agent': 'Mozilla/5.0'
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    if 'pdf' not in response.headers.get('content-type', '').lower() and not url.lower().endswith('.pdf'):
        if not response.content.startswith(b'%PDF-'):
            raise HTTPException(status_code=400, detail="URL does not point to a valid PDF")
    return response.content

def read_local_file(file_path: str) -> bytes:
    path = Path(file_path)
    if not path.exists() or not path.suffix.lower() == '.pdf':
        raise HTTPException(status_code=400, detail="Invalid file path or not a PDF")
    return path.read_bytes()

def clean_extracted_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\f', '\n').replace('\r', '')
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def process_document_content_sync(content: bytes, source_name: str) -> dict:
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 10MB.")

    text = extract_text_from_pdf(content)
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text found in document")

    text = clean_extracted_text(text)
    chunks = chunk_text(text, chunk_size=1500, overlap=300)
    if not chunks:
        raise HTTPException(status_code=400, detail="Failed to chunk document")

    insert_into_pinecone(chunks, metadata={
        "source": source_name,
        "total_chunks": len(chunks),
        "original_text_length": len(text)
    })

    return {
        "chunks_created": len(chunks),
        "text_length": len(text),
        "status": "success"
    }

@app.post("/hackrx/run")
async def hackrx_run(request: HackRXRequest):
    try:
        document_source = request.documents.strip()
        questions = request.questions

        if not document_source or not questions:
            raise HTTPException(status_code=400, detail="Missing document or questions")

        if is_url(document_source):
            content = download_file_from_url(document_source)
            source_name = f"URL: {document_source}"
        elif is_local_path(document_source):
            content = read_local_file(document_source)
            source_name = f"Local: {document_source}"
        else:
            raise HTTPException(status_code=400, detail="Invalid document source")

        processing_result = process_document_content_sync(content, source_name)

        answers = []
        for question in questions:
            if not question.strip():
                answers.append("Invalid question")
                continue

            context_chunks = search_similar_chunks(question, top_k=10)
            context = "\n\n".join(context_chunks)

            if not context.strip():
                answers.append("Sorry, no relevant information found.")
                continue

            prompt = f"""
    You are an expert insurance document analyst. Your task is to extract accurate and complete answers to user questions strictly from the provided document context.

    Output Format:
    - Respond in clear, plain English in a formal tone.
    - Read the document from start to end clearly.
    - Each answer should be **informative, policy-specific, and self-contained**.
    - Include eligibility criteria, time limits, conditions, or caps mentioned in the policy.
    - Use **single-line summaries** where possible, and expand only to list specific clauses or conditions.
    - Structure complex answers using numbering or semicolons—but avoid paragraph-style explanations.
    - DO NOT hallucinate or infer. Only use the information directly found in the document.
    - If no relevant answer is found, respond exactly with: "No relevant information found in the document."
    - If the document partially covers the question, say: "Partially covered" followed by the relevant clause.

    
Context:
{context}

Question: {question}

Answer:"""

            response = chat_model.generate_content(prompt)
            answers.append(response.text.strip())

        return {
            "answers": answers,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
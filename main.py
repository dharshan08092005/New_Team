from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from utils.pdf_utils import extract_text_from_pdf, chunk_text
from utils.embed_utils import insert_into_pinecone, search_similar_chunks
import google.generativeai as genai
import os
import asyncio
import hashlib
import re
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from cachetools import TTLCache
import httpx
from functools import lru_cache

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
chat_model = genai.GenerativeModel('gemini-2.5-flash')

app = FastAPI(title="PDF Bot API")

# Authentication
security = HTTPBearer()
AUTH_TOKEN = os.getenv("AUTHORIZE_TOKEN")

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify the authentication token"""
    if credentials.credentials != AUTH_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Caching
document_cache = TTLCache(maxsize=100, ttl=3600)  # 1 hour
qa_cache = TTLCache(maxsize=1000, ttl=1800)  # 30 minutes

# HTTP client with connection pooling
http_client = httpx.AsyncClient(
    timeout=30.0,
    limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
)

# Thread pool for CPU-bound operations
executor = ThreadPoolExecutor(max_workers=4)

# Compiled regex patterns
whitespace_pattern = re.compile(r'\s+')
newline_pattern = re.compile(r'\n+')

class HackRXRequest(BaseModel):
    documents: str
    questions: list[str]

def get_content_hash(content: bytes) -> str:
    return hashlib.md5(content).hexdigest()

def is_url(string: str) -> bool:
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except:
        return False

def is_local_path(string: str) -> bool:
    return Path(string).exists()

async def download_file_from_url_async(url: str) -> bytes:
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = await http_client.get(url, headers=headers)
    response.raise_for_status()

    content_type = response.headers.get('content-type', '').lower()
    if 'pdf' not in content_type and not url.lower().endswith('.pdf'):
        if not response.content.startswith(b'%PDF-'):
            raise HTTPException(status_code=400, detail="URL does not point to a valid PDF")
    return response.content

def read_local_file(file_path: str) -> bytes:
    path = Path(file_path)
    if not path.exists() or not path.suffix.lower() == '.pdf':
        raise HTTPException(status_code=400, detail="Invalid file path or not a PDF")
    return path.read_bytes()

@lru_cache(maxsize=128)
def clean_extracted_text(text: str) -> str:
    text = whitespace_pattern.sub(' ', text)
    text = text.replace('\f', '\n').replace('\r', '')
    text = newline_pattern.sub('\n', text)
    return text.strip()

async def process_document_content_async(content: bytes, source_name: str) -> dict:
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 10MB.")

    content_hash = get_content_hash(content)
    
    # Check cache first
    if content_hash in document_cache:
        return document_cache[content_hash]

    # Process in thread pool
    loop = asyncio.get_event_loop()
    
    def process_sync():
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
            "original_text_length": len(text),
            "content_hash": content_hash
        })

        return {
            "chunks_created": len(chunks),
            "text_length": len(text),
            "status": "success",
            "content_hash": content_hash
        }

    result = await loop.run_in_executor(executor, process_sync)
    document_cache[content_hash] = result
    return result

async def process_question_async(question: str, content_hash: str) -> str:
    if not question.strip():
        return "Invalid question"

    # Check cache
    cache_key = f"{content_hash}:{question}"
    if cache_key in qa_cache:
        return qa_cache[cache_key]

    loop = asyncio.get_event_loop()
    
    def process_sync():
        context_chunks = search_similar_chunks(question, top_k=8)
        context = "\n\n".join(context_chunks)

        if not context.strip():
            return "Sorry, no relevant information found."

        prompt = f"""
You are an expert insurance document analyst. Your task is to provide clear, definitive answers to user questions using only the information provided in the document context.

Critical Instructions:

1. Answer Format:
   - Write in simple, professional language that sounds authoritative and helpful.
   - Provide complete answers in exactly 1-2 sentences - no more, no less.
   - For Yes/No questions, always start with "Yes" or "No" followed by the explanation.
   - For definition questions, provide the definition directly without prefacing words.
   - For "what is" questions, state the answer definitively (e.g., "The waiting period is X" not "There is a waiting period of X").
   - Give each answer in a new line

2. Content Precision:
   - Include ALL relevant numbers, time periods, percentages, and monetary amounts.
   - Mention key conditions and eligibility requirements naturally within the main sentence.
   - When multiple related points exist, integrate them into one flowing sentence using connecting words.
   - Always specify who benefits (e.g., "insured person", "female insured person") when relevant.

3. Language Transformation Rules:
   - Change "expenses are excluded" → "there is a waiting period"
   - Change "coverage is not available" → "grace period is provided"  
   - Change "the policy covers" → more specific language like "the policy reimburses" or "the policy indemnifies"
   - Avoid negative framing when positive framing conveys the same information better.

4. Specific Response Patterns:
   - Grace period: "A grace period of [X] days is provided for premium payment after the due date..."
   - Waiting periods: "There is a waiting period of [X] months/years for [condition]..."
   - Coverage confirmation: "Yes, the policy covers/reimburses [benefit] provided [conditions]..."
   - Definitions: "[Term] is defined as [definition with specific criteria]..."
   - Limits: "Yes, [benefit] is capped at [X]% of [basis] and [additional details]..."

5. Information Mining:
   - Search thoroughly through the entire context before concluding information is missing.
   - Look for information under different headings, synonyms, and related terms.
   - Only use "No relevant information found in the document." as a last resort.
   - If partial information exists, provide what is available rather than saying nothing is found.

6. Quality Standards:
   - Every answer must be self-contained and complete.
   - Include the most important details first, then supporting information.
   - Ensure answers sound confident and authoritative, not tentative.
   - Use specific numbers and exact terms from the document.

Context:
{context}

Question: {question}

Answer:"""

        response = chat_model.generate_content(prompt)
        return response.text.strip()

    answer = await loop.run_in_executor(executor, process_sync)
    qa_cache[cache_key] = answer

    return answer

@app.post("/hackrx/run")
async def hackrx_run(request: HackRXRequest, token: str = Depends(verify_token)):
    try:
        document_source = request.documents.strip()
        questions = request.questions

        if not document_source or not questions:
            raise HTTPException(status_code=400, detail="Missing document or questions")

        # Get document content
        if is_url(document_source):
            content = await download_file_from_url_async(document_source)
            source_name = f"URL: {document_source}"
        elif is_local_path(document_source):
            content = read_local_file(document_source)
            source_name = f"Local: {document_source}"
        else:
            raise HTTPException(status_code=400, detail="Invalid document source")

        # Process document
        processing_result = await process_document_content_async(content, source_name)
        content_hash = processing_result["content_hash"]

        # Filter valid questions
        valid_questions = [q for q in questions if q.strip()]
        if not valid_questions:
            return {"answers": ["Invalid question"] * len(questions)}

        # Run queries concurrently
        tasks = [process_question_async(q, content_hash) for q in valid_questions]
        answers = await asyncio.gather(*tasks)

        # Map back to original order (with invalids)
        result_answers = []
        valid_idx = 0
        for question in questions:
            if question.strip():
                result_answers.append(answers[valid_idx])
                valid_idx += 1
            else:
                result_answers.append("Invalid question")

        # Clean up formatting (remove escaped characters)
        cleaned_answers = [
            answer.replace("\\n", "\n").replace('\\"', '"').strip()
            for answer in result_answers
        ]

        return {"answers": cleaned_answers}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint (no auth required)
@app.get("/health")
async def health_check():
    return {"status": "healthy"}
    
@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()
    executor.shutdown(wait=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
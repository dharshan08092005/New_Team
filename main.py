from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from utils.pdf_utils import extract_text_from_pdf, chunk_text
from utils.embed_utils import insert_into_pinecone, search_similar_chunks
import uvicorn
import google.generativeai as genai
import os

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
chat_model = genai.GenerativeModel('gemini-2.5-flash')

app = FastAPI(title="PDF Bot API", description="API for PDF text extraction and similarity search")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.get("/")
async def root():
    return {"message": "PDF Bot API is running!"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Check file size (optional - limit to 10MB)
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(status_code=400, detail="File size too large. Maximum 10MB allowed.")
        
        # Extract text from PDF
        text = extract_text_from_pdf(content)
        
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")
        
        # Chunk the text
        chunks = chunk_text(text, chunk_size=1000, overlap=200)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No text chunks could be created")
        
        # Insert into Pinecone
        insert_into_pinecone(chunks, metadata={"filename": file.filename})
        
        return {
            "message": "PDF uploaded and processed successfully",
            "filename": file.filename,
            "chunks_created": len(chunks),
            "text_length": len(text),
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in upload_pdf: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

class QueryRequest(BaseModel):
    queries: list[str]  # Accept multiple questions

@app.post("/query")
async def query_pdf(request: QueryRequest):
    try:
        answers = []

        for query in request.queries:
            query = query.strip()
            if not query:
                continue

            # Search similar chunks
            context_chunks = search_similar_chunks(query, top_k=5)
            context = "\n\n".join(context_chunks)

            # Prompt Gemini with the context
            prompt = f"""Use the following context to answer the user's question. 
If the answer is not found in the context, say 'Sorry, I couldn't find that in the document.'

Context:
{context}

Question: {query}
Answer:"""

            response = chat_model.generate_content(prompt)
            answer = response.text.strip()

            # Add only the answer text to the answers list
            answers.append(answer)

        return {
            "answers": answers
        }

    except Exception as e:
        print(f"Error in query_pdf: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing queries: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "PDF Bot API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
import uuid
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup - Use environment variables for security
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Pinecone with the new method
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index with 1024 dimensions (only run this once)
index_name = os.environ.get("PINECONE_INDEX_NAME")

# Check if index exists, if not create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1024,  # Set dimension to 1024
        metric='cosine',  # or 'euclidean' or 'dotproduct'
        spec=ServerlessSpec(
            cloud=os.environ.get("PINECONE_CLOUD"),  # or 'gcp' or 'azure'
            region=os.environ.get("PINECONE_REGION")  # choose appropriate region
        )
    )
    print(f"Created new index '{index_name}' with 1024 dimensions")
else:
    print(f"Index '{index_name}' already exists")

# Connect to the index
index = pc.Index(index_name)

# Get index stats to verify dimension
stats = index.describe_index_stats()
print(f"Index dimension: {stats.get('dimension', 'Unknown')}")

# Embedding function with error handling and padding to 1024 dimensions
def get_embedding(text: str):
    try:
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        res = genai.embed_content(
            model="models/embedding-001",
            content=text.strip(),
            task_type="retrieval_document"
        )
        embedding = res["embedding"]
        
        # Pad embedding to 1024 dimensions if needed
        if len(embedding) < 1024:
            # Pad with zeros to reach 1024 dimensions
            padding = [0.0] * (1024 - len(embedding))
            embedding.extend(padding)
            print(f"Padded embedding from {len(res['embedding'])} to {len(embedding)} dimensions")
        elif len(embedding) > 1024:
            # Truncate if somehow longer than 1024
            embedding = embedding[:1024]
            print(f"Truncated embedding to 1024 dimensions")
        
        return embedding
        
    except Exception as e:
        print(f"Error getting embedding for text: {str(e)}")
        raise

# Inserting vectors with better error handling
def insert_into_pinecone(chunks: list, metadata: dict = {}):
    try:
        if not chunks:
            print("No chunks to insert")
            return
        
        vectors = []
        for i, chunk in enumerate(chunks):
            if not chunk or not chunk.strip():
                print(f"Skipping empty chunk at index {i}")
                continue
                
            print(f"Processing chunk {i+1}/{len(chunks)}")
            vec = get_embedding(chunk)
            
            # Ensure vector dimension matches index dimension
            if len(vec) != 1024:
                print(f"Warning: Vector dimension {len(vec)} doesn't match index dimension 1024")
                # You might need to pad or truncate here depending on your needs
            
            vectors.append({
                "id": str(uuid.uuid4()),
                "values": vec,
                "metadata": {"text": chunk, **metadata}
            })
        
        if vectors:
            index.upsert(vectors=vectors)
            print(f"Successfully inserted {len(vectors)} vectors into Pinecone")
        else:
            print("No valid vectors to insert")
            
    except Exception as e:
        print(f"Error inserting into Pinecone: {str(e)}")
        raise

# Querying vectors with error handling
def search_similar_chunks(query: str, top_k=5):
    try:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
            
        query_vector = get_embedding(query)
        res = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
        return [match['metadata']['text'] for match in res['matches']]
    except Exception as e:
        print(f"Error searching similar chunks: {str(e)}")
        raise
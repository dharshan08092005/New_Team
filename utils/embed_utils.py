# embed_util.py
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
import uuid
import os
import re
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple
from datetime import datetime
import numpy as np

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

# Enhanced index creation with better error handling
def initialize_pinecone_index():
    """Initialize Pinecone index with proper error handling"""
    try:
        # Check if index exists, if not create it
        existing_indexes = pc.list_indexes().names()
        
        if index_name not in existing_indexes:
            pc.create_index(
                name=index_name,
                dimension=1024,  # Set dimension to 1024
                metric='cosine',  # Best for semantic similarity
                spec=ServerlessSpec(
                    cloud=os.environ.get("PINECONE_CLOUD", "aws"),
                    region=os.environ.get("PINECONE_REGION", "us-east-1")
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
        print(f"Total vectors in index: {stats.get('total_vector_count', 0)}")
        
        return index
        
    except Exception as e:
        print(f"Error initializing Pinecone index: {str(e)}")
        raise

# Initialize the index
index = initialize_pinecone_index()

def preprocess_query(query: str) -> List[str]:
    """Generate multiple query variations for better search"""
    query = query.strip().lower()
    variations = [query]
    
    # Common insurance term mappings for better retrieval
    term_mappings = {
        'ncd': ['no claim discount', 'no claim bonus', 'cumulative bonus', 'loyalty bonus'],
        'no claim discount': ['ncd', 'no claim bonus', 'cumulative bonus', 'renewal discount'],
        'no claim bonus': ['ncd', 'no claim discount', 'cumulative bonus'],
        'waiting period': ['wait period', 'exclusion period', 'cooling period'],
        'pre-existing': ['pre existing', 'ped', 'pre-existing disease', 'existing condition'],
        'maternity': ['pregnancy', 'childbirth', 'delivery', 'obstetric', 'confinement'],
        'room rent': ['accommodation', 'boarding', 'room charges', 'lodging'],
        'icu': ['intensive care unit', 'critical care', 'iccu'],
        'ayush': ['ayurveda', 'yoga', 'naturopathy', 'unani', 'siddha', 'homeopathy'],
        'organ donor': ['organ donation', 'transplant', 'harvesting', 'donor coverage'],
        'health checkup': ['preventive checkup', 'medical checkup', 'annual checkup', 'wellness'],
        'hospital': ['medical facility', 'healthcare facility', 'nursing home'],
        'grace period': ['grace time', 'payment grace', 'premium grace']
    }
    
    # Add variations based on mappings
    query_words = query.split()
    for key, synonyms in term_mappings.items():
        # Check if key phrase is in query
        if key in query:
            for synonym in synonyms:
                new_query = query.replace(key, synonym)
                if new_query not in variations:
                    variations.append(new_query)
        
        # Check individual words
        for synonym in synonyms:
            if synonym in query:
                new_query = query.replace(synonym, key)
                if new_query not in variations:
                    variations.append(new_query)
    
    # Add partial matches for compound terms
    for word in query_words:
        if len(word) > 4:  # Only for longer words
            variations.append(word)
    
    print(f"Generated {len(variations)} query variations for: '{query}'")
    return list(set(variations))  # Remove duplicates

def get_embedding(text: str, task_type: str = "retrieval_document"):
    """Enhanced embedding function with better error handling"""
    try:
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Clean text before embedding
        text = text.strip()
        if len(text) > 10000:  # Truncate very long texts
            text = text[:10000]
            print(f"Truncated text to 10000 characters for embedding")
        
        res = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type=task_type
        )
        embedding = res["embedding"]
        
        # Ensure exactly 1024 dimensions
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
        print(f"Text preview: {text[:200]}...")
        raise

def insert_into_pinecone(chunks: List[str], metadata: Dict[str, Any] = {}):
    """Enhanced insertion with better metadata and batch processing"""
    try:
        if not chunks:
            print("No chunks to insert")
            return
        
        print(f"Starting to process {len(chunks)} chunks...")
        
        # Enhanced metadata with processing info
        base_metadata = {
            **metadata,
            'insertion_timestamp': datetime.now().isoformat(),
            'total_chunks': len(chunks),
            'processing_version': '2.0'
        }
        
        vectors = []
        successful_chunks = 0
        failed_chunks = 0
        
        for i, chunk in enumerate(chunks):
            try:
                if not chunk or not chunk.strip():
                    print(f"Skipping empty chunk at index {i}")
                    failed_chunks += 1
                    continue
                
                print(f"Processing chunk {i+1}/{len(chunks)} (length: {len(chunk)} chars)")
                vec = get_embedding(chunk, "retrieval_document")
                
                # Enhanced metadata for each chunk
                chunk_metadata = {
                    **base_metadata,
                    'text': chunk,
                    'chunk_index': i,
                    'chunk_length': len(chunk),
                    'chunk_words': len(chunk.split()),
                    'chunk_preview': chunk[:100] + "..." if len(chunk) > 100 else chunk
                }
                
                vectors.append({
                    "id": str(uuid.uuid4()),
                    "values": vec,
                    "metadata": chunk_metadata
                })
                
                successful_chunks += 1
                
            except Exception as chunk_error:
                print(f"Error processing chunk {i}: {str(chunk_error)}")
                failed_chunks += 1
                continue
        
        # Batch insert with progress tracking
        if vectors:
            batch_size = 100  # Process in batches to avoid timeouts
            total_batches = (len(vectors) + batch_size - 1) // batch_size
            
            for batch_idx in range(0, len(vectors), batch_size):
                batch = vectors[batch_idx:batch_idx + batch_size]
                try:
                    index.upsert(vectors=batch)
                    print(f"Inserted batch {(batch_idx // batch_size) + 1}/{total_batches} ({len(batch)} vectors)")
                except Exception as batch_error:
                    print(f"Error inserting batch {batch_idx // batch_size + 1}: {str(batch_error)}")
                    raise
            
            print(f"Successfully inserted {successful_chunks} vectors into Pinecone")
            print(f"Failed to process {failed_chunks} chunks")
            
            # Get updated index stats
            stats = index.describe_index_stats()
            print(f"Total vectors in index: {stats.get('total_vector_count', 0)}")
            
        else:
            print("No valid vectors to insert")
            
    except Exception as e:
        print(f"Error inserting into Pinecone: {str(e)}")
        raise

def search_similar_chunks(query: str, top_k: int = 8, similarity_threshold: float = 0.7) -> List[str]:
    """Enhanced similarity search with query preprocessing and filtering"""
    try:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        print(f"Searching for: '{query}' (top_k={top_k}, threshold={similarity_threshold})")
        
        # Generate query variations
        query_variations = preprocess_query(query)
        
        all_matches = []
        seen_texts = set()
        
        # Search with each query variation
        for variation in query_variations[:3]:  # Limit to top 3 variations to avoid too many API calls
            try:
                query_vector = get_embedding(variation, "retrieval_query")
                res = index.query(
                    vector=query_vector, 
                    top_k=top_k * 2,  # Get more results to filter
                    include_metadata=True,
                    include_values=False
                )
                
                # Filter by similarity threshold and remove duplicates
                for match in res['matches']:
                    score = match.get('score', 0)
                    text = match['metadata'].get('text', '')
                    
                    if score >= similarity_threshold and text not in seen_texts:
                        all_matches.append({
                            'text': text,
                            'score': score,
                            'query_variation': variation
                        })
                        seen_texts.add(text)
                        
            except Exception as variation_error:
                print(f"Error searching with variation '{variation}': {str(variation_error)}")
                continue
        
        # Sort by score and return top results
        all_matches.sort(key=lambda x: x['score'], reverse=True)
        final_results = [match['text'] for match in all_matches[:top_k]]
        
        print(f"Found {len(final_results)} relevant chunks (from {len(all_matches)} total matches)")
        for i, match in enumerate(all_matches[:top_k]):
            print(f"  Match {i+1}: Score {match['score']:.3f} - {match['text'][:100]}...")
        
        return final_results
        
    except Exception as e:
        print(f"Error searching similar chunks: {str(e)}")
        # Fallback to simple search
        try:
            print("Attempting fallback search...")
            query_vector = get_embedding(query, "retrieval_query")
            res = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
            return [match['metadata']['text'] for match in res['matches']]
        except Exception as fallback_error:
            print(f"Fallback search also failed: {str(fallback_error)}")
            raise

def get_index_statistics() -> Dict[str, Any]:
    """Get comprehensive statistics about the Pinecone index"""
    try:
        stats = index.describe_index_stats()
        return {
            'total_vector_count': stats.get('total_vector_count', 0),
            'dimension': stats.get('dimension', 0),
            'index_fullness': stats.get('index_fullness', 0),
            'namespaces': stats.get('namespaces', {}),
            'index_name': index_name
        }
    except Exception as e:
        print(f"Error getting index statistics: {str(e)}")
        return {'error': str(e)}

def clear_index(confirm: bool = False):
    """Clear all vectors from the index (use with caution!)"""
    if not confirm:
        print("This will delete ALL vectors from the index. Call with confirm=True to proceed.")
        return
    
    try:
        # Delete all vectors (this is a destructive operation)
        index.delete(delete_all=True)
        print(f"Successfully cleared all vectors from index '{index_name}'")
    except Exception as e:
        print(f"Error clearing index: {str(e)}")
        raise

def search_by_metadata(metadata_filter: Dict[str, Any], top_k: int = 10) -> List[Dict[str, Any]]:
    """Search vectors by metadata filters"""
    try:
        res = index.query(
            vector=[0.0] * 1024,  # Dummy vector
            filter=metadata_filter,
            top_k=top_k,
            include_metadata=True
        )
        return [
            {
                'text': match['metadata'].get('text', ''),
                'metadata': match['metadata'],
                'score': match.get('score', 0)
            }
            for match in res['matches']
        ]
    except Exception as e:
        print(f"Error searching by metadata: {str(e)}")
        raise
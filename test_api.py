import google.generativeai as genai
from pinecone import Pinecone
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def test_google_ai_api():
    """Test Google AI API connection"""
    try:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("❌ GOOGLE_API_KEY not found in environment variables")
            return False
            
        genai.configure(api_key=api_key)
        
        # Test embedding
        result = genai.embed_content(
            model="models/embedding-001",
            content="This is a test",
            task_type="retrieval_document"
        )
        
        print("✅ Google AI API is working!")
        print(f"Embedding dimension: {len(result['embedding'])}")
        return True
        
    except Exception as e:
        print("❌ Google AI API test failed:")
        print(f"Error: {e}")
        return False

def test_pinecone_api():
    """Test Pinecone API connection"""
    try:
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            print("❌ PINECONE_API_KEY not found in environment variables")
            return False
            
        pc = Pinecone(api_key=api_key)
        
        # List indexes
        indexes = pc.list_indexes()
        print("✅ Pinecone API is working!")
        print(f"Available indexes: {[idx.name for idx in indexes]}")
        
        # Test connection to your index
        index_name = "pdf-bot-index"  # Replace with your actual index name
        if any(idx.name == index_name for idx in indexes):
            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            print(f"✅ Connected to index '{index_name}'")
            print(f"Index stats: {stats}")
        else:
            print(f"⚠️  Index '{index_name}' not found. Available indexes: {[idx.name for idx in indexes]}")
        
        return True
        
    except Exception as e:
        print("❌ Pinecone API test failed:")
        print(f"Error: {e}")
        return False

def main():
    print("Testing API connections...\n")
    
    google_ok = test_google_ai_api()
    print()
    pinecone_ok = test_pinecone_api()
    
    print("\n" + "="*50)
    if google_ok and pinecone_ok:
        print("✅ All API connections are working!")
        print("You can now run your PDF Bot application.")
    else:
        print("❌ Some API connections failed.")
        print("Please check your API keys and try again.")

if __name__ == "__main__":
    main()
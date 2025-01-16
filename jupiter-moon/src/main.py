from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import os

# Load environment variables
load_dotenv()

def create_embeddings():
    """Create and return a LangChain OpenAI embeddings object."""
    
    # Initialize the OpenAI embeddings with the API key from .env
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-ada-002",  # Using the latest embedding model
        chunk_size=1000  # Process 1000 texts at a time for efficiency
    )
    
    return embeddings

def main():
    # Create the embeddings object
    embeddings = create_embeddings()
    
    # Test the embeddings with a sample text
    test_text = "Testing the embedding functionality"
    vector = embeddings.embed_query(test_text)
    
    print(f"Successfully created embeddings!")
    print(f"Vector dimension: {len(vector)}")

if __name__ == "__main__":
    main() 
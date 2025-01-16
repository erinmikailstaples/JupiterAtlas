from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import os
from typing import List, Optional

# Load environment variables
load_dotenv()

def create_embeddings(batch_size: Optional[int] = 1000):
    """Create and return a LangChain OpenAI embeddings object with enhanced configuration.
    
    Args:
        batch_size: Number of texts to process in each batch for efficiency
        
    Returns:
        OpenAIEmbeddings: Configured embeddings object
    """
    
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-ada-002",  # Latest embedding model
        chunk_size=batch_size,
        timeout=60,  # Add timeout for robustness
        max_retries=3,  # Add retries for reliability
        show_progress_bar=True  # Show progress for large batches
    )
    
    return embeddings

def embed_with_error_handling(texts: List[str], embeddings: OpenAIEmbeddings):
    """Embed texts with error handling and retries.
    
    Args:
        texts: List of texts to embed
        embeddings: OpenAIEmbeddings object
        
    Returns:
        List of embeddings vectors
    """
    try:
        return embeddings.embed_documents(texts)
    except Exception as e:
        print(f"Error during embedding: {str(e)}")
        # Retry with smaller batches if we hit rate limits
        if "rate_limit" in str(e).lower():
            print("Rate limit hit, retrying with smaller batches...")
            results = []
            batch_size = len(texts) // 2
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                results.extend(embeddings.embed_documents(batch))
            return results
        raise

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
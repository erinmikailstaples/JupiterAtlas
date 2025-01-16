from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import time
from chunk import read_moons_data, create_moon_chunks, chunk_for_embedding
from main import create_embeddings

# Load environment variables
load_dotenv()

def create_vector_store():
    """Create and populate Pinecone vector store with Jupiter moons data."""
    
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # Configuration
    INDEX_NAME = "jupitermoons-2"
    NAMESPACE = "moonvector"
    
    # Create embeddings object
    embeddings = create_embeddings()
    
    # Process moon data
    df = read_moons_data('jupiter_moons.tsv')
    moon_chunks = create_moon_chunks(df)
    final_chunks = chunk_for_embedding(moon_chunks)
    
    # Create vector store from documents
    docsearch = PineconeVectorStore.from_texts(
        texts=[chunk["text"] for chunk in final_chunks],
        embedding=embeddings,
        index_name=INDEX_NAME,
        namespace=NAMESPACE,
        metadatas=[{
            "moon_name": chunk["metadata"]["moon_name"],
            "source_url": chunk["source_url"]
        } for chunk in final_chunks]
    )
    
    time.sleep(5)
    
    # Print index statistics
    print("Index after upsert:")
    print(pc.Index(INDEX_NAME).describe_index_stats())
    print("\n")

if __name__ == "__main__":
    create_vector_store()

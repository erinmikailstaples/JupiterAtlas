from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
from chunk import read_moons_data, create_moon_chunks, chunk_for_embedding
from main import create_embeddings

# Load environment variables
load_dotenv()

def init_pinecone():
    """Initialize Pinecone client and create index if it doesn't exist."""
    
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # Index configuration
    INDEX_NAME = "jupitermoons-2"
    DIMENSION = 1536  
    
    # Check if index already exists
    existing_indexes = pc.list_indexes()
    
    # Create index if it doesn't exist
    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Created new index: {INDEX_NAME}")
    
    # Get the index
    index = pc.Index(INDEX_NAME)
    return index

def upsert_documents(index, chunks, embeddings):
    """Embed and upsert documents into Pinecone index."""
    
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        # Create embeddings for the batch
        texts = [chunk["text"] for chunk in batch]
        embedded_batch = embeddings.embed_documents(texts)
        
        # Prepare vectors for upserting
        vectors = []
        for j, embedding in enumerate(embedded_batch):
            vectors.append({
                "id": f"moon_{i+j}",
                "values": embedding,
                "metadata": batch[j]["metadata"]
            })
        
        # Upsert to Pinecone
        index.upsert(vectors=vectors)
        print(f"Upserted batch {i//batch_size + 1}")

def main():
    # Initialize Pinecone and get index
    index = init_pinecone()
    
    # Create embeddings object
    embeddings = create_embeddings()
    
    # Process moon data
    df = read_moons_data('jupiter_moons.tsv')
    moon_chunks = create_moon_chunks(df)
    final_chunks = chunk_for_embedding(moon_chunks)
    
    # Upsert documents to Pinecone
    upsert_documents(index, final_chunks, embeddings)
    print("Completed upserting all documents to Pinecone")

if __name__ == "__main__":
    main()

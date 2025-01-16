from pinecone import Pinecone
from dotenv import load_dotenv
import os
import json
from pprint import pprint

# Load environment variables
load_dotenv()

def review_vectors():
    """Review individual vectors from the Pinecone index."""
    
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # Configuration
    INDEX_NAME = "jupitermoons-2"
    NAMESPACE = "moonvector"
    
    # Get the index
    index = pc.Index(INDEX_NAME)
    
    # List all vector IDs in the namespace and convert to list
    vector_ids = list(index.list(namespace=NAMESPACE))
    
    print(f"Found {len(vector_ids)} vectors in namespace {NAMESPACE}\n")
    
    # Query each vector and display its contents
    for vector_id in vector_ids:
        try:
            query_response = index.fetch(
                ids=[vector_id[0]],
                namespace=NAMESPACE
            )
            
            # Print raw response for debugging
            print(f"Raw response type: {type(query_response)}")
            print(f"Raw response: {query_response}")
            
            # Skip if no response
            if not query_response:
                print(f"No data found for Vector ID: {vector_id[0]}")
                continue
            
            # Format the response
            formatted_response = {
                "id": vector_id[0],
                "metadata": query_response.get("vectors", {}).get(vector_id[0], {}).get("metadata", {}),
                "values": query_response.get("vectors", {}).get(vector_id[0], {}).get("values", [])
            }
            
            # Truncate vector values for readability
            if formatted_response["values"]:
                formatted_response["values"] = {
                    "first_5": formatted_response["values"][:5],
                    "last_5": formatted_response["values"][-5:],
                    "total_dimensions": len(formatted_response["values"])
                }
            
            print(f"\nVector ID: {vector_id[0]}")
            pprint(formatted_response, indent=2)
            print("\n" + "="*80 + "\n")
            
        except Exception as e:
            print(f"Error processing Vector ID {vector_id[0]}: {str(e)}")
            continue

if __name__ == "__main__":
    review_vectors()

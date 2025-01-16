import pandas as pd
from typing import List, Dict
from dataclasses import dataclass
from langchain_text_splitters import MarkdownHeaderTextSplitter

@dataclass
class MoonChunk:
    moon_name: str
    content: str
    metadata: Dict
    source_url: str

def read_moons_data(file_path: str) -> pd.DataFrame:
    """Read the TSV file and return a DataFrame."""
    return pd.read_csv(file_path, sep='\t')

def create_moon_chunks(df: pd.DataFrame) -> List[MoonChunk]:
    """Create chunks organized by moon with combined related information."""
    moon_chunks = []
    
    # Group by moon name to combine related information
    for moon_name, group in df.groupby('Moon Name'):
        # Combine all content for this moon
        combined_content = []
        
        for _, row in group.iterrows():
            combined_content.append(f"{row['Document Title']}:\n{row['Document Content']}")
        
        # Create a comprehensive chunk for this moon
        content = f"# {moon_name}\n\n" + "\n\n".join(combined_content)
        
        # Create metadata
        metadata = {
            "moon_name": moon_name,
            "document_count": len(group),
            "source_urls": group['Source URL'].unique().tolist()
        }
        
        # Use the first source URL as primary source
        primary_source = group['Source URL'].iloc[0]
        
        moon_chunks.append(
            MoonChunk(
                moon_name=moon_name,
                content=content,
                metadata=metadata,
                source_url=primary_source
            )
        )
    
    return moon_chunks

def chunk_for_embedding(moon_chunks: List[MoonChunk], chunk_size: int = 500):
    """Further chunk the moon content if needed for optimal embedding size."""
    text_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "moon_name"),
        ],
        strip_headers=False
    )
    
    final_chunks = []
    
    for moon in moon_chunks:
        # Split if content is too large
        splits = text_splitter.split_text(moon.content)
        
        for split in splits:
            # Preserve the original metadata while adding the split content
            metadata = moon.metadata.copy()
            metadata.update(split.metadata)
            
            final_chunks.append({
                "text": split.page_content,
                "metadata": metadata,
                "source_url": moon.source_url
            })
    
    return final_chunks

def main():
    # Read and process the data
    df = read_moons_data('jupiter_moons.tsv')
    moon_chunks = create_moon_chunks(df)
    final_chunks = chunk_for_embedding(moon_chunks)
    
    # Print example chunk for verification
    print(f"Created {len(final_chunks)} chunks")
    print("\nExample chunk:")
    print(final_chunks[0])

if __name__ == "__main__":
    main()

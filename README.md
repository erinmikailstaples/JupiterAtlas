# Jupiter Moons Chatbot

An AI-powered chatbot that provides information about Jupiter's moons using LangChain, OpenAI, and Pinecone.

## Features
- Vector-based search using Pinecone
- OpenAI GPT-4 integration
- Comprehensive data about Jupiter's moons
- Interactive chat interface

## Prerequisites
- Python 3.8+
- OpenAI API key
- Pinecone API key
- Galileo API key (for evaluation)

## Installation
1. Clone the repository:

```git clone https://github.com/erinmikailstaples/JupiterAtlas.git```

navigate into the correct directory
```cd backend```

2. Create a virtual environment using uv:

```uv venv```

3. Activate the virtual environment:

```uv venv/bin/activate```

4. Install the dependencies:

```uv pip install -r requirements.txt```

5. Copy and rename .env.example to .env and add your OpenAI and Pinecone API keys.


## Configuration

1. Update your .env file with the respective secrets

2. The chatbot uses the following default settings:
   - Index Name: "jupitermoons-2"
   - Namespace: "moonvector"

## Usage

1. First, initialize the vector store:

```python vector_store.py```

2. Verify vedctors were created successfully by running:

```python review_vectors.py```

3. Run the chatbot:

```python chatbot.py```


## How It Works

1. **Data Processing**: The system reads Jupiter moon data from TSV (reference: jupiter_moons.tsv, startLine: 1, endLine: 157)

2. **Vector Store Creation**: Data is processed and stored in Pinecone (reference: vector_store.py, startLine: 12, endLine: 47)

3. **Chatbot Interaction**: The chatbot uses:
   - OpenAI embeddings for text understanding
   - Pinecone for efficient vector search
   - GPT-4 for generating responses

## Troubleshooting

1. If vectors aren't appearing, check:
   - Pinecone API key validity
   - Index name and namespace configuration
   - Vector store initialization logs

2. For embedding issues:
   - Verify OpenAI API key
   - Check embedding model availability
   - Review chunking configuration

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NASA for Jupiter moons data
- OpenAI for GPT-4 and embeddings
- Pinecone for vector storage
- LangChain for the chat framework

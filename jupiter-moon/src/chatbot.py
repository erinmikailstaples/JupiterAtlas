from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
import os
from langchain.globals import set_debug

# Load environment variables
load_dotenv()

set_debug(False)

def init_chatbot():
    """Initialize the chatbot with retrieval capabilities."""
    
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # Configuration
    INDEX_NAME = "jupitermoons-2"
    NAMESPACE = "moonvector"
    
    # Create vector store
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-ada-002"
    )
    
    docsearch = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        namespace=NAMESPACE
    )
    
    # Initialize retriever with more documents
    retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5  # Retrieve more documents
        }
    )
    
    # Initialize ChatOpenAI
    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model_name='gpt-4',
        temperature=0.0,
        callbacks=None
    )
    
    # Create the chain with more specific prompt
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    
    # Modify the prompt to be more specific about using the context
    combine_docs_chain = create_stuff_documents_chain(
        llm,
        retrieval_qa_chat_prompt,
        document_variable_name="context"
    )
    
    retrieval_chain = create_retrieval_chain(
        retriever,
        combine_docs_chain
    )
    
    return retrieval_chain

def chat_with_moons():
    """Interactive chat function about Jupiter's moons."""
    chain = init_chatbot()
    
    print("Chat with me about Jupiter's moons! (type 'quit' to exit)")
    print("="*50)
    
    while True:
        question = input("\nYour question: ")
        if question.lower() == 'quit':
            break
            
        response = chain.invoke({"input": question})
        print("\nAnswer:", response["answer"])

if __name__ == "__main__":
    chat_with_moons()

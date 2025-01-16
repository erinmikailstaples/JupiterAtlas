from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
import os
from langchain.globals import set_debug
from typing import Dict, Any, List
import logging
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from galileo_observe import ObserveWorkflows
import uuid
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@dataclass
class Message:
    role: str
    content: str
    metadata: Dict[str, Any] = None

class JupiterObserver:
    def __init__(self):
        self.observe_logger = ObserveWorkflows(project_name="JupiterAtlasObs")
        self.current_workflow = None
        self.thread_id = str(uuid.uuid4())
        self._initialized = False
    
    def init_workflow(self) -> bool:
        try:
            if not self._initialized:
                galileo_api_key = os.getenv("GALILEO_API_KEY")
                if not galileo_api_key:
                    logger.error("❌ Galileo API key not found")
                    return False
                
                self._initialized = True
                logger.info("✅ Galileo workflow initialized")
            return True
        except Exception as e:
            logger.error(f"❌ Error initializing Galileo workflow: {str(e)}")
            return False
    
    def process_interaction(self, question: str, context: List[str], 
                          response: Dict[str, Any], messages: List[Message]) -> None:
        try:
            self.current_workflow = self.observe_logger.add_workflow(
                input={"question": question},
                metadata={
                    "thread_id": self.thread_id,
                    "message_count": len(messages)
                }
            )
            
            if context:
                self.current_workflow.add_retriever(
                    input=question,
                    documents=[{
                        "content": str(doc),
                        "metadata": {"source": "jupiter_moons"}
                    } for doc in context]
                )
            
            self.current_workflow.add_llm(
                input={
                    "messages": [{"role": m.role, "content": m.content} 
                               for m in messages],
                    "question": question
                },
                output={"answer": response.get("answer", "")},
                model="gpt-4",
                metadata={
                    "env": "production",
                    "thread_id": self.thread_id
                }
            )
            
            self.current_workflow.conclude(
                output={
                    "final_answer": response.get("answer", ""),
                    "context_used": bool(context)
                }
            )
            
            self.observe_logger.upload_workflows()
            logger.info(f"✅ Workflow completed and uploaded for thread {self.thread_id}")
            
        except Exception as e:
            logger.error(f"❌ Error processing interaction: {str(e)}")

def init_chatbot(
    temperature: float = 0.0,
    model_name: str = 'gpt-4',
    retrieval_k: int = 5
) -> Any:
    """Initialize the chatbot with enhanced retrieval capabilities and error handling.
    
    Args:
        temperature: Controls randomness in responses (0.0 = deterministic)
        model_name: Name of the GPT model to use
        retrieval_k: Number of relevant documents to retrieve
        
    Returns:
        Chain: The configured retrieval chain
    """
    try:
        # Initialize Pinecone with error handling
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # Configuration
        INDEX_NAME = "jupitermoons-2"
        NAMESPACE = "moonvector"
        
        # Create vector store with enhanced embeddings
        embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="text-embedding-ada-002",
            timeout=60,
            max_retries=3
        )
        
        docsearch = PineconeVectorStore(
            index_name=INDEX_NAME,
            embedding=embeddings,
            namespace=NAMESPACE
        )
        
        # Enhanced retriever configuration
        retriever = docsearch.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": retrieval_k
            }
        )
        
        # Initialize ChatOpenAI with system message
        llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model_name=model_name,
            temperature=temperature,
            request_timeout=60,
            max_retries=3
        )
        
        # Define the system and human message templates
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a knowledgeable assistant specializing in Jupiter's moons. Use the provided context to answer questions accurately. If you're unsure, admit it rather than making assumptions."),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "Context: {context}\n\nQuestion: {input}"),
        ])
        
        # Create the chain with the updated prompt
        combine_docs_chain = create_stuff_documents_chain(
            llm,
            prompt,
            document_variable_name="context"
        )
        
        return create_retrieval_chain(retriever, combine_docs_chain)
        
    except Exception as e:
        logger.error(f"Error initializing chatbot: {str(e)}")
        raise

def chat_with_moons():
    """Interactive chat function about Jupiter's moons with enhanced error handling and user experience."""
    try:
        # Initialize components
        chain = init_chatbot()
        observer = JupiterObserver()
        galileo_enabled = observer.init_workflow()
        
        # Track conversation history
        messages = []
        
        if galileo_enabled:
            print("\n✅ Galileo observation enabled")
        else:
            print("\n❌ Galileo observation disabled - check your API key")
        
        print("\n🌔 Welcome to the Jupiter Moons Chatbot! 🌔")
        print("Ask me anything about Jupiter's moons (type 'quit' to exit)")
        print("="*50)
        
        while True:
            try:
                question = input("\nYour question: ").strip()
                if question.lower() == 'quit':
                    print("\nThank you for chatting about Jupiter's moons! Goodbye! 👋")
                    break
                
                if not question:
                    print("Please enter a valid question!")
                    continue
                
                # Add user message to history
                messages.append(Message(role="user", content=question))
                
                # Get response
                response = chain.invoke({
                    "input": question,
                    "chat_history": []
                })
                
                # Add assistant message to history
                messages.append(Message(
                    role="assistant", 
                    content=response["answer"],
                    metadata={"context_used": bool(response.get("context"))}
                ))
                
                # Log to Galileo if enabled
                if galileo_enabled:
                    observer.process_interaction(
                        question=question,
                        context=response.get("context", []),
                        response=response,
                        messages=messages
                    )
                
                print("\nAnswer:", response["answer"])
                
            except KeyboardInterrupt:
                print("\nChat session interrupted. Goodbye! 👋")
                break
            except Exception as e:
                logger.error(f"Error processing question: {str(e)}")
                print("\nI apologize, but I encountered an error processing your question. Please try again.")
                
    except Exception as e:
        logger.error(f"Fatal error in chat session: {str(e)}")
        print("\nI apologize, but I encountered a serious error and need to shut down. Please restart the application.")

if __name__ == "__main__":
    chat_with_moons()

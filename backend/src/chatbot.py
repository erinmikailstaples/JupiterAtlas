from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
import os
from langchain.globals import set_debug
from typing import Dict, Any, List, Optional
import logging
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, BaseMessage
from galileo_observe import ObserveWorkflows
import uuid
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class Message(BaseMessage):
    role: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

class JupiterObserver:
    _instance = None
    _workflow = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(JupiterObserver, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            # Set Galileo console URL only
            os.environ["GALILEO_CONSOLE_URL"] = "https://console.acme.rungalileo.io"
            
            self.observe_logger = ObserveWorkflows(project_name="JupiterAtlasObs")
            self.current_workflow = None
            self.thread_id = str(uuid.uuid4())
            self.initialized = False
    
    def init_workflow(self) -> bool:
        try:
            if not self.initialized:
                galileo_api_key = os.getenv("GALILEO_API_KEY")
                if not galileo_api_key:
                    logger.error("❌ Galileo API key not found")
                    return False
                
                self.initialized = True
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
                    "message_count": str(len(messages))
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
            
            # Convert messages list to a formatted string
            messages_str = "\n".join([
                f"{msg.role}: {msg.content}" 
                for msg in messages
            ])
            
            self.current_workflow.add_llm(
                input=question,
                output=response.get("answer", ""),
                model="gpt-4",
                metadata={
                    "env": "production",
                    "thread_id": self.thread_id,
                    "messages": messages_str  # Now a string instead of a list
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

def init_chatbot():
    """Initialize the chatbot with better error handling"""
    try:
        # Check for required environment variables
        required_vars = ['OPENAI_API_KEY', 'PINECONE_API_KEY', 'PINECONE_ENVIRONMENT']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # Initialize Pinecone
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        
        # Initialize vector store
        vector_store = PineconeVectorStore(
            index_name="jupitermoons-2",
            namespace="moonvector",
            embedding=OpenAIEmbeddings()
        )
        
        # Create retriever
        retriever = vector_store.as_retriever()
        
        # Initialize LLM
        llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0.7
        )
        
        # Create the chain
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert on Jupiter's moons. Provide accurate, scientific information."),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "Context: {context}\n\nQuestion: {input}"),
        ])
        
        combine_docs_chain = create_stuff_documents_chain(
            llm,
            prompt,
            document_variable_name="context"
        )
        
        return create_retrieval_chain(retriever, combine_docs_chain)
    except Exception as e:
        logger.error(f"Error initializing chatbot: {str(e)}")
        raise RuntimeError(f"Failed to initialize chatbot: {str(e)}")

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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging
from .chatbot import init_chatbot, Message, JupiterObserver

app = FastAPI(title="Jupiter Moons API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chatbot components
chain = init_chatbot()
observer = JupiterObserver()
galileo_enabled = observer.init_workflow()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/")
async def root():
    return {"status": "healthy", "message": "Jupiter Moons API is running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "galileo_enabled": galileo_enabled,
        "chain_initialized": chain is not None
    }

class ChatRequest(BaseModel):
    question: str
    messages: List[Message]

class ChatResponse(BaseModel):
    answer: str
    context: Optional[List[str]] = None

@app.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        if not chain:
            raise HTTPException(
                status_code=503,
                detail="Chatbot service unavailable. Please try again later."
            )
            
        # Log incoming request
        logger.info(f"Received chat request: {request.question}")
            
        response = chain.invoke({
            "input": request.question,
            "chat_history": []
        })
        
        if not response or "answer" not in response:
            logger.error(f"Invalid response from chain: {response}")
            raise HTTPException(
                status_code=500,
                detail="Invalid response from chatbot"
            )
        
        return ChatResponse(
            answer=response["answer"],
            context=response.get("context", [])
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Server error: {str(e)}"
        )

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores.pgvector import PGVector
from dotenv import load_dotenv
from sqlalchemy import create_engine
import google.generativeai as genai
from langchain.embeddings.base import Embeddings

# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

class GoogleGenerativeEmbeddings(Embeddings):
    def __init__(self, model_name="models/text-embedding-004", dimension=768):
        self.model_name = model_name
        self.dimension = dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                output_dimensionality=self.dimension
            )
            embeddings.append(result["embedding"])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        result = genai.embed_content(
            model=self.model_name,
            content=text,
            output_dimensionality=self.dimension
        )
        return result["embedding"]

app = FastAPI()

# Database configuration
CONNECTION_STRING = f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('DB_NAME')}"
COLLECTION_NAME = "document_embeddings"

# Initialize Google Generative embeddings
embeddings = GoogleGenerativeEmbeddings()

# Initialize PGVector
vector_store = PGVector(
    connection_string=CONNECTION_STRING,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
)

# Create database engine
engine = create_engine(CONNECTION_STRING)

# Pydantic models for request validation
class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[tuple]] = []

class EmbeddingRequest(BaseModel):
    texts: List[str]
    metadata: Optional[List[dict]] = None

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Initialize ChatOpenAI
        llm = ChatOpenAI(temperature=0)
        
        # Create conversation chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(
                search_kwargs={"k": 3}  # Return top 3 most relevant documents
            ),
            return_source_documents=True
        )
        
        # Get response from the chain
        result = qa_chain({"question": request.question, "chat_history": request.chat_history})
        
        return {
            "answer": result["answer"],
            "source_documents": [doc.page_content for doc in result["source_documents"]]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embedding")
async def create_embedding(request: EmbeddingRequest):
    try:
        # Prepare metadata if not provided
        if request.metadata is None:
            request.metadata = [{"source": f"doc_{i}"} for i in range(len(request.texts))]
        
        # Add documents to PGVector
        vector_store.add_texts(
            texts=request.texts,
            metadatas=request.metadata
        )
        
        return {"message": "Embeddings created and stored successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/clear-embeddings")
async def clear_embeddings():
    try:
        # Drop the collection and recreate it
        vector_store.delete_collection()
        vector_store.create_collection()
        return {"message": "Embeddings cleared successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

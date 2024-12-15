import os
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Kiểm tra OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")

# Initial system instructions prompt for context (cập nhật để hướng dẫn AI tìm thông tin về người trong tài liệu)
initial_prompt = initial_prompt = """
You are a friendly AI assistant. Your goal is to help users by answering questions accurately and clearly. 
When responding to questions such as "Hello", "Hi", "How are you?", or similar social greetings, 
respond with a friendly, polite, and human-like tone. Your answers should be natural, empathetic, 
and engaging, just like how a person would respond in a casual conversation.

For example:
- "Hello, how can I assist you today?"
- "Hi there! I'm doing great, thanks for asking! How about you?"
- "I'm doing well, thanks for checking in! How can I help you today?"

Be sure to also adjust your tone based on the context of the conversation.
"""

# Load all PDF files from the specified folder
try:
    loader = DirectoryLoader("data", glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
except Exception as e:
    raise Exception(f"Error loading PDF files: {e}")

# Clean text by removing unnecessary characters
def clean_text(text):
    text = re.sub(r'\n+', ' ', text)  # Remove extra newlines
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'\d+\s*/\s*\d+', ' ', text)  # Remove page numbers (e.g., 2/12)
    text = re.sub(r'[^a-zA-Z0-9,.?!\\s]', '', text)  # Remove special characters but keep essential punctuation
    return text.strip().lower()  # Normalize to lowercase and strip leading/trailing spaces

# Clean text for all documents
documents = [Document(page_content=clean_text(doc.page_content), metadata=doc.metadata) for doc in documents]

# Split documents into smaller chunks to improve vector storage
try:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = splitter.split_documents(documents)
except Exception as e:
    raise Exception(f"Error splitting documents: {e}")

# Remove empty or very short documents
documents = [doc for doc in documents if len(doc.page_content.strip()) > 50]

# Generate embeddings from document content
try:
    embeddings = OpenAIEmbeddings()
except Exception as e:
    raise Exception(f"Error initializing OpenAI embeddings: {e}")

# Store document embeddings in a FAISS vector store
try:
    vectorstore = FAISS.from_documents(documents, embeddings)
except Exception as e:
    raise Exception(f"Error creating FAISS vector store: {e}")

# Create a retriever to search through the vector store
try:
    retriever = vectorstore.as_retriever()
except Exception as e:
    raise Exception(f"Error creating retriever: {e}")

# Create a QA chain that connects the retriever to the OpenAI LLM
try:
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0.7),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True  # Ensure context-aware responses
    )
except Exception as e:
    raise Exception(f"Error creating QA chain: {e}")

# Set up FastAPI application
app = FastAPI()

# Define a Pydantic model for the API request
class QueryRequest(BaseModel):
    query: str

# Endpoint to handle user questions and return answers
@app.post("/ask")
async def ask_question(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        # Pass the query and initial prompt for context to the QA chain
        response = qa_chain.invoke({"query": request.query, "context": initial_prompt}) 
        return {"question": request.query, "answer": response['result'], "source_documents": response['source_documents']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the query: {e}")

# Health check endpoint to ensure the API is running properly
@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}

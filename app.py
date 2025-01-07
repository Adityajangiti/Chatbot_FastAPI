from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
from transformers import pipeline
import re
from databases import Database


# Initialize FastAPI app
app = FastAPI()

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:8501/"],  # React frontend
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database Configuration
DATABASE_URL = "postgresql://username:password@localhost/dbname"
database = Database(DATABASE_URL)

# Global Knowledge Base
knowledge_base = ""

# Load QA Pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

@app.lifespan("startup")
async def startup():
    await database.connect()

@app.lifespan("shutdown")
async def shutdown():
    await database.disconnect()

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile):
    """Upload a PDF file, extract text, and set it as the base knowledge."""
    global knowledge_base

    try:
        # Save the uploaded file
        contents = await file.read()
        with open("temp.pdf", "wb") as f:
            f.write(contents)

        # Extract text from the PDF
        reader = PdfReader("temp.pdf")
        extracted_text = "".join(page.extract_text() for page in reader.pages)

        # Update the global knowledge base
        knowledge_base = extracted_text
        return {"message": "Knowledge base updated successfully!"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/chat/")
async def chat(question: str = Form(...)):
    """Answer a question based on the current knowledge base while ensuring at least 50% of the words from the
question are present in the PDF content.
"""
    if not knowledge_base:
        return {"error": "Knowledge base is empty. Please upload a PDF first!"}
    try:
        # Tokenize question and knowledge base
        question_words = re.findall(r'\w+',question.lower())
        knowledge_base_words = set(re.findall(r'\w+',knowledge_base.lower()))

        # Calculate matching percentage
        matched_words = sum(1 for word in question_words if word in knowledge_base_words)
        if matched_words / len(question_words) < 0.5:
            return {"answer": "Sorry, I didn't understand. Do you want to connect with a live agent?"}
        
        # Generate answer
        result = qa_pipeline(question=question,context=knowledge_base)
        answer = result.get("answer", "").strip()
        if not answer:
            answer = "Sorry,I didn't understand. Do you want to connect with a live agent?"
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}
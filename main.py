from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
import uuid
import os
import PyPDF2
import io
from openai import OpenAI  # Updated import
import json
from typing import Optional, Dict, Any
import asyncio
from pydantic import BaseModel

# Database setup
# SQLALCHEMY_DATABASE_URL = "postgresql://postgres:password@localhost/resume_screener"
# For development, you can also use SQLite:
SQLALCHEMY_DATABASE_URL = "sqlite:///./resume_screener.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class ResumeRecord(Base):
    __tablename__ = "resumes"
    
    id = Column(String, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    content = Column(Text, nullable=True)
    score = Column(Float, nullable=True)
    analysis = Column(JSON, nullable=True)
    status = Column(String, default="processing")  # processing, completed, error
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic models
class ResumeResponse(BaseModel):
    id: str
    filename: str
    score: Optional[float] = None
    analysis: Optional[Dict[str, Any]] = None
    status: str
    created_at: datetime

class AnalysisResult(BaseModel):
    name: str
    email: str
    phone: str
    location: str
    experience_years: int
    skills: list[str]
    education: list[str]
    strengths: list[str]
    weaknesses: list[str]
    recommendation: str

# FastAPI app
app = FastAPI(title="Resume Screener API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "https://i-amour.github.io"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# OpenAI client setup - Updated to use new OpenAI library

openai_client = None
openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key:
    openai_client = OpenAI(api_key=openai_api_key)
    print("OpenAI client initialized successfully")
else:
    print("Warning: OPENAI_API_KEY not found. Using mock data.")

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Helper functions
def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error extracting PDF text: {str(e)}")

def extract_text_from_docx(file_content: bytes) -> str:
    """Extract text from DOCX file (basic implementation)"""
    # For a full implementation, you'd use python-docx
    # For now, we'll return a placeholder
    return "DOCX content extraction not implemented in this demo"

async def analyze_resume_with_ai(resume_text: str) -> tuple[float, AnalysisResult]:
    """Analyze resume using OpenAI API"""
    try:
        # Check if OpenAI client is available
        if not openai_client:
            print("OpenAI client not available, using mock data")
            return await mock_ai_analysis(resume_text)
        
        print(f"Analyzing resume with OpenAI. Text length: {len(resume_text)}")
        
        prompt = f"""
        Analyze the following resume and provide a detailed assessment. 
        Return your response as a JSON object with the following structure:
        {{
            "score": (integer from 0-100),
            "name": "candidate's name",
            "email": "candidate's email",
            "phone": "candidate's phone",
            "location": "candidate's location",
            "experience_years": (integer),
            "skills": ["list of skills"],
            "education": ["list of education entries"],
            "strengths": ["list of 3-5 strengths"],
            "weaknesses": ["list of 3-5 areas for improvement"],
            "recommendation": "detailed hiring recommendation paragraph"
        }}

        Resume text:
        {resume_text[:4000]}
        """
        
        # Updated API call for new OpenAI library
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert HR recruiter and resume analyst. Analyze resumes objectively and provide constructive feedback. Always return valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.3
        )
        
        # Extract the response content
        content = response.choices[0].message.content
        print(f"OpenAI response: {content}")
        
        # Parse JSON response
        try:
            result = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Raw content: {content}")
            return await mock_ai_analysis(resume_text)
        
        score = result.get("score", 0)
        
        analysis = AnalysisResult(
            name=result.get("name", "Not provided"),
            email=result.get("email", "Not provided"),
            phone=result.get("phone", "Not provided"),
            location=result.get("location", "Not provided"),
            experience_years=result.get("experience_years", 0),
            skills=result.get("skills", []),
            education=result.get("education", []),
            strengths=result.get("strengths", []),
            weaknesses=result.get("weaknesses", []),
            recommendation=result.get("recommendation", "No recommendation available")
        )
        
        print(f"Analysis completed with score: {score}")
        return score, analysis
        
    except Exception as e:
        print(f"AI analysis error: {e}")
        return await mock_ai_analysis(resume_text)

async def mock_ai_analysis(resume_text: str) -> tuple[float, AnalysisResult]:
    """Mock AI analysis for demonstration when OpenAI API is not available"""
    # Simulate processing time
    await asyncio.sleep(2)
    
    print("Using mock analysis - OpenAI API not configured")
    
    # Try to extract some basic info from the resume text
    lines = resume_text.lower().split('\n')
    
    # Simple email extraction
    email = "Not provided"
    for line in lines:
        if '@' in line and '.' in line:
            words = line.split()
            for word in words:
                if '@' in word and '.' in word:
                    email = word.strip()
                    break
            if email != "Not provided":
                break
    
    # Simple skill detection
    common_skills = ['python', 'javascript', 'java', 'react', 'sql', 'aws', 'docker', 'git', 'html', 'css']
    found_skills = []
    resume_lower = resume_text.lower()
    for skill in common_skills:
        if skill in resume_lower:
            found_skills.append(skill.title())
    
    # Generate mock analysis with some actual resume content
    score = min(85, max(60, 70 + len(found_skills) * 2))  # Score based on skills found
    
    analysis = AnalysisResult(
        name="Information extracted from resume",
        email=email,
        phone="Check resume for phone number",
        location="Location not clearly identified",
        experience_years=3,
        skills=found_skills if found_skills else ["Skills not clearly identified"],
        education=["Education details in uploaded resume"],
        strengths=[
            "Resume uploaded successfully",
            f"Document contains {len(resume_text.split())} words of content",
            "Skills and experience mentioned in document" if found_skills else "Professional document format"
        ],
        weaknesses=[
            "OpenAI API not configured for detailed analysis",
            "Set OPENAI_API_KEY environment variable for AI-powered analysis",
            "Mock analysis provided - actual analysis requires API key"
        ],
        recommendation=f"This resume analysis is using mock data because OpenAI API is not configured. The uploaded resume contains {len(resume_text.split())} words. To get real AI-powered analysis, please set the OPENAI_API_KEY environment variable with a valid OpenAI API key."
    )
    
    return score, analysis

async def process_resume(resume_id: str, db: Session):
    """Background task to process resume"""
    resume = db.query(ResumeRecord).filter(ResumeRecord.id == resume_id).first()
    if not resume:
        return
    
    try:
        print(f"Processing resume {resume_id}: {resume.filename}")
        
        # Analyze with AI
        score, analysis = await analyze_resume_with_ai(resume.content)
        
        # Update database
        resume.score = score
        resume.analysis = analysis.dict()
        resume.status = "completed"
        resume.updated_at = datetime.utcnow()
        
        db.commit()
        print(f"Resume {resume_id} processed successfully")
        
    except Exception as e:
        print(f"Error processing resume {resume_id}: {e}")
        resume.status = "error"
        resume.updated_at = datetime.utcnow()
        db.commit()

# API Routes
@app.get("/")
async def root():
    return {
        "message": "Resume Screener API",
        "openai_configured": openai_client is not None,
        "status": "ready"
    }

@app.post("/upload-resume")
async def upload_resume(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload and process a resume file"""
    
    # Validate file type
    allowed_types = ["application/pdf", "application/msword", 
                     "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
    
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Only PDF, DOC, and DOCX files are supported")
    
    # Read file content
    file_content = await file.read()
    
    # Extract text based on file type
    if file.content_type == "application/pdf":
        text_content = extract_text_from_pdf(file_content)
    elif file.content_type in ["application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        text_content = extract_text_from_docx(file_content)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    print(f"Extracted {len(text_content)} characters from {file.filename}")
    
    # Create database record
    resume_id = str(uuid.uuid4())
    resume = ResumeRecord(
        id=resume_id,
        filename=file.filename,
        content=text_content,
        status="processing"
    )
    
    db.add(resume)
    db.commit()
    
    # Start background processing
    background_tasks.add_task(process_resume, resume_id, db)
    
    return {
        "id": resume_id, 
        "message": "Resume uploaded successfully",
        "openai_available": openai_client is not None
    }

@app.get("/resume/{resume_id}")
async def get_resume(resume_id: str, db: Session = Depends(get_db)):
    """Get resume analysis results"""
    resume = db.query(ResumeRecord).filter(ResumeRecord.id == resume_id).first()
    
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")
    
    return ResumeResponse(
        id=resume.id,
        filename=resume.filename,
        score=resume.score,
        analysis=resume.analysis,
        status=resume.status,
        created_at=resume.created_at
    )

@app.get("/resumes")
async def list_resumes(db: Session = Depends(get_db)):
    """List all resumes"""
    resumes = db.query(ResumeRecord).order_by(ResumeRecord.created_at.desc()).all()
    
    return [
        ResumeResponse(
            id=resume.id,
            filename=resume.filename,
            score=resume.score,
            analysis=resume.analysis,
            status=resume.status,
            created_at=resume.created_at
        )
        for resume in resumes
    ]

@app.delete("/resume/{resume_id}")
async def delete_resume(resume_id: str, db: Session = Depends(get_db)):
    """Delete a resume"""
    resume = db.query(ResumeRecord).filter(ResumeRecord.id == resume_id).first()
    
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")
    
    db.delete(resume)
    db.commit()
    
    return {"message": "Resume deleted successfully"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "timestamp": datetime.utcnow(),
        "openai_configured": openai_client is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

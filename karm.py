import streamlit as st
import os
import docx
from PyPDF2 import PdfReader
from langchain_together import Together
from langchain_core.messages import HumanMessage, AIMessage
from typing import Dict
import logging

# Set your API key
TOGETHER_API_KEY = '09018cc36965afd53da91adb6117f2084bb34f80f209d7d357bbad8c08bc8b26'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text(file) -> str:
    """Extracts text from PDF and DOCX files."""
    try:
        file_extension = file.name.split(".")[-1].lower()
        
        if file_extension == "pdf":
            return " ".join(page.extract_text() for page in PdfReader(file).pages if page.extract_text())
        elif file_extension in ["doc", "docx"]:
            return " ".join(p.text for p in docx.Document(file).paragraphs)
            
    except Exception as e:
        st.error(f"Error processing {file.name}: {str(e)}")
        return ""

def analyze_job_seeker_resume(resume_text: str, job_description: str) -> str:
    """Analyzes resume for job seekers with improvement suggestions."""
    prompt = f"""
    You are KARM, an expert AI assistant for resume analysis. Analyze this resume against the job description 
    and provide specific, actionable suggestions for improvement. Focus on:
    
    1. Skills alignment and gaps
    2. Experience relevance
    3. Achievement descriptions
    4. Keywords optimization
    5. Specific improvements needed
    
    Resume: {resume_text}
    Job Description: {job_description}
    
    Provide a clear, conversational response with specific suggestions.
    """
    
    return Together(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature=0.7,
        max_tokens=1024,
        api_key=TOGETHER_API_KEY
    ).invoke(prompt)

def analyze_resumes_for_recruiter(job_description: str, resumes: Dict[str, str]) -> str:
    """Analyzes multiple resumes for recruiters and ranks candidates."""
    prompt = f"""
    You are KARM, an expert AI recruiter. Analyze these resumes against the job description and identify the best candidates.
    Create a ranked list of candidates with only the following information:
    
    1. Full Name
    2. Contact Number
    3. Email Address
    4. Match percentage with the job description
    
    Sort candidates from highest to lowest match percentage.
    Present the information in a clear, tabulated format.
    
    Job Description: {job_description}
    
    Resumes:
    {resumes}
    """
    
    return Together(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature=0.7,
        max_tokens=1024,
        api_key=TOGETHER_API_KEY
    ).invoke(prompt)

def main():
    st.set_page_config(page_title="KARM", layout="wide")
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed_resumes" not in st.session_state:
        st.session_state.processed_resumes = {}
    
    # Sidebar
    with st.sidebar:
        st.title("KARM")
        st.subheader("AI Resume Assistant")
        
        # User type selection
        user_type = st.selectbox(
            "Select User Type",
            ["Job Seeker", "Recruiter"]
        )
        
        # File upload - handle multiple files for recruiter, single file for job seeker
        if user_type == "Recruiter":
            uploaded_files = st.file_uploader(
                "Upload Resumes",
                accept_multiple_files=True,
                type=["pdf", "doc", "docx"]
            )
            
            if uploaded_files:
                # Process uploaded files
                for file in uploaded_files:
                    text = extract_text(file)
                    if text:
                        st.session_state.processed_resumes[file.name] = text
                
                st.success(f"Successfully processed {len(st.session_state.processed_resumes)} resume(s)")
        else:
            # Single file upload for job seeker
            uploaded_file = st.file_uploader(
                "Upload Resume",
                accept_multiple_files=False,
                type=["pdf", "doc", "docx"]
            )
            
            if uploaded_file:
                text = extract_text(uploaded_file)
                if text:
                    st.session_state.processed_resumes = {uploaded_file.name: text}
                    st.success("Successfully processed your resume")
    
    # Main chat interface
    st.header("Knowledge-Enhanced Resume Management")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
            st.write(message.content)
    
    # Chat input
    chat_prompt = "Paste the Job Description to start with" if user_type == "Recruiter" else "Ask questions about your resume"
    user_input = st.chat_input(chat_prompt)
    
    if user_input and st.session_state.processed_resumes:
        # Add user message to chat
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        
        try:
            # Process based on user type
            if user_type == "Job Seeker":
                response = analyze_job_seeker_resume(
                    list(st.session_state.processed_resumes.values())[0],
                    user_input
                )
            else:  # Recruiter
                response = analyze_resumes_for_recruiter(
                    user_input,
                    st.session_state.processed_resumes
                )
                
            # Add AI response to chat
            st.session_state.chat_history.append(AIMessage(content=response))
            
            # Rerun to update chat display
            st.rerun()
            
        except Exception as e:
            st.error(f"Error processing request: {str(e)}")
    
    elif user_input:
        st.warning("Please upload resume(s) first")

if __name__ == "__main__":
    main()

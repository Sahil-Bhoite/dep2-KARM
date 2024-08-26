import streamlit as st
import os
import io
from PyPDF2 import PdfReader
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain_community.llms import GooglePalm
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
import logging
import spacy
from config import GOOGLE_API_KEY
import re
from typing import List, Dict, Any
from langchain_community.chat_models import ChatOllama
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from io import BytesIO
import textwrap

# Set up environment and configurations
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
nlp = spacy.load("en_core_web_sm")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text(file) -> str:
    text = ""
    file_extension = file.name.split(".")[-1].lower()
    try:
        if file_extension == "pdf":
            text = extract_text_from_pdf(file)
        elif file_extension in ["doc", "docx"]:
            text = extract_text_from_docx(file)
        elif file_extension in ["txt", "rtf"]:
            text = file.read().decode("utf-8")
        else:
            st.error(f"Unsupported file type: {file_extension}")
    except Exception as e:
        handle_file_processing_error(file_extension.upper(), e)
    return text

def extract_text_from_pdf(pdf_file) -> str:
    text = ""
    try:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    except Exception as e:
        handle_file_processing_error("PDF", e)
    return text

def extract_text_from_docx(docx_file) -> str:
    text = ""
    try:
        doc = docx.Document(docx_file)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        handle_file_processing_error("DOCX", e)
    return text

def handle_file_processing_error(file_type: str, error: Exception):
    st.error(f"Error processing {file_type} file: {error}")
    logger.exception(f"Error processing {file_type} file", exc_info=True)

def handle_model_interaction_error(error: Exception):
    st.error(f"Error interacting with AI model: {error}")
    logger.exception("Error interacting with AI model", exc_info=True)

def validate_user_input(user_input) -> bool:
    if not user_input:
        st.warning("Please provide valid input.")
        return False
    return True

def log_event(event: str):
    logger.info(event)

def parse_resume(text: str) -> Dict[str, Any]:
    doc = nlp(text)
    
    parsed_data = {
        "name": "",
        "email": "",
        "phone": "",
        "education": [],
        "experience": [],
        "skills": []
    }
    
    for ent in doc.ents:
        if ent.label_ == "PERSON" and not parsed_data["name"]:
            parsed_data["name"] = ent.text
        elif ent.label_ == "EMAIL":
            parsed_data["email"] = ent.text
        elif ent.label_ in ["PHONE_NUMBER", "CARDINAL"] and not parsed_data["phone"]:
            parsed_data["phone"] = ent.text
    
    education_keywords = ["education", "degree", "university", "college"]
    for sent in doc.sents:
        if any(keyword in sent.text.lower() for keyword in education_keywords):
            parsed_data["education"].append(sent.text.strip())
    
    experience_keywords = ["experience", "work", "job", "position"]
    for sent in doc.sents:
        if any(keyword in sent.text.lower() for keyword in experience_keywords):
            parsed_data["experience"].append(sent.text.strip())
    
    skills_section = re.search(r"skills:?(.*?)(^|\n\n)", text, re.IGNORECASE | re.DOTALL)
    if skills_section:
        skills = skills_section.group(1).strip().split(',')
        parsed_data["skills"] = [skill.strip() for skill in skills]
    
    return parsed_data

def review_and_score_resume(parsed_data: Dict[str, Any]) -> Dict[str, Any]:
    llm = GooglePalm()
    
    prompt = f"""
    Review the following resume data and provide a score out of 100 based on ATS compliance, formatting, and keyword usage.
    Also provide a detailed breakdown with improvement suggestions for each section.

    Resume data:
    {parsed_data}

    Your response should be in the following format:
    Score: [score]
    Overall feedback: [overall feedback]
    
    Section-wise breakdown and suggestions:
    [section name]:
    - [feedback point 1]
    - [feedback point 2]
    ...

    Improvement suggestions:
    1. [suggestion 1]
    2. [suggestion 2]
    ...
    """

    response = llm(prompt)
    
    # Extract score from the response
    score_match = re.search(r"Score: (\d+)", response)
    score = int(score_match.group(1)) if score_match else 0

    return {
        "score": score,
        "feedback": response
    }

def get_text_chunks(text: str) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)
    chunks = text_splitter.split_text(text)
    return chunks 

def get_vector_store(text_chunks: List[str]):
    embeddings = GooglePalmEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain(vector_store, use_online_model):
    if use_online_model:
        llm = GooglePalm()
    else:
        llm = ChatOllama(model="phi3")
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    if vector_store:
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            retriever=vector_store.as_retriever(), 
            memory=memory
        )
        return conversation_chain
    else:
        st.error("Vector store is not available.")
        return None

def generate_pdf_resume(data: Dict[str, Any]) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Add custom font
    pdfmetrics.registerFont(TTFont('Arial', 'Arial.ttf'))

    # Add name
    c.setFont("Arial", 16)
    c.drawString(100, height - 50, data['name'])

    # Add contact info
    c.setFont("Arial", 10)
    c.drawString(100, height - 70, f"Email: {data['email']} | Phone: {data['phone']}")

    # Add sections
    y_position = height - 100
    sections = ['education', 'experience', 'skills']
    for section in sections:
        c.setFont("Arial", 14)
        c.drawString(100, y_position, section.capitalize())
        y_position -= 20
        c.setFont("Arial", 10)
        if isinstance(data[section], list):
            for item in data[section]:
                wrapped_text = textwrap.wrap(item, width=70)
                for wrap in wrapped_text:
                    c.drawString(120, y_position, wrap)
                    y_position -= 15
                y_position -= 5
        else:
            wrapped_text = textwrap.wrap(data[section], width=70)
            for wrap in wrapped_text:
                c.drawString(120, y_position, wrap)
                y_position -= 15
        y_position -= 10

    c.save()
    buffer.seek(0)
    return buffer.getvalue()

def main():
    st.set_page_config(page_title="KARM", layout="wide")
    st.header("Knowledge-Enhanced Assistance for Resume Management 📈")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        st.session_state.chat_history = []
        st.session_state.files_uploaded = False
        st.session_state.parsed_resume_data = None
        st.session_state.resume_score = None
        st.session_state.resume_feedback = None
    
    with st.sidebar:
        st.title("Upload and Review")
        
        use_online_model = st.toggle("Use Online Model", value=True)
        
        st.subheader("Upload your Resume")
        resume_file = st.file_uploader("Upload your Resume and Click on the Review Button", type=["pdf", "docx", "txt", "rtf"])
        
        if st.button("Review Resume"):
            if validate_user_input(resume_file):
                with st.spinner("Processing and reviewing your Resume..."):
                    text = extract_text(resume_file)
                    parsed_data = parse_resume(text)
                    st.session_state.parsed_resume_data = parsed_data
                    
                    review_result = review_and_score_resume(parsed_data)
                    st.session_state.resume_score = review_result['score']
                    st.session_state.resume_feedback = review_result['feedback']
                    
                    text_chunks = get_text_chunks(text)
                    vector_store = get_vector_store(text_chunks)
                    st.session_state.conversation = get_conversational_chain(vector_store, use_online_model)
                    
                    if st.session_state.conversation:
                        st.session_state.files_uploaded = True
                        st.success("Resume processed and reviewed successfully!")
                        
                        # Add initial feedback to chat history
                        st.session_state.chat_history.append(AIMessage(content=st.session_state.resume_feedback))
            else:
                st.warning("Please upload a resume file.")
        
        if st.session_state.parsed_resume_data:
            st.download_button(
                label="Download ATS-friendly Resume (PDF)",
                data=generate_pdf_resume(st.session_state.parsed_resume_data),
                file_name="ATS_friendly_resume.pdf",
                mime="application/pdf"
            )
    
    if st.session_state.files_uploaded and st.session_state.parsed_resume_data:
        st.subheader("Resume Review and Refinement")
        st.write(f"Current ATS-friendliness Score: {st.session_state.resume_score}/100")
        st.write("Chat with the AI to improve your resume. Ask for specific improvements or general advice.")
        
        # Display chat history
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                with st.chat_message("User"):
                    st.write(message.content)
            elif isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
        
        # Check if chat_input is available, otherwise use text_input
        if hasattr(st, 'chat_input'):
            user_input = st.chat_input("How would you like to improve your resume?")
        else:
            user_input = st.text_input("How would you like to improve your resume?")
        
        if user_input:
            try:
                response = st.session_state.conversation({'question': user_input})
                st.session_state.chat_history.append(HumanMessage(content=user_input))
                st.session_state.chat_history.append(AIMessage(content=response['answer']))
                
                # Rerun to update the chat display
                st.experimental_rerun()
            except Exception as e:
                handle_model_interaction_error(e)

if __name__ == "__main__":
    main()

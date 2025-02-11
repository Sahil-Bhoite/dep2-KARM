import streamlit as st
import os
import docx
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_together import Together
from langchain_core.messages import HumanMessage, AIMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Structure for storing resume analysis results"""
    skills_match: Dict[str, float]
    experience_analysis: Dict[str, str]
    improvement_suggestions: List[str]
    overall_score: float
    detailed_feedback: Dict[str, str]

class DocumentProcessor:
    """Handles document extraction and processing"""
    
    @staticmethod
    def extract_text(file) -> str:
        """Extract text from PDF and DOCX files"""
        try:
            file_extension = file.name.split(".")[-1].lower()
            
            if file_extension == "pdf":
                pdf_reader = PdfReader(file)
                text = " ".join(
                    page.extract_text() 
                    for page in pdf_reader.pages 
                    if page.extract_text()
                )
            elif file_extension in ["doc", "docx"]:
                doc = docx.Document(file)
                text = " ".join(paragraph.text for paragraph in doc.paragraphs)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error processing file {file.name}: {str(e)}")
            raise

    @staticmethod
    def get_text_chunks(text: str, chunk_size: int = 1024, chunk_overlap: int = 256) -> List[str]:
        """Split text into chunks for processing"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return splitter.split_text(text)

class ResumeAnalyzer:
    """Core resume analysis functionality"""
    
    def __init__(self, api_key: str):
        self.llm = Together(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            temperature=0.7,
            max_tokens=1024
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
    def _create_analysis_prompt(self, resume_text: str, job_description: str) -> str:
        """Create detailed prompt for resume analysis"""
        return f"""
        You are KARM, an expert AI assistant specializing in resume analysis and career coaching.
        Analyze the following resume against the job description and provide detailed feedback.
        
        Format your response as a JSON object with the following structure:
        {{
            "skills_match": {{
                "matching_skills": [list of matching skills],
                "missing_skills": [list of important missing skills],
                "skill_relevance_score": float (0-100)
            }},
            "experience_analysis": {{
                "relevant_experience": string (detailed analysis),
                "achievement_impact": string (analysis of quantified achievements),
                "improvement_areas": [list of specific improvements]
            }},
            "ats_optimization": {{
                "keyword_suggestions": [list of keywords to add],
                "format_improvements": [list of formatting suggestions],
                "section_organization": string (organization feedback)
            }},
            "overall_assessment": {{
                "score": float (0-100),
                "key_strengths": [list of strengths],
                "priority_improvements": [list of priority improvements],
                "final_recommendation": string (clear recommendation)
            }}
        }}

        Resume:
        {resume_text}

        Job Description:
        {job_description}
        """

    def analyze_resume(self, resume_text: str, job_description: str) -> AnalysisResult:
        """Perform comprehensive resume analysis"""
        try:
            prompt = self._create_analysis_prompt(resume_text, job_description)
            response = self.llm.invoke(prompt)
            analysis = json.loads(response)
            
            return AnalysisResult(
                skills_match=analysis["skills_match"],
                experience_analysis=analysis["experience_analysis"],
                improvement_suggestions=analysis["ats_optimization"]["keyword_suggestions"],
                overall_score=analysis["overall_assessment"]["score"],
                detailed_feedback=analysis["overall_assessment"]
            )
            
        except Exception as e:
            logger.error(f"Error in resume analysis: {str(e)}")
            raise

    def rank_candidates(self, job_description: str, resumes: Dict[str, str]) -> List[Tuple[str, float, Dict]]:
        """Rank multiple candidates for a position"""
        results = []
        
        for name, resume in resumes.items():
            try:
                analysis = self.analyze_resume(resume, job_description)
                results.append((
                    name,
                    analysis.overall_score,
                    {
                        "skills_match": analysis.skills_match,
                        "key_strengths": analysis.detailed_feedback["key_strengths"],
                        "improvement_areas": analysis.improvement_suggestions
                    }
                ))
            except Exception as e:
                logger.error(f"Error ranking candidate {name}: {str(e)}")
                continue
                
        return sorted(results, key=lambda x: x[1], reverse=True)

class StreamlitUI:
    """Handles all Streamlit UI components"""
    
    def __init__(self):
        self.initialize_session_state()
        self.setup_page_config()
        self.apply_custom_css()
        
    @staticmethod
    def initialize_session_state():
        """Initialize session state variables"""
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "analysis_results" not in st.session_state:
            st.session_state.analysis_results = None
            
    @staticmethod
    def setup_page_config():
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="KARM - AI Resume Assistant",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
    @staticmethod
    def apply_custom_css():
        """Apply custom CSS styling"""
        st.markdown("""
            <style>
            .main {
                padding: 2rem;
            }
            .stButton>button {
                width: 100%;
                margin-top: 1rem;
            }
            .analysis-card {
                padding: 1.5rem;
                border-radius: 0.5rem;
                border: 1px solid #e0e0e0;
                margin: 1rem 0;
            }
            .score-badge {
                padding: 0.5rem 1rem;
                border-radius: 0.25rem;
                font-weight: bold;
            }
            .chat-message {
                padding: 1rem;
                border-radius: 0.5rem;
                margin: 0.5rem 0;
            }
            </style>
        """, unsafe_allow_html=True)
        
    def display_sidebar(self) -> Tuple[str, List]:
        """Display and handle sidebar components"""
        with st.sidebar:
            st.title("KARM")
            st.subheader("AI Resume Assistant")
            
            user_type = st.selectbox(
                "Select User Type",
                ["Job Seeker", "Recruiter"],
                key="user_type"
            )
            
            st.subheader("Upload Documents")
            files = st.file_uploader(
                "Upload Resume(s)",
                accept_multiple_files=True,
                type=["pdf", "doc", "docx"]
            )
            
            return user_type, files
            
    def display_analysis_results(self, analysis: AnalysisResult):
        """Display resume analysis results"""
        st.subheader("Analysis Results")
        
        # Skills Match Section
        with st.expander("Skills Match Analysis", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Skills Relevance Score",
                    f"{analysis.skills_match['skill_relevance_score']:.1f}%"
                )
                st.subheader("Matching Skills")
                for skill in analysis.skills_match["matching_skills"]:
                    st.markdown(f"✅ {skill}")
                    
            with col2:
                st.subheader("Missing Critical Skills")
                for skill in analysis.skills_match["missing_skills"]:
                    st.markdown(f"❗ {skill}")
                    
        # Experience Analysis
        with st.expander("Experience Analysis", expanded=True):
            st.markdown(f"**Relevant Experience:**\n{analysis.experience_analysis['relevant_experience']}")
            st.markdown(f"**Achievement Impact:**\n{analysis.experience_analysis['achievement_impact']}")
            
        # Improvement Suggestions
        with st.expander("Improvement Suggestions", expanded=True):
            for idx, suggestion in enumerate(analysis.improvement_suggestions, 1):
                st.markdown(f"{idx}. {suggestion}")
                
        # Overall Assessment
        st.subheader("Overall Assessment")
        st.progress(analysis.overall_score / 100)
        st.markdown(f"**Final Recommendation:**\n{analysis.detailed_feedback['final_recommendation']}")

    def display_candidate_rankings(self, rankings: List[Tuple[str, float, Dict]]):
        """Display candidate rankings for recruiters"""
        st.subheader("Candidate Rankings")
        
        for rank, (name, score, details) in enumerate(rankings, 1):
            with st.container():
                st.markdown(f"### {rank}. {name}")
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.metric("Match Score", f"{score:.1f}%")
                    
                with col2:
                    st.markdown("**Key Strengths:**")
                    for strength in details["key_strengths"]:
                        st.markdown(f"✓ {strength}")
                        
                with st.expander("Detailed Analysis"):
                    st.markdown("**Skills Match:**")
                    for skill, relevance in details["skills_match"].items():
                        st.progress(relevance / 100)
                        st.markdown(f"{skill}: {relevance:.1f}%")
                        
                    st.markdown("**Areas for Improvement:**")
                    for area in details["improvement_areas"]:
                        st.markdown(f"- {area}")

def main():
    try:
        # Initialize components
        ui = StreamlitUI()
        analyzer = ResumeAnalyzer(os.getenv("TOGETHER_API_KEY"))
        
        # Display sidebar and get user inputs
        user_type, uploaded_files = ui.display_sidebar()
        
        # Process uploaded files
        if uploaded_files:
            documents = {}
            for file in uploaded_files:
                try:
                    text = DocumentProcessor.extract_text(file)
                    documents[file.name] = text
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
                    continue
            
            if documents:
                st.success(f"Successfully processed {len(documents)} document(s)")
                
                # Handle different user types
                if user_type == "Job Seeker":
                    if len(documents) > 1:
                        st.warning("Please upload only one resume for analysis")
                    else:
                        job_description = st.text_area(
                            "Paste the job description for analysis",
                            height=200
                        )
                        
                        if job_description and st.button("Analyze Resume"):
                            with st.spinner("Analyzing resume..."):
                                analysis = analyzer.analyze_resume(
                                    list(documents.values())[0],
                                    job_description
                                )
                                ui.display_analysis_results(analysis)
                                
                else:  # Recruiter
                    job_description = st.text_area(
                        "Paste the job description to rank candidates",
                        height=200
                    )
                    
                    if job_description and st.button("Rank Candidates"):
                        with st.spinner("Ranking candidates..."):
                            rankings = analyzer.rank_candidates(
                                job_description,
                                documents
                            )
                            ui.display_candidate_rankings(rankings)
                            
        # Chat interface
        user_input = st.chat_input(
            "Ask questions about the analysis or get specific recommendations"
        )
        
        if user_input:
            st.session_state.chat_history.append(HumanMessage(content=user_input))
            
            # Process chat input based on context
            response = analyzer.llm.invoke(
                f"Previous analysis context: {st.session_state.analysis_results if 'analysis_results' in st.session_state else 'None'}\n"
                f"User question: {user_input}\n"
                "Provide a helpful, specific response:"
            )
            
            st.session_state.chat_history.append(AIMessage(content=response))
            
            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message("User" if isinstance(message, HumanMessage) else "AI"):
                    st.write(message.content)
                    
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An error occurred. Please try again or contact support.")

if __name__ == "__main__":
    main()

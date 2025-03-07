import os
import streamlit as st
import openai
import json
import fitz  # PyMuPDF
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set up Streamlit with a professional config
st.set_page_config(page_title="Refine", layout="wide", initial_sidebar_state="expanded")
st.title("Refine")
st.markdown("**AI-Powered Resume Optimization** - Elevate your resume to match your dream job with precision and clarity.")

# Custom CSS based on your theme
st.markdown("""
    <style>
    .main { padding: 20px; background-color: #FFFFFF; }
    .stButton>button { background-color: #FF4B4B; color: #FFFFFF; border-radius: 5px; }
    .stTabs { margin-top: 20px; }
    .card { border: 1px solid #E0E0E0; border-radius: 10px; padding: 15px; background-color: #F0F2F6; }
    .highlight { background-color: #FF4B4B; color: #FFFFFF; padding: 5px; border-radius: 3px; }
    body { background-color: #FFFFFF; color: #31333F; }
    .stTextInput, .stFileUploader, .stTextArea { background-color: #FFFFFF; color: #31333F; }
    .stMarkdown, .stExpander { color: #31333F; }
    </style>
""", unsafe_allow_html=True)

# Function to extract text from PDFs using PyMuPDF
def extract_pdf_text(uploaded_file):
    """Extract text from an uploaded PDF file using PyMuPDF."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    text = ""
    try:
        doc = fitz.open(tmp_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
    finally:
        os.unlink(tmp_path)
    return text

# Generate an expert hiring manager persona
def generate_expert_prompt(job_description):
    prompt = f"""
    Imagine you’re a top-tier hiring manager for this job at American Express. Based on the job description below, craft a persona in a friendly, second-person tone, focusing on critical skills (e.g., SQL, Hadoop, Spark), projects (data pipelines), experience (5+ years professional), and domain (financial services).
    Job Description:
    ```
    {job_description}
    ```
    """
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You’re an expert AI recruiter."},
            {"role": "user", "content": prompt}
        ],
        temperature=1
    )
    return response.choices[0].message.content

# Evaluate resume text against the job description
def evaluate_resume_text(jd_text, resume_text):
    expert_prompt = generate_expert_prompt(jd_text)
    score_guidelines = """
    Evaluate the resume in these areas: Education, Skills and Techstack, Projects, Experience, Soft Skills, Profile, Industry/Domain, Location. Score each from 0-100 with strict criteria:
    - 86-100: Near-perfect match, meets or exceeds all key requirements.
    - 61-85: Good fit, meets most requirements with minor gaps.
    - 31-60: Partial match, meets some requirements but has significant gaps.
    - 0-30: Poor fit, major gaps in critical requirements (e.g., missing 5+ years experience, key skills).

    Key JD requirements:
    - Experience: At least 5+ years of professional (non-internship) experience with SQL and data pipelines. Internships count minimally unless explicitly long-term and relevant.
    - Skills and Techstack: Must include SQL (5+ years), Hadoop, Hive, Spark, Python/PySpark, and cloud (GCP/AWS/Azure).
    - Projects: Must involve designing/developing data pipelines, preferably in financial services.
    - Education: BS/MS in computer science, computer engineering, or closely related field.
    - Industry/Domain: Financial services experience is a plus.

    For Experience, extract the JD’s required years (e.g., '5+ years') and calculate the candidate’s total professional experience (exclude internships unless substantial). Penalize heavily if below 5 years.
    Calculate an overall match score as a weighted average:
    - Skills and Techstack: 25%
    - Projects: 25%
    - Experience: 20%
    - Profile: 15%
    - Education: 10%
    - Soft Skills: 3%
    - Industry/Domain: 2%
    - Location: 0% (do not include in overall score)
    Return JSON like:
    {
        "education_match": {"score": int, "reasoning": str},
        "skills_and_techstack_match": {"score": int, "reasoning": str},
        "projects_match": {"score": int, "reasoning": str},
        "experience_match": {"score": int, "reasoning": str},
        "soft_skills_match": {"score": int, "reasoning": str},
        "profile_match": {"score": int, "reasoning": str},
        "industry_and_domain_match": {"score": int, "reasoning": str},
        "location_match": {"score": int, "reasoning": str},
        "overall_match": {"score": int, "reasoning": str, "pros": str, "cons": str}
    }
    Ensure the response is a valid JSON object without additional text.
    """
    prompt = f"{expert_prompt}\n\nEvaluate this resume text:\n{resume_text}\n\n{score_guidelines}"
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content or "{}")

# Refine the resume
def refine_resume(jd_text, original_resume, evaluation):
    prompt = f"""
    Optimize this resume for the job description based on the evaluation results, while maintaining honesty in all sections, especially experience. 
    Focus on enhancing areas with low scores (e.g., skills and tech stack, projects, and soft skills) to better align with the job requirements. 
    If the candidate’s total professional experience is below 5+ years or not clearly stated, include a factual summary of their experience (e.g., based on listed roles and durations), but do not fabricate or exaggerate years of experience. 
    Incorporate missing skills (e.g., Hadoop, Spark) or enhance project descriptions to emphasize relevant objectives, but only if these can be reasonably inferred from the candidate’s existing background—do not invent skills or experiences not supported by the resume. 
    Wrap only the sections that are newly added or improved with **improved** markers; do not apply markers to unchanged sections (e.g., name, contact details, or areas already meeting requirements). 
    Job Description:
    {jd_text}
    Original Resume:
    {original_resume}
    Evaluation Results:
    {json.dumps(evaluation)}
    Return the refined resume text with **improved** markers around improvements.
    """
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You’re an expert resume writer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )
    return response.choices[0].message.content

# Highlight improvements
def highlight_improvements(refined_resume):
    parts = refined_resume.split("**improved**")
    highlighted = ""
    for i, part in enumerate(parts):
        if i % 2 == 0:
            highlighted += part
        else:
            highlighted += f'<span class="highlight">{part}</span>'
    return highlighted

# Sidebar for inputs
with st.sidebar:
    st.header("Get Started")
    st.markdown("Upload your resume and paste the job description to optimize your fit.")
    resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    jd_text = st.text_area("Paste Job Description", height=200)
    process_button = st.button("Optimize Resume")

# Main content
if process_button and resume_file and jd_text:
    with st.spinner("Processing your resume..."):
        # Extract resume text
        resume_text = extract_pdf_text(resume_file)

        # Evaluate original resume
        original_eval = evaluate_resume_text(jd_text, resume_text)
        if not original_eval:
            st.error("Failed to evaluate original resume.")
            st.stop()
        original_score = original_eval['overall_match']['score']

        # Refine resume
        refined_resume = refine_resume(jd_text, resume_text, original_eval)
        if not refined_resume:
            st.error("Failed to refine resume.")
            st.stop()

        # Evaluate refined resume
        refined_eval = evaluate_resume_text(jd_text, refined_resume)
        if not refined_eval:
            st.error("Failed to evaluate refined resume.")
            st.stop()
        refined_score = refined_eval['overall_match']['score']
        improvement_score = refined_score - original_score

        # Highlight improvements
        highlighted_resume = highlight_improvements(refined_resume)

    # Tabbed layout
    tab1, tab2, tab3 = st.tabs(["Original Resume", "Evaluation", "Refined Resume"])

    with tab1:
        st.subheader("Your Original Resume")
        st.markdown(f'<div class="card">{resume_text}</div>', unsafe_allow_html=True)

    with tab2:
        st.subheader("Resume Evaluation")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Score", original_score, delta_color="inverse")
        with col2:
            st.metric("Refined Score", refined_score, delta_color="normal")
        with col3:
            st.metric("Improvement", f"{improvement_score:+d}", delta=improvement_score, delta_color="normal")
        
        st.markdown("**Summary**")
        col_pros, col_cons = st.columns(2)
        with col_pros:
            st.write("Pros:", original_eval['overall_match']['pros'])
        with col_cons:
            st.write("Cons:", original_eval['overall_match']['cons'])

        with st.expander("Detailed Scores"):
            for key, value in original_eval.items():
                if key != "overall_match":
                    score = value['score']
                    color = "green" if score >= 86 else "orange" if score >= 61 else "red"
                    st.markdown(f"**{key.replace('_match', '').title()}**: <span style='color:{color}'>{score}</span> - {value['reasoning']}", unsafe_allow_html=True)

    with tab3:
        st.subheader("Optimized Resume")
        st.markdown(f'<div class="card">{highlighted_resume}</div>', unsafe_allow_html=True)
        st.markdown("*Highlighted sections in red indicate improvements tailored to the job description.*")

else:
    st.info("Please upload a resume and paste a job description, then click 'Optimize Resume' to begin.")

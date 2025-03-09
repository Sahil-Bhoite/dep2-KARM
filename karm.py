import os
import streamlit as st
import openai
import json
from dotenv import load_dotenv
import fitz  # PyMuPDF

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set up Streamlit with a professional config
st.set_page_config(page_title="Refine", layout="wide", initial_sidebar_state="expanded")
st.title("Refine")
st.markdown("**AI-Powered Resume Optimization** - Elevate your resume to match your dream job with precision and clarity.")

# Custom CSS for styling
st.markdown("""
<style>
.card {
    background-color: #f9f9f9;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}
.highlight {
    color: red;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Generate an expert hiring manager persona
def generate_expert_prompt(job_description):
    prompt = f"""
    Analyze the following job description and create a persona of a strict hiring manager with deep expertise in the relevant field. Your persona should reflect the specific requirements and preferences outlined in the job description.

    Job Description:
    {job_description}

    When creating your persona, consider the following:
    - Identify the industry or domain of the role and position yourself as an expert with over 10 years of experience in that area.
    - Determine whether the role is technical or non-technical. For technical roles, emphasize the importance of specific technical qualifications and a high rejection rate (e.g., rejecting 95% of applicants who don’t meet exact criteria). For non-technical or entry-level roles, focus on domain expertise, measurable outcomes, and potential for growth.
    - Extract key measurable achievements or metrics mentioned in the job description (e.g., revenue growth, system performance, customer metrics) and prioritize these in your evaluation.
    - If the job description specifies a minimum experience requirement (e.g., '5+ years'), enforce it strictly. If not, infer a reasonable minimum based on role complexity and industry norms (e.g., 2-3 years for mid-level, 0-1 years for entry-level) and evaluate flexibly.
    - Identify the most critical skills for the role and value depth in these areas over breadth, but allow for transferable skills when experience is limited.

    Based on this analysis, describe your persona in 2-3 sentences, highlighting how you will evaluate candidates based on the job description’s requirements. Ensure your persona is consistent, authoritative, and tailored to the role, while being adaptable to JDs without explicit experience minimums.
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
    STEP 1:
    Using the hiring persona from above and the job description, evaluate the candidate's resume and assign scores (0-100) with a brief explanation for each section:

    Evaluate these areas: Experience, Skills/Techstack, Projects, Education, Profile, Industry/Domain, Certifications/Achievements. 
    Assign each a score from 0 to 100 based on its alignment with the job description. Follow these guidelines:

    1. Experience Match:
    - If the JD specifies a minimum experience requirement (e.g., '5+ years'), enforce it strictly and base the score on that.
    - If no minimum is specified, infer a reasonable requirement based on role complexity and industry norms (e.g., 2-3 years for mid-level, 0-1 years for entry-level) and score flexibly, considering transferable experience.
    - 95-100: Exceeds inferred or required experience with directly relevant work.
    - 85-94: Meets exact inferred or required experience with relevant roles.
    - 75-84: 80-99% of inferred or required experience with some relevance.
    - 65-74: 50-79% of inferred or required experience with partial relevance.
    - <65: Below 50% of inferred or required experience or irrelevant roles.

    2. Skills/Techstack:
    - 95-100: All required skills present plus advanced bonus skills.
    - 85-94: All required skills present.
    - 75-84: Missing 1-2 secondary skills but has core skills.
    - 65-74: Missing some core skills but has transferable ones.
    - <65: Missing multiple core skills critical to the JD.

    3. Project Relevance:
    - 95-100: Multiple production-grade projects directly matching JD requirements.
    - 85-94: 1-2 relevant projects with clear, measurable impact.
    - 75-84: Academic or research projects relevant to the JD.
    - 65-74: Unrelated projects showing applicable skills.
    - <65: No relevant projects or impact demonstrated.

    4. Education:
    - Evaluate the candidate’s educational background flexibly, considering formal degrees, practical experience, and alternative qualifications.
    - For technical roles:
      * 95-100: Degree from a top-tier technical institute (e.g., MIT, Stanford) with relevant focus; or 5+ years of proven technical impact in a related role; or highly regarded bootcamp completion (e.g., Lambda School) with strong portfolios.
      * 85-94: Degree from a Tier-2 technical institute (e.g., strong regional schools); or reputable online certifications (e.g., Coursera, edX) with practical projects.
      * 75-84: Degree from a Tier-3 institute with relevant focus; or self-taught skills with significant open-source/personal projects.
      * 65-74: Non-technical degree with relevant certifications (e.g., AWS); or limited practical experience.
      * <65: Unrelated or unaccredited degree with no compensating qualifications.
    - For non-technical roles:
      * 95-100: Degree from a top-tier institute (e.g., Ivy League) matching JD; or 7+ years of relevant experience.
      * 85-94: Degree from a Tier-1 institute (e.g., national universities) with relevant focus; or advanced certifications.
      * 75-84: Degree from a Tier-2 institute with relevant focus; or practical experience showing required skills.
      * 65-74: Related degree from a lesser-known institute; or limited relevant experience.
      * <65: Unrelated or unaccredited degree with no compensating skills.
    - Adjust upward if practical experience or alternative qualifications strongly compensate for formal education gaps.

    5. Profile:
    - Assess career trajectory, role progression, and alignment with JD seniority.
    - 95-100: Consistent progression in directly relevant roles with leadership or advanced duties.
    - 85-94: Strong alignment with JD seniority, minor deviations (e.g., brief unrelated roles).
    - 75-84: Some relevance but inconsistent progression or unrelated roles.
    - 65-74: Loosely related trajectory with limited progression.
    - <65: Unrelated trajectory or no progression.

    6. Industry/Domain:
    - 95-100: Extensive experience in the exact industry/domain of the JD.
    - 85-94: Experience in a closely related industry/domain.
    - 75-84: Experience in a somewhat related industry/domain.
    - 65-74: Limited related industry/domain experience.
    - <65: No relevant industry/domain experience.

    7. Certifications/Achievements:
    - 95-100: Holds JD-required certifications or notable achievements (e.g., awards, patents).
    - 85-94: Holds related certifications or achievements not in JD.
    - 75-84: Holds general certifications or achievements.
    - 65-74: Limited certifications or achievements.
    - <65: No relevant certifications or achievements.

    For each section, provide a brief reasoning (1-2 sentences) explaining the score, specifically referencing the job description’s requirements and how the resume meets or falls short of them. Ensure reasoning is clear, concise, and actionable.

    STEP 2: Calculate weighted overall score:
    - Use these weights based on job seniority:
      - Senior-level roles (5+ years):
        * Experience: 40%
        * Skills/Techstack: 25%
        * Projects: 15%
        * Education: 10%
        * Profile: 5%
        * Industry/Domain: 3%
        * Certifications/Achievements: 2%
      - Mid-level roles (2-5 years):
        * Experience: 30%
        * Skills/Techstack: 30%
        * Projects: 20%
        * Education: 10%
        * Profile: 5%
        * Industry/Domain: 3%
        * Certifications/Achievements: 2%
      - Entry-level roles (0-2 years):
        * Experience: 20%
        * Skills/Techstack: 30%
        * Projects: 25%
        * Education: 15%
        * Profile: 5%
        * Industry/Domain: 3%
        * Certifications/Achievements: 2%
      - If the JD emphasizes specific areas (e.g., 'strong project portfolio'), adjust weights (e.g., increase Projects by 5-10%, reduce another category).

    - Calculate: (Experience * weight) + (Skills * weight) + (Projects * weight) + (Education * weight) + (Profile * weight) + (Industry/Domain * weight) + (Certifications/Achievements * weight)

    For overall reasoning, list critical gaps in 2-3 sentences tied to the JD. Keep pros, cons, and fit reasoning distinct and concise.
    Return a JSON:
    {
        "experience_match": { "score": integer, "reasoning": string },
        "skills_and_techstack_match": { "score": integer, "reasoning": string },
        "projects_match": { "score": integer, "reasoning": string },
        "education_match": { "score": integer, "reasoning": string },
        "profile_match": { "score": integer, "reasoning": string },
        "industry_and_domain_match": { "score": integer, "reasoning": string },
        "certifications_and_achievements_match": { "score": integer, "reasoning": string },
        "overall_match": { 
            "score": integer, 
            "reasoning": string, 
            "pros": string, 
            "cons": string, 
            "fit": { "decision": boolean, "reasoning": string }
        }
    }
    - Fit decision: False if experience_match < 85 AND the JD explicitly specifies a minimum experience requirement; otherwise, True if overall score >= 85, with reasoning tied to JD alignment and flexibility for transferable skills if no minimum is specified.
    Ensure JSON is valid, reasoning is specific, and no extra text is included.
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

# Refine the resume with a perfectionist approach
def refine_resume(jd_text, original_resume, evaluation):
    prompt = f"""
    You are a perfectionist resume writer with an eye for detail and a deep understanding of how to align resumes precisely with job descriptions. Your task is to optimize this resume for the given job description based on the evaluation results, while maintaining absolute honesty, especially regarding years of experience.

    ### Instructions:
    - **Focus on Low-Scoring Areas**: Prioritize enhancing sections with low scores (Skills/Techstack, Projects, Certifications/Achievements) to better align with the job requirements, even if experience is limited, using strong action verbs (e.g., 'optimized', 'delivered', 'streamlined') and quantifiable achievements where possible.
    - **Maintain Honesty**: Do not alter years of experience, fabricate roles, or add fictitious projects/certifications. Only refine descriptions within existing roles to highlight relevant responsibilities or achievements if they can be reasonably inferred from the text.
    - **Skills/Techstack**: Add missing skills only if they can be directly inferred from projects or roles (e.g., add 'Python' if PySpark is mentioned). Mark inferred skills with '[Inferred from related experience]' to maintain transparency.
    - **Projects**: Enhance project descriptions to emphasize measurable impact (e.g., 'improved system efficiency by 20%' or 'processed 1M records daily') only if such impact can be inferred from the resume text. Do not add new projects.
    - **Certifications/Achievements**: Add relevant certifications or achievements only if they are implied by the resume (e.g., 'AWS Certified' if AWS projects are listed). Mark inferred items with '[Inferred from project experience]'.
    - **Conservative Inferences**: Ensure all enhancements are directly supported by the existing resume content. Do not make assumptions beyond what is explicitly stated or reasonably implied.
    - **Maintain Original Tone**: Preserve the original tone and style of the resume (e.g., formal, concise, or detailed) to ensure a professional and coherent refined version.
    - **Mark Improvements**: Wrap only newly added or improved sections with **improved** markers. Leave unchanged sections (e.g., name, contact details, experience years, or areas already meeting requirements) unmarked.

    ### Job Description:
    {jd_text}

    ### Original Resume:
    {original_resume}

    ### Evaluation Results:
    {json.dumps(evaluation)}

    Return the refined resume text with **improved** markers around improvements only.
    """
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a perfectionist resume writer with an eye for detail and a deep understanding of how to align resumes precisely with job descriptions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )
    return response.choices[0].message.content

# Highlight improvements in the UI
def highlight_improvements(refined_resume):
    parts = refined_resume.split("**improved**")
    highlighted = ""
    for i, part in enumerate(parts):
        if i % 2 == 0:
            highlighted += part
        else:
            highlighted += f'<span class="highlight">{part}</span>'
    return highlighted

# Sidebar for user inputs
with st.sidebar:
    st.header("Get Started")
    st.markdown("Provide your resume (via text or PDF) and job description to optimize your fit.")
    
    resume_input_method = st.selectbox(
        "How would you like to provide your resume?",
        ["Paste Text", "Upload PDF"],
        index=1  # Makes "Upload PDF" the default option
    )
    
    if resume_input_method == "Paste Text":
        resume_text = st.text_area("Paste Resume Text", height=200)
    else:
        pdf_file = st.file_uploader("Upload Resume PDF", type=["pdf"])
        if pdf_file is not None:
            try:
                pdf_bytes = pdf_file.read()
                with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf_document:
                    resume_text = ""
                    for page in pdf_document:
                        resume_text += page.get_text()
                if not resume_text.strip():
                    st.warning("The uploaded PDF does not contain extractable text. Please ensure your resume is not scanned or image-based.")
            except Exception as e:
                st.error(f"Error processing PDF: {e}")
                resume_text = ""
        else:
            resume_text = ""
    
    jd_text = st.text_area("Paste Job Description", height=200)
    process_button = st.button("Optimize Resume")

# Main content processing
if process_button:
    if not jd_text:
        st.error("Please provide a job description.")
    elif not resume_text:
        if resume_input_method == "Paste Text":
            st.error("Please paste your resume text.")
        else:
            st.error("Please upload your resume PDF.")
    else:
        with st.spinner("Processing your resume..."):
            # Evaluate original resume
            original_eval = evaluate_resume_text(jd_text, resume_text)
            if not original_eval:
                st.error("Failed to evaluate original resume.")
                st.stop()
            original_score = original_eval['overall_match']['score']
            fit_decision = original_eval['overall_match']['fit']['decision']

            # Refine resume with perfectionist approach
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

        # Display warning if experience is low, with softer language if no explicit requirement
        if not fit_decision:
            warning_message = "Based on the evaluation, your experience may not fully align with the job description. Consider gaining more relevant experience or highlighting transferable skills."
            if "years" in jd_text.lower() and any(char.isdigit() for char in jd_text):  # Check if JD mentions explicit experience
                warning_message = "Your experience does not meet the minimum requirement of the job description. You should not apply for this role unless you gain more relevant experience."
            st.warning(warning_message)

        # Tabbed layout for results
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
                        st.markdown(f"**{key.replace('_match', '').replace('_and_', '/').title()}**: <span style='color:{color}'>{score}</span> - {value['reasoning']}", unsafe_allow_html=True)

            # Add fit decision
            fit_decision = original_eval['overall_match']['fit']['decision']
            fit_reasoning = original_eval['overall_match']['fit']['reasoning']
            st.markdown(f"**Fit for Role**: {'Yes' if fit_decision else 'No'} - {fit_reasoning}")

        with tab3:
            st.subheader("Optimized Resume")
            st.markdown(f'<div class="card">{highlighted_resume}</div>', unsafe_allow_html=True)
            st.markdown("*Highlighted sections in red indicate improvements tailored to the job description.*")

else:
    st.info("Please provide your resume and job description, then click 'Optimize Resume' to begin.")

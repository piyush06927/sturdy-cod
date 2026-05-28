import streamlit as st
from parser import read_file
from matcher import calculate_match_score
from skills import extract_skills, generate_recommendations

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="AI CV Matcher",
    page_icon="📄",
    layout="wide"
)

# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.stButton button {
    width: 100%;
    border-radius: 10px;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
}

.skill-box {
    padding: 10px;
    border-radius: 10px;
    background-color: #1E1E1E;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #

with st.sidebar:
    st.title("📄 AI CV Matcher")

    st.markdown("---")

    st.info(
        """
        Upload:
        - Resume/CV
        - Job Description

        Get:
        - Semantic similarity
        - Skill matching
        - ATS-style recommendations
        """
    )

    cv_file = st.file_uploader(
        "Upload CV",
        type=["pdf", "docx", "txt"]
    )

    jd_file = st.file_uploader(
        "Upload Job Description",
        type=["pdf", "docx", "txt"]
    )

# ---------------- MAIN UI ---------------- #

st.title("📄 AI CV Matcher")
st.caption("AI-powered resume screening using semantic embeddings and NLP")

additional_skills = st.text_area(
    "Additional Skills (comma-separated)",
    placeholder="Python, AWS, Docker"
)

extra_skills = set()

if additional_skills:
    extra_skills = {
        skill.strip().lower()
        for skill in additional_skills.split(",")
        if skill.strip()
    }

# ---------------- ANALYSIS ---------------- #

if cv_file and jd_file:

    with st.spinner("Analyzing CV..."):

        cv_text = read_file(cv_file)
        jd_text = read_file(jd_file)

        semantic_score = calculate_match_score(
            cv_text,
            jd_text
        )

        cv_skills = extract_skills(cv_text)
        jd_skills = extract_skills(jd_text)

        jd_skills = jd_skills.union(extra_skills)

        matched_skills = cv_skills & jd_skills
        missing_skills = jd_skills - cv_skills

        skill_score = (
            len(matched_skills) / len(jd_skills)
            if len(jd_skills) > 0 else 0
        )

        overall_score = (
            0.7 * semantic_score +
            0.3 * skill_score
        )

        recommendations = generate_recommendations(
            missing_skills
        )

    # ---------------- METRICS ---------------- #

    st.subheader("📊 Match Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Overall Match",
            f"{overall_score * 100:.1f}%"
        )

    with col2:
        st.metric(
            "Semantic Match",
            f"{semantic_score * 100:.1f}%"
        )

    with col3:
        st.metric(
            "Skill Match",
            f"{skill_score * 100:.1f}%"
        )

    st.progress(float(overall_score))

    st.divider()

    # ---------------- SKILLS ---------------- #

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("✅ Matched Skills")

        if matched_skills:
            for skill in sorted(matched_skills):
                st.success(skill)

        else:
            st.warning("No matched skills found.")

    with col2:
        st.subheader("❌ Missing Skills")

        if missing_skills:
            for skill in sorted(missing_skills):
                st.error(skill)

        else:
            st.success("All required skills found!")

    st.divider()

    # ---------------- RECOMMENDATIONS ---------------- #

    st.subheader("💡 ATS Recommendations")

    if recommendations:
        for rec in recommendations:
            st.info(rec)

    else:
        st.success("Excellent alignment with the job description.")

    # ---------------- EXPANDERS ---------------- #

    with st.expander("📄 View Extracted CV Skills"):
        st.write(sorted(cv_skills))

    with st.expander("📋 View Extracted Job Skills"):
        st.write(sorted(jd_skills))

else:
    st.info("👈 Upload both files to begin analysis")


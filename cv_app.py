import streamlit as st
from parser import read_file
from matcher import semantic_similarity
from skills import (
    extract_skills,
    generate_recommendations
)

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="AI CV Matcher",
    page_icon="📄",
    layout="wide"
)

# ---------------- SIDEBAR ---------------- #

with st.sidebar:

    st.title("📄 AI CV Matcher")

    st.markdown("---")

    st.info(
        """
        Upload:
        - Resume/CV
        - Job Description

        Features:
        - Semantic embeddings
        - ATS-style scoring
        - Skill extraction
        - NLP recommendations
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
st.caption(
    "AI-powered ATS matching using NLP and semantic embeddings"
)

# ---------------- EXTRA SKILLS ---------------- #

additional_skills = st.text_area(
    "Additional Required Skills",
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

    with st.spinner("Analyzing documents..."):

        cv_text = read_file(cv_file)
        jd_text = read_file(jd_file)

        # ---------- Semantic Similarity ---------- #

        semantic_score = semantic_similarity(
            cv_text,
            jd_text
        )

        # ---------- Skill Extraction ---------- #

        cv_skills = extract_skills(cv_text)

        jd_skills = extract_skills(jd_text)

        jd_skills = jd_skills.union(extra_skills)

        # ---------- Skill Comparison ---------- #

        matched_skills = cv_skills & jd_skills

        missing_skills = jd_skills - cv_skills

        # ---------- Skill Score ---------- #

        skill_score = (
            len(matched_skills) / len(jd_skills)
            if len(jd_skills) > 0 else 0
        )

        # ---------- Weighted Overall Score ---------- #

        overall_score = (
            semantic_score * 0.7 +
            skill_score * 0.3
        )

        recommendations = generate_recommendations(
            missing_skills
        )

    # ---------------- RESULTS ---------------- #

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

        for recommendation in recommendations:
            st.info(recommendation)

    else:
        st.success(
            "Excellent alignment with the job description."
        )

    # ---------------- DEBUG / EXPANDERS ---------------- #

    with st.expander("📄 Extracted CV Skills"):
        st.write(sorted(cv_skills))

    with st.expander("📋 Extracted JD Skills"):
        st.write(sorted(jd_skills))

else:

    st.info(
        "👈 Upload both CV and Job Description to begin."
    )

import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# Page configuration
st.set_page_config(page_title="CV Matcher", layout="wide")

# Title
st.title("CV vs Job Description Matcher")

# Sidebar for file uploads
st.sidebar.header("Upload Files")
cv_file = st.sidebar.file_uploader("Upload your CV (PDF)", type=["pdf"])
jd_file = st.sidebar.file_uploader("Upload Job Description (DOCX or TXT)", type=["docx", "txt"])

# File reading functions
def read_pdf(file):
    reader = PdfReader(file)
    return "".join([page.extract_text() or "" for page in reader.pages])

def read_docx(file):
    doc = Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

# Load models
@st.cache_resource
def load_models():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    nlp = spacy.load("en_core_web_sm")
    return model, nlp

model, nlp = load_models()

# Similarity function
def get_similarity(text1, text2):
    emb1 = model.encode(text1)
    emb2 = model.encode(text2)
    return cosine_similarity([emb1], [emb2])[0][0]

# Extract skills
def extract_skills(text):
    doc = nlp(text)
    skills = set()
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:
            skills.add(token.text.lower())
    return skills

# Main analysis
if cv_file and jd_file:
    cv_text = read_pdf(cv_file)
    
    if jd_file.name.endswith(".docx"):
        jd_text = read_docx(jd_file)
    else:
        jd_text = jd_file.read().decode("utf-8")
    
    # Calculate scores
    semantic_score = get_similarity(cv_text, jd_text)
    jd_skills = extract_skills(jd_text)
    cv_skills = extract_skills(cv_text)
    matched = jd_skills & cv_skills
    missing = jd_skills - cv_skills
    
    skill_score = len(matched) / len(jd_skills) if len(jd_skills) > 0 else 0
    overall = (0.6 * skill_score) + (0.4 * semantic_score)
    
    # Display results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Match", f"{overall*100:.1f}%")
    with col2:
        st.metric("Skill Match", f"{skill_score*100:.1f}%")
    with col3:
        st.metric("Semantic Match", f"{semantic_score*100:.1f}%")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("✅ Matched Skills")
        st.write(", ".join(sorted(matched)) if matched else "No skills matched")
    with col2:
        st.subheader("❌ Missing Skills")
        st.write(", ".join(sorted(missing)) if missing else "All skills present!")
else:
    st.info("👈 Please upload both files in the sidebar to begin analysis")
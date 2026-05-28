from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

@st.cache_resource
def load_model():
    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

model = load_model()


def calculate_match_score(cv_text, jd_text):

    cv_embedding = model.encode(
        cv_text,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    jd_embedding = model.encode(
        jd_text,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    similarity = cosine_similarity(
        [cv_embedding],
        [jd_embedding]
    )[0][0]

    return similarity

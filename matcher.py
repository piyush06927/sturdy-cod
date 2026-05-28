import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_resource
def load_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


model = load_model()


def get_embedding(text):

    return model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True
    )


def semantic_similarity(text1, text2):

    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)

    similarity = cosine_similarity(
        [emb1],
        [emb2]
    )[0][0]

    return float(similarity)

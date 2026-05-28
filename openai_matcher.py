from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import numpy as np
import streamlit as st
import os

# Load environment variables
load_dotenv()

# Read API Key
api_key = (
    st.secrets.get("OPENAI_API_KEY")
    or os.getenv("OPENAI_API_KEY")
)

if not api_key:
    raise ValueError(
        "OpenAI API key not found."
    )

# Initialize client
client = OpenAI(api_key=api_key)


model="text-embedding-3-small"



# ---------------- GET EMBEDDING ---------------- #

def get_embedding(text):

    response = client.embeddings.create(
        model=model,
        input=text
    )

    embedding = response.data[0].embedding

    return np.array(embedding)


# ---------------- SEMANTIC SIMILARITY ---------------- #

def semantic_similarity(text1, text2):

    emb1 = get_embedding(text1)

    emb2 = get_embedding(text2)

    similarity = cosine_similarity(
        [emb1],
        [emb2]
    )[0][0]

    return float(similarity)

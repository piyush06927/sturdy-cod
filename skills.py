import re
import numpy as np
import spacy
import streamlit as st

from matcher import model
from sklearn.metrics.pairwise import cosine_similarity
from skills_db import SKILL_ONTOLOGY


# ---------------- LOAD NLP ---------------- #

@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")


nlp = load_nlp()


# ---------------- PREPARE SKILLS ---------------- #

CANONICAL_SKILLS = list(SKILL_ONTOLOGY.keys())

ALL_ALIASES = {}

for canonical, aliases in SKILL_ONTOLOGY.items():

    ALL_ALIASES[canonical] = canonical

    for alias in aliases:
        ALL_ALIASES[alias.lower()] = canonical


# ---------------- EMBEDDINGS ---------------- #

SKILL_EMBEDDINGS = model.encode(
    CANONICAL_SKILLS,
    convert_to_numpy=True,
    normalize_embeddings=True
)


# ---------------- CLEAN TEXT ---------------- #

def clean_text(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z0-9+#./ ]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------------- RULE-BASED EXTRACTION ---------------- #

def extract_rule_based_skills(text):

    text = clean_text(text)

    extracted = set()

    for alias, canonical in ALL_ALIASES.items():

        pattern = r"\b" + re.escape(alias) + r"\b"

        if re.search(pattern, text):
            extracted.add(canonical)

    return extracted


# ---------------- SPACY PHRASE EXTRACTION ---------------- #

def extract_candidate_phrases(text):

    doc = nlp(text)

    phrases = set()

    # noun chunks
    for chunk in doc.noun_chunks:

        phrase = chunk.text.strip().lower()

        if 2 <= len(phrase) <= 40:
            phrases.add(phrase)

    # named entities
    for ent in doc.ents:

        phrase = ent.text.strip().lower()

        if 2 <= len(phrase) <= 40:
            phrases.add(phrase)

    return phrases


# ---------------- SEMANTIC MATCHING ---------------- #

def semantic_skill_matching(phrases, threshold=0.60):

    matched_skills = set()

    for phrase in phrases:

        phrase_embedding = model.encode(
            phrase,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        similarities = cosine_similarity(
            [phrase_embedding],
            SKILL_EMBEDDINGS
        )[0]

        best_idx = np.argmax(similarities)

        best_score = similarities[best_idx]

        if best_score >= threshold:

            matched_skills.add(
                CANONICAL_SKILLS[best_idx]
            )

    return matched_skills


# ---------------- FINAL EXTRACTION ---------------- #

def extract_skills(text):

    text = clean_text(text)

    # Exact + synonym matching
    rule_skills = extract_rule_based_skills(text)

    # NLP phrase extraction
    phrases = extract_candidate_phrases(text)

    # Semantic inference
    semantic_skills = semantic_skill_matching(phrases)

    # Merge results
    final_skills = (
        rule_skills
        .union(semantic_skills)
    )

    return final_skills


# ---------------- RECOMMENDATIONS ---------------- #

def generate_recommendations(missing_skills):

    recommendations = []

    recommendation_map = {

        "aws": "Consider adding AWS cloud deployment experience or certifications.",

        "docker": "Include containerization or Docker-based deployment projects.",

        "kubernetes": "Add Kubernetes orchestration experience if applicable.",

        "terraform": "Mention Infrastructure as Code tools like Terraform.",

        "python": "Highlight Python scripting or backend automation projects.",

        "sql": "Include database querying and optimization experience.",

        "spark": "Add big data or distributed processing experience.",

        "airflow": "Mention workflow orchestration or ETL scheduling tools.",

        "teamcenter": "Include Teamcenter customization or PLM implementation experience.",

        "bmide": "Highlight BMIDE configuration/customization work.",

        "itk": "Mention Teamcenter ITK development projects.",

        "active workspace": "Include Active Workspace customization experience.",

        "solution architecture": "Add system design or architecture responsibilities.",

        "microservices": "Highlight distributed systems or API-based architectures.",

        "devops": "Include CI/CD pipelines and deployment automation projects."

    }

    for skill in sorted(missing_skills):

        if skill in recommendation_map:

            recommendations.append(
                recommendation_map[skill]
            )

        else:

            recommendations.append(
                f"Consider adding experience related to '{skill}'."
            )

    return recommendations

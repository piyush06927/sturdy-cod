TECH_SKILLS = {

    "python",
    "java",
    "c++",
    "sql",
    "aws",
    "docker",
    "kubernetes",
    "tensorflow",
    "pytorch",
    "machine learning",
    "deep learning",
    "nlp",
    "data analysis",
    "pandas",
    "numpy",
    "streamlit",
    "fastapi",
    "git",
    "linux",
    "azure",
    "power bi",
    "excel",
    "javascript",
    "react",
    "node.js",
    "mongodb"

}


def extract_skills(text):

    text = text.lower()

    found_skills = set()

    for skill in TECH_SKILLS:

        if skill in text:
            found_skills.add(skill)

    return found_skills


def generate_recommendations(missing_skills):

    recommendations = []

    for skill in missing_skills:

        recommendations.append(
            f"Consider adding experience/projects related to '{skill}'."
        )

    return recommendations

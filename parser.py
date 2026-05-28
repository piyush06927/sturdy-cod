from PyPDF2 import PdfReader
from docx import Document


def read_file(file):

    try:

        if file.name.endswith(".pdf"):

            reader = PdfReader(file)

            text = ""

            for page in reader.pages:
                text += page.extract_text() or ""

            return text

        elif file.name.endswith(".docx"):

            doc = Document(file)

            return "\n".join(
                [p.text for p in doc.paragraphs]
            )

        elif file.name.endswith(".txt"):

            return file.read().decode("utf-8")

        return ""

    except Exception as e:

        return f"Error reading file: {str(e)}"

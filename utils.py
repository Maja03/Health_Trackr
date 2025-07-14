import io
import fitz
import docx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the embedding model once globally for efficiency
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_file(filename: str, content: bytes) -> str:
    """Extract text from PDF, DOCX, or plain text files."""
    if filename.lower().endswith('.pdf'):
        with fitz.open(stream=content, filetype='pdf') as doc:
            return " ".join([page.get_text() for page in doc])
    elif filename.lower().endswith('.docx'):
        f = io.BytesIO(content)
        doc = docx.Document(f)
        return " ".join([para.text for para in doc.paragraphs])
    else:
        return content.decode("utf-8", errors="ignore")


def get_embedding(text: str):
    return model.encode([text])[0]


def semantic_similarity(text1: str, text2: str) -> float:
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    score = cosine_similarity([emb1], [emb2])[0][0]
    return score  # Between 0 and 1


def extract_keywords(text: str) -> set:
    # Simple keyword extractor: split by commas or spaces, lowercase
    words = set(word.strip().lower() for word in text.replace(',', ' ').split() if word.strip())
    return words


def get_ml_match_suggestions(resume_text: str, job_text: str) -> dict:
    """Calculate semantic similarity + keyword matching and provide suggestions."""

    if not resume_text or not job_text:
        return {
            "score": 0,
            "matched_keywords": [],
            "missing_keywords": [],
            "suggestions": "Please provide both resume text and job description."
        }

    # Semantic similarity score (0-1)
    sem_sim = semantic_similarity(resume_text, job_text)

    # Keyword sets from job description and resume
    job_keywords = extract_keywords(job_text)
    resume_keywords = extract_keywords(resume_text)

    matched_keywords = sorted(list(job_keywords & resume_keywords))
    missing_keywords = sorted(list(job_keywords - resume_keywords))

    # Keyword match ratio
    if len(job_keywords) > 0:
        kw_score = len(matched_keywords) / len(job_keywords)
    else:
        kw_score = 0

    # Combine semantic similarity and keyword matching (weights: 70% semantic, 30% keywords)
    combined_score = 0.7 * sem_sim + 0.3 * kw_score
    combined_score_pct = float(round(combined_score * 100, 2))

    # Suggestions
    if missing_keywords:
        suggestions = (
            "Your resume is missing the following keywords: " +
            ", ".join(missing_keywords) +
            ". Consider adding them to improve your match score."
        )
    else:
        suggestions = "Excellent match! Your resume covers all requested keywords."

    return {
        "score": combined_score_pct,
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords,
        "suggestions": suggestions
    }

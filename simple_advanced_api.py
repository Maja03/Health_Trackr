"""
Simplified Advanced API for Resume-Job Matching
A sophisticated ML system that works with available packages
"""

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import re
import json
from datetime import datetime
import logging
from utils import extract_text_from_file
from auth import hash_password, verify_password, create_token
from models import SessionLocal, User, init_db

# Initialize database
init_db()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Advanced Resume-Job Matching API",
    description="Sophisticated ML-powered resume-job matching system",
    version="2.0.0"
)

# CORS configuration
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ML components
try:
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Sentence transformer model loaded successfully")
except Exception as e:
    logger.warning(f"Could not load sentence transformer: {e}")
    sentence_model = None

# Initialize models
tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
scaler = StandardScaler()

# Skill databases
TECHNICAL_SKILLS = {
    'python', 'java', 'javascript', 'sql', 'html', 'css', 'react', 'angular', 'vue',
    'node.js', 'express', 'django', 'flask', 'fastapi', 'spring', 'docker', 'kubernetes',
    'aws', 'azure', 'gcp', 'git', 'linux', 'agile', 'scrum', 'machine learning',
    'data science', 'artificial intelligence', 'deep learning', 'tensorflow', 'pytorch',
    'pandas', 'numpy', 'matplotlib', 'seaborn', 'tableau', 'power bi', 'excel',
    'salesforce', 'hubspot', 'seo', 'sem', 'google analytics', 'email marketing',
    'content marketing', 'social media marketing', 'ux', 'ui', 'figma', 'sketch',
    'adobe creative suite', 'project management', 'business analysis', 'testing',
    'qa', 'devops', 'ci/cd', 'microservices', 'api', 'rest', 'graphql', 'mobile development',
    'ios', 'android', 'react native', 'flutter', 'blockchain', 'cryptocurrency'
}

SOFT_SKILLS = {
    'leadership', 'communication', 'teamwork', 'collaboration', 'problem solving',
    'critical thinking', 'analytical thinking', 'creativity', 'innovation',
    'adaptability', 'flexibility', 'time management', 'organization', 'planning',
    'attention to detail', 'accuracy', 'quality', 'efficiency', 'productivity',
    'initiative', 'self-motivation', 'drive', 'persistence', 'resilience',
    'stress management', 'emotional intelligence', 'empathy', 'interpersonal skills',
    'relationship building', 'networking', 'influence', 'persuasion', 'negotiation',
    'conflict resolution', 'decision making', 'judgment', 'strategic thinking',
    'vision', 'mentoring', 'coaching', 'training', 'teaching', 'presentation',
    'public speaking', 'writing', 'documentation', 'research', 'analysis',
    'customer service', 'client relations', 'stakeholder management',
    'project management', 'risk management', 'change management', 'cultural awareness',
    'diversity', 'inclusion', 'ethics', 'integrity', 'trustworthiness', 'reliability',
    'accountability', 'responsibility', 'professionalism', 'business acumen',
    'commercial awareness', 'market knowledge', 'industry expertise',
    'continuous learning', 'growth mindset', 'curiosity', 'open-mindedness',
    'feedback', 'improvement', 'excellence'
}

PROGRAMMING_LANGUAGES = {
    'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby',
    'go', 'rust', 'swift', 'kotlin', 'scala', 'matlab', 'sas', 'stata',
    'spss', 'julia', 'perl', 'bash', 'shell', 'powershell', 'sql', 'pl/sql',
    't-sql', 'html', 'css', 'sass', 'less', 'xml', 'json', 'yaml', 'markdown',
    'latex', 'dart', 'lua', 'haskell', 'erlang', 'elixir', 'clojure', 'f#',
    'ocaml', 'nim', 'crystal', 'zig', 'odin', 'carbon', 'mojo'
}

FRAMEWORKS = {
    'react', 'angular', 'vue', 'svelte', 'ember', 'backbone', 'jquery', 'bootstrap',
    'tailwind', 'material-ui', 'ant design', 'node.js', 'express', 'koa', 'hapi',
    'fastify', 'nest.js', 'django', 'flask', 'fastapi', 'tornado', 'bottle',
    'web2py', 'spring', 'hibernate', 'struts', 'jsf', 'wicket', 'vaadin',
    'laravel', 'symfony', 'codeigniter', 'yii', 'cakephp', 'rails', 'sinatra',
    'hanami', 'grape', 'padrino', 'asp.net', 'entity framework', 'nhibernate',
    'castle windsor', 'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas',
    'numpy', 'matplotlib', 'seaborn', 'plotly', 'bokeh', 'd3.js', 'chart.js',
    'highcharts', 'amcharts', 'fusioncharts', 'docker', 'kubernetes', 'rancher',
    'openshift', 'docker swarm', 'terraform', 'ansible', 'puppet', 'chef',
    'salt', 'jenkins', 'gitlab ci', 'github actions', 'circleci', 'travis ci',
    'aws', 'azure', 'gcp', 'heroku', 'digitalocean', 'linode', 'vultr',
    'rackspace', 'ibm cloud', 'oracle cloud'
}

class MatchRequest(BaseModel):
    resume: str
    job_description: str
    analysis_level: str = "comprehensive"

class MatchResponse(BaseModel):
    match_score: float
    overall_assessment: str
    skill_analysis: Dict[str, Any]
    detailed_feedback: Dict[str, Any]
    model_predictions: Dict[str, float]
    feature_analysis: Dict[str, float]
    confidence_metrics: Dict[str, float]
    processing_time: float
    timestamp: str

class SkillAnalysisRequest(BaseModel):
    text: str

class SkillAnalysisResponse(BaseModel):
    technical_skills: List[str]
    soft_skills: List[str]
    programming_languages: List[str]
    frameworks: List[str]
    all_skills: List[str]
    keyword_extraction: Dict[str, List[str]]

class TextAnalysisRequest(BaseModel):
    text: str

class TextAnalysisResponse(BaseModel):
    text_statistics: Dict[str, Any]
    readability_metrics: Dict[str, float]
    keyword_density: Dict[str, float]

def extract_skills(text: str) -> Dict[str, Any]:
    """Extract skills using multiple techniques with improved filtering"""
    text_lower = text.lower()
    
    # Direct matching with word boundary checking
    def find_skills_in_text(skill_set: set, text: str) -> set:
        found_skills = set()
        for skill in skill_set:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text):
                found_skills.add(skill)
        return found_skills
    
    found_technical = find_skills_in_text(TECHNICAL_SKILLS, text_lower)
    found_soft = find_skills_in_text(SOFT_SKILLS, text_lower)
    found_programming = {lang for lang in PROGRAMMING_LANGUAGES if len(lang) > 1 and re.search(r'\b' + re.escape(lang) + r'\b', text_lower)}
    found_frameworks = find_skills_in_text(FRAMEWORKS, text_lower)
    
    # Simple keyword extraction using TF-IDF
    try:
        tfidf_matrix = tfidf_vectorizer.fit_transform([text])
        feature_names = tfidf_vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]
        
        # Get top keywords with minimum length filter
        top_indices = np.argsort(tfidf_scores)[-10:][::-1]
        tfidf_keywords = [feature_names[i] for i in top_indices 
                         if tfidf_scores[i] > 0 and len(feature_names[i]) >= 3]
    except:
        tfidf_keywords = []
    
    # Regex-based extraction with better filtering
    regex_keywords = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())  # Minimum 4 characters
    regex_keywords = [word for word in regex_keywords 
                     if len(word) >= 4 and word not in ['the', 'and', 'for', 'with', 'from', 'this', 'that', 'have', 'been', 'will', 'they', 'their', 'there']][:10]
    
    # Filter out single letters and very short words from all skills
    all_skills = found_technical | found_soft | found_programming | found_frameworks
    filtered_skills = {skill for skill in all_skills if len(skill) >= 3}
    
    return {
        'technical_skills': list(found_technical),
        'soft_skills': list(found_soft),
        'programming_languages': list(found_programming),
        'frameworks': list(found_frameworks),
        'tfidf_keywords': tfidf_keywords,
        'regex_keywords': regex_keywords,
        'all_skills': list(filtered_skills)
    }

def calculate_text_statistics(text: str) -> Dict[str, Any]:
    """Calculate comprehensive text statistics"""
    sentences = re.split(r'[.!?]+', text)
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Calculate readability metrics
    avg_sentence_length = len(words) / max(len(sentences), 1)
    avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
    
    # Calculate keyword density
    skill_words = TECHNICAL_SKILLS | SOFT_SKILLS | PROGRAMMING_LANGUAGES | FRAMEWORKS
    skill_count = sum(1 for word in words if word in skill_words)
    skill_density = skill_count / max(len(words), 1)
    
    return {
        'character_count': len(text),
        'word_count': len(words),
        'sentence_count': len(sentences),
        'unique_words': len(set(words)),
        'avg_sentence_length': avg_sentence_length,
        'avg_word_length': avg_word_length,
        'skill_count': skill_count,
        'skill_density': skill_density
    }

def calculate_readability_metrics(text: str) -> Dict[str, float]:
    """Calculate readability metrics"""
    sentences = re.split(r'[.!?]+', text)
    words = re.findall(r'\b\w+\b', text.lower())
    syllables = sum(len(re.findall(r'[aeiouy]+', word)) for word in words)
    
    # Flesch Reading Ease (simplified)
    if len(sentences) > 0 and len(words) > 0:
        flesch_score = 206.835 - (1.015 * (len(words) / len(sentences))) - (84.6 * (syllables / len(words)))
        flesch_score = max(0, min(100, flesch_score))
    else:
        flesch_score = 0
    
    # Flesch-Kincaid Grade Level (simplified)
    if len(sentences) > 0 and len(words) > 0:
        fk_grade = 0.39 * (len(words) / len(sentences)) + 11.8 * (syllables / len(words)) - 15.59
        fk_grade = max(0, fk_grade)
    else:
        fk_grade = 0
    
    return {
        'flesch_reading_ease': flesch_score,
        'flesch_kincaid_grade': fk_grade,
        'syllable_count': syllables,
        'complexity_score': (len(words) / max(len(sentences), 1)) * (syllables / max(len(words), 1))
    }

def extract_features(resume_text: str, job_text: str) -> Dict[str, Any]:
    """Extract comprehensive features from resume and job description"""
    
    # Extract skills
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_text)
    
    # Calculate similarities
    similarities = {}
    
    # TF-IDF similarity
    try:
        tfidf_matrix = tfidf_vectorizer.fit_transform([resume_text, job_text])
        tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        similarities['tfidf_similarity'] = tfidf_similarity
    except:
        similarities['tfidf_similarity'] = 0.0
    
    # Sentence transformer similarity
    if sentence_model is not None:
        try:
            resume_embedding = sentence_model.encode(resume_text)
            job_embedding = sentence_model.encode(job_text)
            semantic_similarity = cosine_similarity([resume_embedding], [job_embedding])[0][0]
            similarities['semantic_similarity'] = semantic_similarity
        except:
            similarities['semantic_similarity'] = 0.0
    else:
        similarities['semantic_similarity'] = 0.0
    
    # Skill matching features
    skill_features = {
        'technical_skill_match': len(set(resume_skills['technical_skills']) & set(job_skills['technical_skills'])),
        'soft_skill_match': len(set(resume_skills['soft_skills']) & set(job_skills['soft_skills'])),
        'programming_language_match': len(set(resume_skills['programming_languages']) & set(job_skills['programming_languages'])),
        'framework_match': len(set(resume_skills['frameworks']) & set(job_skills['frameworks'])),
        'total_skill_match': len(set(resume_skills['all_skills']) & set(job_skills['all_skills'])),
        'skill_coverage': len(set(resume_skills['all_skills']) & set(job_skills['all_skills'])) / max(len(job_skills['all_skills']), 1)
    }
    
    # Text statistics
    resume_stats = calculate_text_statistics(resume_text)
    job_stats = calculate_text_statistics(job_text)
    
    text_features = {
        'resume_length': resume_stats['character_count'],
        'job_length': job_stats['character_count'],
        'length_ratio': resume_stats['character_count'] / max(job_stats['character_count'], 1),
        'resume_words': resume_stats['word_count'],
        'job_words': job_stats['word_count'],
        'resume_sentences': resume_stats['sentence_count'],
        'job_sentences': job_stats['sentence_count'],
        'resume_skill_density': resume_stats['skill_density'],
        'job_skill_density': job_stats['skill_density']
    }
    
    # Combine all features
    all_features = {
        **similarities,
        **skill_features,
        **text_features
    }
    
    return {
        'features': all_features,
        'resume_skills': resume_skills,
        'job_skills': job_skills,
        'resume_stats': resume_stats,
        'job_stats': job_stats
    }

def predict_match_score(features: Dict[str, Any]) -> Dict[str, Any]:
    """Predict match score using ensemble approach"""
    
    # Create feature vector
    feature_values = list(features.values())
    feature_array = np.array(feature_values).reshape(1, -1)
    
    # Simple ensemble scoring
    scores = {}
    
    # TF-IDF similarity score
    scores['tfidf_score'] = features.get('tfidf_similarity', 0.0)
    
    # Semantic similarity score
    scores['semantic_score'] = features.get('semantic_similarity', 0.0)
    
    # Skill matching score
    skill_coverage = features.get('skill_coverage', 0.0)
    scores['skill_score'] = min(skill_coverage * 2, 1.0)  # Scale up skill coverage
    
    # Text quality score
    length_ratio = features.get('length_ratio', 1.0)
    length_score = 1.0 - abs(length_ratio - 1.0) * 0.5  # Penalize extreme length differences
    scores['text_quality_score'] = max(0.0, length_score)
    
    # Ensemble score (weighted average)
    weights = {
        'tfidf_score': 0.25,
        'semantic_score': 0.35,
        'skill_score': 0.30,
        'text_quality_score': 0.10
    }
    
    ensemble_score = sum(scores[key] * weights[key] for key in weights.keys())
    
    return {
        'ensemble_score': ensemble_score,
        'individual_scores': scores,
        'feature_importance': weights
    }

def clean_skills_list(skills_list: List[str]) -> List[str]:
    """Clean and filter skills list to remove invalid entries"""
    if not skills_list:
        return []
    
    # Filter out single letters, very short words, and common stop words
    stop_words = {'the', 'and', 'for', 'with', 'from', 'this', 'that', 'have', 'been', 'will', 'they', 'their', 'there', 'are', 'was', 'were', 'has', 'had', 'can', 'may', 'would', 'could', 'should'}
    
    cleaned_skills = []
    for skill in skills_list:
        skill_lower = skill.lower().strip()
        # Only include skills that are at least 3 characters and not stop words
        if len(skill_lower) >= 3 and skill_lower not in stop_words:
            cleaned_skills.append(skill)
    
    return cleaned_skills

def generate_feedback(feature_data: Dict, match_score: float) -> Dict[str, Any]:
    """Generate detailed feedback based on analysis"""
    
    resume_skills = feature_data['resume_skills']
    job_skills = feature_data['job_skills']
    features = feature_data['features']
    
    feedback = {
        'overall_assessment': get_overall_assessment(match_score),
        'strengths': [],
        'weaknesses': [],
        'recommendations': [],
        'skill_gaps': [],
        'improvement_areas': []
    }
    
    # Clean skills before analysis
    resume_all_skills = clean_skills_list(resume_skills['all_skills'])
    job_all_skills = clean_skills_list(job_skills['all_skills'])
    
    # Analyze skill gaps
    missing_skills = set(job_all_skills) - set(resume_all_skills)
    if missing_skills:
        feedback['skill_gaps'] = list(missing_skills)[:10]
    
    # Analyze strengths
    matched_skills = set(resume_all_skills) & set(job_all_skills)
    if matched_skills:
        feedback['strengths'] = list(matched_skills)[:10]
    
    # Generate recommendations
    if match_score < 0.3:
        feedback['recommendations'].append("Consider adding more relevant skills and experience to your resume")
        feedback['recommendations'].append("Focus on highlighting achievements that align with the job requirements")
    elif match_score < 0.6:
        feedback['recommendations'].append("Focus on highlighting relevant achievements and quantifiable results")
        feedback['recommendations'].append("Consider customizing your resume further for this specific role")
    else:
        feedback['recommendations'].append("Strong match! Consider customizing your resume further for this specific role")
        feedback['recommendations'].append("Highlight specific achievements that demonstrate the required skills")
    
    # Analyze text quality
    if features.get('resume_length', 0) < 500:
        feedback['recommendations'].append("Consider expanding your resume with more detailed descriptions")
    
    if features.get('skill_coverage', 0) < 0.3:
        feedback['recommendations'].append("Add more relevant skills and keywords from the job description")
    
    return feedback

def get_overall_assessment(match_score: float) -> str:
    """Get overall assessment based on match score"""
    if match_score >= 0.8:
        return "Excellent Match - Strong alignment with job requirements"
    elif match_score >= 0.6:
        return "Good Match - Good alignment with some room for improvement"
    elif match_score >= 0.4:
        return "Moderate Match - Some alignment but significant gaps exist"
    elif match_score >= 0.2:
        return "Weak Match - Limited alignment with job requirements"
    else:
        return "Poor Match - Very limited alignment with job requirements"

# Health check endpoint
@app.get("/")
async def root():
    return {
        "message": "Advanced Resume-Job Matching API",
        "version": "2.0.0",
        "status": "operational",
        "models_loaded": sentence_model is not None
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": sentence_model is not None,
        "api_version": "2.0.0"
    }

@app.post("/api/v2/match", response_model=MatchResponse)
async def advanced_match_resume_job(request: MatchRequest):
    """Advanced resume-job matching with comprehensive analysis"""
    start_time = datetime.now()
    
    try:
        # Extract features
        feature_data = extract_features(request.resume, request.job_description)
        
        # Predict match score
        prediction_result = predict_match_score(feature_data['features'])
        match_score = prediction_result['ensemble_score']
        
        # Generate feedback
        feedback = generate_feedback(feature_data, match_score)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Clean skills for response
        cleaned_resume_skills = {k: clean_skills_list(v) if isinstance(v, list) else v 
                               for k, v in feature_data['resume_skills'].items()}
        cleaned_job_skills = {k: clean_skills_list(v) if isinstance(v, list) else v 
                            for k, v in feature_data['job_skills'].items()}
        
        # Prepare response
        response = MatchResponse(
            match_score=match_score,
            overall_assessment=feedback['overall_assessment'],
            skill_analysis={
                'resume_skills': cleaned_resume_skills,
                'job_skills': cleaned_job_skills,
                'skill_matches': list(set(cleaned_resume_skills['all_skills']) & 
                                    set(cleaned_job_skills['all_skills'])),
                'missing_skills': list(set(cleaned_job_skills['all_skills']) - 
                                     set(cleaned_resume_skills['all_skills']))
            },
            detailed_feedback=feedback,
            model_predictions={k: float(v) for k, v in prediction_result['individual_scores'].items()},
            feature_analysis={k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in feature_data['features'].items()},
            confidence_metrics={
                'model_agreement': float(np.std(list(prediction_result['individual_scores'].values()))),
                'feature_importance': {k: float(v) for k, v in prediction_result['feature_importance'].items()}
            },
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Match analysis completed in {processing_time:.2f}s with score {match_score:.3f}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in advanced matching: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing match request: {str(e)}"
        )

@app.post("/api/v2/analyze-skills", response_model=SkillAnalysisResponse)
async def analyze_skills(request: SkillAnalysisRequest):
    """Extract and analyze skills from text"""
    try:
        skills = extract_skills(request.text)
        
        return SkillAnalysisResponse(
            technical_skills=skills['technical_skills'],
            soft_skills=skills['soft_skills'],
            programming_languages=skills['programming_languages'],
            frameworks=skills['frameworks'],
            all_skills=skills['all_skills'],
            keyword_extraction={
                'tfidf_keywords': skills['tfidf_keywords'],
                'regex_keywords': skills['regex_keywords']
            }
        )
        
    except Exception as e:
        logger.error(f"Error in skill analysis: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing skills: {str(e)}"
        )

@app.post("/api/v2/analyze-text", response_model=TextAnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    """Comprehensive text analysis"""
    try:
        text_stats = calculate_text_statistics(request.text)
        readability_metrics = calculate_readability_metrics(request.text)
        
        # Calculate keyword density
        skills = extract_skills(request.text)
        skill_density = text_stats['skill_density']
        
        return TextAnalysisResponse(
            text_statistics=text_stats,
            readability_metrics=readability_metrics,
            keyword_density={
                'skill_density': skill_density,
                'technical_skill_density': len(skills['technical_skills']) / max(text_stats['word_count'], 1),
                'soft_skill_density': len(skills['soft_skills']) / max(text_stats['word_count'], 1)
            }
        )
        
    except Exception as e:
        logger.error(f"Error in text analysis: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing text: {str(e)}"
        )

@app.get("/api/v2/models/info")
async def get_model_info():
    """Get information about loaded models"""
    try:
        model_info = {
            "sentence_transformer_loaded": sentence_model is not None,
            "tfidf_vectorizer_loaded": tfidf_vectorizer is not None,
            "random_forest_loaded": random_forest is not None,
            "skill_databases": {
                "technical_skills": len(TECHNICAL_SKILLS),
                "soft_skills": len(SOFT_SKILLS),
                "programming_languages": len(PROGRAMMING_LANGUAGES),
                "frameworks": len(FRAMEWORKS)
            },
            "api_version": "2.0.0"
        }
        
        return model_info
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving model information: {str(e)}"
        )

@app.get("/api/v2/features/available")
async def get_available_features():
    """Get list of available analysis features"""
    return {
        "features": [
            {
                "name": "Advanced Matching",
                "endpoint": "/api/v2/match",
                "description": "Comprehensive resume-job matching with multiple ML models"
            },
            {
                "name": "Skill Analysis",
                "endpoint": "/api/v2/analyze-skills",
                "description": "Extract and analyze skills from text"
            },
            {
                "name": "Text Analysis",
                "endpoint": "/api/v2/analyze-text",
                "description": "Comprehensive text analysis including readability and statistics"
            },
            {
                "name": "Model Information",
                "endpoint": "/api/v2/models/info",
                "description": "Get information about loaded ML models"
            }
        ],
        "analysis_levels": [
            "basic",
            "detailed", 
            "comprehensive"
        ]
    }

@app.post("/api/v2/match/file")
async def match_resume_file(
    resume_file: UploadFile = File(None),
    resume: str = Form(None),
    job_description: str = Form(...),
    analysis_level: str = Form("comprehensive")
):
    """Match resume from uploaded file or text with job description"""
    try:
        # Handle both file upload and text input
        if resume_file:
            # Read the uploaded file
            content = await resume_file.read()
            # Extract text from the file using the utility function
            resume_text = extract_text_from_file(resume_file.filename, content)
        elif resume:
            # Use the provided text
            resume_text = resume
        else:
            raise HTTPException(
                status_code=400,
                detail="Either resume_file or resume text must be provided"
            )
        
        # Use the existing match endpoint logic
        start_time = datetime.now()
        
        # Extract features
        feature_data = extract_features(resume_text, job_description)
        
        # Predict match score
        prediction_result = predict_match_score(feature_data['features'])
        match_score = prediction_result['ensemble_score']
        
        # Generate feedback
        feedback = generate_feedback(feature_data, match_score)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Return response
        return {
            'match_score': match_score,
            'overall_assessment': feedback['overall_assessment'],
            'skill_analysis': {
                'resume_skills': feature_data['resume_skills'],
                'job_skills': feature_data['job_skills'],
                'skill_matches': list(set(feature_data['resume_skills']['all_skills']) & 
                                    set(feature_data['job_skills']['all_skills'])),
                'missing_skills': list(set(feature_data['job_skills']['all_skills']) - 
                                     set(feature_data['resume_skills']['all_skills']))
            },
            'detailed_feedback': feedback,
            'model_predictions': {k: float(v) for k, v in prediction_result['individual_scores'].items()},
            'feature_analysis': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in feature_data['features'].items()},
            'confidence_metrics': {
                'model_agreement': float(np.std(list(prediction_result['individual_scores'].values()))),
                'feature_importance': {k: float(v) for k, v in prediction_result['feature_importance'].items()}
            },
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat(),
            'file_name': resume_file.filename if resume_file else None
        }
        
    except Exception as e:
        logger.error(f"Error in file matching: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )

class BatchMatchRequest(BaseModel):
    resumes: List[str]
    job_description: str
    analysis_level: str = "comprehensive"

@app.post("/api/v2/batch-match")
async def batch_match_resumes(request: BatchMatchRequest):
    """Match multiple resumes with a job description"""
    try:
        results = []
        start_time = datetime.now()
        
        for i, resume in enumerate(request.resumes):
            try:
                # Extract features
                feature_data = extract_features(resume, request.job_description)
                
                # Predict match score
                prediction_result = predict_match_score(feature_data['features'])
                match_score = prediction_result['ensemble_score']
                
                # Generate feedback
                feedback = generate_feedback(feature_data, match_score)
                
                results.append({
                    'resume_index': i,
                    'match_score': float(match_score),
                    'overall_assessment': feedback['overall_assessment'],
                    'skill_matches': list(set(feature_data['resume_skills']['all_skills']) & 
                                        set(feature_data['job_skills']['all_skills'])),
                    'missing_skills': list(set(feature_data['job_skills']['all_skills']) - 
                                         set(feature_data['resume_skills']['all_skills'])),
                    'detailed_feedback': feedback,
                    'model_predictions': {k: float(v) for k, v in prediction_result['individual_scores'].items()}
                })
                
            except Exception as e:
                logger.error(f"Error processing resume {i}: {str(e)}")
                results.append({
                    'resume_index': i,
                    'error': str(e),
                    'match_score': 0.0
                })
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'batch_results': results,
            'total_resumes': len(request.resumes),
            'successful_matches': len([r for r in results if 'error' not in r]),
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in batch matching: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing batch request: {str(e)}"
        )

@app.post("/api/signup")
async def signup(
    email: str = Form(...), 
    password: str = Form(...), 
    user_type: str = Form("jobseeker")
):
    """User signup endpoint"""
    try:
        db = SessionLocal()
        
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == email).first()
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="User with this email already exists"
            )
        
        # Validate user_type
        if user_type not in ["jobseeker", "recruiter"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid user type. Must be 'jobseeker' or 'recruiter'"
            )
        
        # Create new user
        hashed_password = hash_password(password)
        new_user = User(
            email=email, 
            hashed_password=hashed_password,
            user_type=user_type
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        # Create token
        token = create_token(email, new_user.id, user_type)
        
        logger.info(f"New user created: {email} ({user_type})")
        
        return {
            "token": token,
            "email": email,
            "user_type": user_type,
            "user_id": new_user.id,
            "message": "Signup successful"
        }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signup error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )
    finally:
        db.close()

@app.post("/api/login")
async def login(email: str = Form(...), password: str = Form(...)):
    """User login endpoint"""
    try:
        db = SessionLocal()
        user = db.query(User).filter(User.email == email).first()
        
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid email or password"
            )
        
        # Verify password
        if verify_password(password, user.hashed_password):
            # Update last login
            user.last_login = datetime.utcnow()
            db.commit()
            
            # Create token
            token = create_token(email, user.id, user.user_type)
            
            return {
                "token": token,
                "email": email,
                "user_type": user.user_type,
                "user_id": user.id,
                "message": "Login successful"
            }
        else:
            raise HTTPException(
                status_code=401,
                detail="Invalid email or password"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )
    finally:
        db.close()

@app.get("/api/user/profile")
async def get_user_profile(token: str):
    """Get user profile from token"""
    try:
        # Verify token
        payload = verify_token(token)
        if not payload:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token"
            )
        
        db = SessionLocal()
        user = db.query(User).filter(User.id == payload.get("user_id")).first()
        
        if not user:
            raise HTTPException(
                status_code=404,
                detail="User not found"
            )
        
        return {
            "email": user.email,
            "user_type": user.user_type,
            "user_id": user.id,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "last_login": user.last_login.isoformat() if user.last_login else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )
    finally:
        db.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
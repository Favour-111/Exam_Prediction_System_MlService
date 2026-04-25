"""
Data Processor Module
Handles preprocessing of questions and topics for ML model
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import logging

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    logger.warning(f"NLTK download warning: {e}")


class DataProcessor:
    """Processes raw data for ML model training and prediction"""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
        
        # Difficulty and type mappings
        self.difficulty_map = {'Easy': 0, 'Medium': 1, 'Hard': 2}
        self.type_map = {
            'Theory': 0, 
            'Objective': 1, 
            'Calculation': 2, 
            'Practical': 3, 
            'Case Study': 4
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text: str) -> str:
        """Tokenize and lemmatize text"""
        try:
            tokens = word_tokenize(text)
            # Remove stopwords and lemmatize
            tokens = [
                self.lemmatizer.lemmatize(token) 
                for token in tokens 
                if token not in self.stop_words and len(token) > 2
            ]
            return ' '.join(tokens)
        except Exception as e:
            logger.warning(f"Tokenization error: {e}")
            return text
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """Extract features from question text"""
        features = {
            'word_count': len(text.split()),
            'char_count': len(text),
            'has_calculation_keywords': self._has_calculation_keywords(text),
            'has_theory_keywords': self._has_theory_keywords(text),
            'has_objective_keywords': self._has_objective_keywords(text),
            'complexity_score': self._calculate_complexity(text)
        }
        return features

    def _note_priority_score(self, note_text: str, question_text: str) -> float:
        """Measure overlap between lecturer notes and question text"""
        if not note_text or not question_text:
            return 0.0

        note_tokens = set(self.clean_text(note_text).split())
        question_tokens = set(self.clean_text(question_text).split())
        if not note_tokens:
            return 0.0

        return min(len(note_tokens.intersection(question_tokens)) / len(note_tokens), 1.0)
    
    def _has_calculation_keywords(self, text: str) -> int:
        """Check for calculation-related keywords"""
        keywords = ['calculate', 'compute', 'solve', 'find', 'determine', 
                   'formula', 'equation', 'value', 'numerical']
        text_lower = text.lower()
        return 1 if any(kw in text_lower for kw in keywords) else 0
    
    def _has_theory_keywords(self, text: str) -> int:
        """Check for theory-related keywords"""
        keywords = ['explain', 'describe', 'discuss', 'define', 'what is',
                   'elaborate', 'state', 'list', 'differentiate']
        text_lower = text.lower()
        return 1 if any(kw in text_lower for kw in keywords) else 0
    
    def _has_objective_keywords(self, text: str) -> int:
        """Check for objective/MCQ related keywords"""
        keywords = ['which of', 'select', 'choose', 'following', 'correct',
                   'true', 'false', 'option']
        text_lower = text.lower()
        return 1 if any(kw in text_lower for kw in keywords) else 0
    
    def _calculate_complexity(self, text: str) -> float:
        """Calculate text complexity score"""
        words = text.split()
        if not words:
            return 0.0
        
        # Simple complexity based on word length and count
        avg_word_length = sum(len(w) for w in words) / len(words)
        word_count = len(words)
        
        # Normalize to 0-1 scale
        complexity = min((avg_word_length / 10) + (word_count / 100), 1.0)
        return complexity
    
    def process_training_data(
        self, 
        questions: List[Dict[str, Any]], 
        topics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process training data for model"""
        logger.info(f"Processing {len(questions)} questions and {len(topics)} topics")
        
        processed_questions = []
        
        for q in questions:
            cleaned_text = self.clean_text(q.get('text', ''))
            lecturer_notes = q.get('lecturer_notes', '')
            cleaned_notes = self.clean_text(lecturer_notes)
            processed_text = self.tokenize_and_lemmatize(
                f"{cleaned_text} {cleaned_notes}".strip()
            )
            features = self.extract_features(q.get('text', ''))
            
            processed_q = {
                'text': processed_text,
                'original_text': q.get('text', ''),
                'topic': q.get('topic', 'Unknown'),
                'topic_emphasis': q.get('topic_emphasis', 5),
                'lecturer_note_strength': self._note_priority_score(lecturer_notes, q.get('text', '')),
                'difficulty': self.difficulty_map.get(q.get('difficulty', 'Medium'), 1),
                'question_type': self.type_map.get(q.get('question_type', 'Theory'), 0),
                'occurrence_count': q.get('occurrence_count', 1),
                'year': q.get('year', 2024),
                **features
            }
            processed_questions.append(processed_q)
        
        # Process topics
        processed_topics = []
        for t in topics:
            processed_t = {
                'name': t.get('name', ''),
                'emphasis': t.get('emphasis', 5),
                'frequency': t.get('frequency', 0),
                'keywords': t.get('keywords', []),
                'lecturer_notes': t.get('lecturer_notes', '')
            }
            processed_topics.append(processed_t)
        
        # Create TF-IDF features from question texts
        texts = [q['text'] for q in processed_questions]
        if texts:
            try:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            except:
                tfidf_matrix = None
        else:
            tfidf_matrix = None
        
        return {
            'questions': processed_questions,
            'topics': processed_topics,
            'tfidf_matrix': tfidf_matrix,
            'feature_names': self.tfidf_vectorizer.get_feature_names_out() if tfidf_matrix is not None else []
        }
    
    def process_prediction_data(
        self, 
        questions: List[Dict[str, Any]], 
        topics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process data for prediction"""
        logger.info(f"Processing {len(questions)} questions for prediction")
        
        processed_questions = []
        
        for q in questions:
            cleaned_text = self.clean_text(q.get('text', ''))
            lecturer_notes = q.get('lecturer_notes', '')
            cleaned_notes = self.clean_text(lecturer_notes)
            processed_text = self.tokenize_and_lemmatize(
                f"{cleaned_text} {cleaned_notes}".strip()
            )
            features = self.extract_features(q.get('text', ''))
            
            processed_q = {
                'id': q.get('id'),
                'text': processed_text,
                'original_text': q.get('text', ''),
                'topic': q.get('topic', 'Unknown'),
                'topic_emphasis': q.get('topic_emphasis', 5),
                'lecturer_note_strength': self._note_priority_score(lecturer_notes, q.get('text', '')),
                'difficulty': self.difficulty_map.get(q.get('difficulty', 'Medium'), 1),
                'question_type': self.type_map.get(q.get('question_type', 'Theory'), 0),
                'occurrence_count': q.get('occurrence_count', 1),
                'year': q.get('year', 2024),
                **features
            }
            processed_questions.append(processed_q)
        
        # Process topics
        processed_topics = []
        for t in topics:
            processed_t = {
                'id': t.get('id'),
                'name': t.get('name', ''),
                'emphasis': t.get('emphasis', 5),
                'frequency': t.get('frequency', 0),
                'lecturer_notes': t.get('lecturer_notes', '')
            }
            processed_topics.append(processed_t)
        
        # Apply TF-IDF transformation if vectorizer is fitted
        texts = [q['text'] for q in processed_questions]
        try:
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        except:
            # If vectorizer not fitted, fit it first
            try:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            except:
                tfidf_matrix = None
        
        return {
            'questions': processed_questions,
            'topics': processed_topics,
            'tfidf_matrix': tfidf_matrix
        }
    
    def create_feature_matrix(
        self, 
        processed_data: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[str]]:
        """Create feature matrix for model training/prediction"""
        questions = processed_data['questions']
        
        # Numerical features
        numerical_features = []
        for q in questions:
            features = [
                q.get('topic_emphasis', 5) / 10,  # Normalize to 0-1
                q.get('occurrence_count', 1) / 10,  # Normalize
                q.get('difficulty', 1) / 2,  # Normalize to 0-1
                q.get('question_type', 0) / 4,  # Normalize to 0-1
                q.get('word_count', 0) / 100,  # Normalize
                q.get('complexity_score', 0),
                q.get('lecturer_note_strength', 0),
                q.get('has_calculation_keywords', 0),
                q.get('has_theory_keywords', 0),
                q.get('has_objective_keywords', 0)
            ]
            numerical_features.append(features)
        
        numerical_matrix = np.array(numerical_features)
        
        # Combine with TF-IDF features if available
        tfidf_matrix = processed_data.get('tfidf_matrix')
        if tfidf_matrix is not None:
            tfidf_array = tfidf_matrix.toarray()
            feature_matrix = np.hstack([numerical_matrix, tfidf_array])
        else:
            feature_matrix = numerical_matrix
        
        feature_names = [
            'topic_emphasis', 'occurrence_count', 'difficulty', 
            'question_type', 'word_count', 'complexity_score',
            'lecturer_note_strength',
            'has_calculation_keywords', 'has_theory_keywords', 
            'has_objective_keywords'
        ]
        
        return feature_matrix, feature_names

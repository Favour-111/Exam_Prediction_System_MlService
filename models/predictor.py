"""
Exam Predictor Module
Random Forest based prediction model for exam questions
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ExamPredictor:
    """
    Random Forest based predictor for exam questions.
    Predicts probability of topics/questions appearing in exams.
    """
    
    def __init__(self, model_dir: str = "saved_models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize models
        self.topic_classifier = None
        self.probability_regressor = None
        self.type_classifier = None
        self.difficulty_classifier = None
        
        self.model_version = "1.0.0"
        self.model_accuracy = None
        self.feature_names = []
        self._is_loaded = False
        
        # Try to load existing model
        self._load_model()
    
    def _load_model(self) -> bool:
        """Load saved model if exists"""
        model_path = os.path.join(self.model_dir, "exam_predictor.pkl")
        if os.path.exists(model_path):
            try:
                saved_data = joblib.load(model_path)
                self.topic_classifier = saved_data.get('topic_classifier')
                self.probability_regressor = saved_data.get('probability_regressor')
                self.type_classifier = saved_data.get('type_classifier')
                self.difficulty_classifier = saved_data.get('difficulty_classifier')
                self.model_version = saved_data.get('version', '1.0.0')
                self.model_accuracy = saved_data.get('accuracy')
                self.feature_names = saved_data.get('feature_names', [])
                self._is_loaded = True
                logger.info(f"Loaded model version {self.model_version}")
                return True
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                return False
        return False
    
    def _save_model(self):
        """Save model to disk"""
        model_path = os.path.join(self.model_dir, "exam_predictor.pkl")
        try:
            joblib.dump({
                'topic_classifier': self.topic_classifier,
                'probability_regressor': self.probability_regressor,
                'type_classifier': self.type_classifier,
                'difficulty_classifier': self.difficulty_classifier,
                'version': self.model_version,
                'accuracy': self.model_accuracy,
                'feature_names': self.feature_names,
                'saved_at': datetime.now().isoformat()
            }, model_path)
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._is_loaded or self.probability_regressor is not None
    
    def get_model_version(self) -> str:
        """Get current model version"""
        return self.model_version
    
    def get_model_accuracy(self) -> Optional[float]:
        """Get model accuracy"""
        return self.model_accuracy
    
    def get_feature_names(self) -> List[str]:
        """Get feature names used in model"""
        return self.feature_names
    
    def train(self, processed_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Train the prediction model using Random Forest
        
        Args:
            processed_data: Preprocessed training data
            
        Returns:
            Training metrics dictionary
        """
        logger.info("Starting model training...")
        
        questions = processed_data['questions']
        
        if len(questions) < 5:
            raise ValueError("Insufficient data for training. Need at least 5 questions.")
        
        # Prepare features
        from models.data_processor import DataProcessor
        processor = DataProcessor()
        X, self.feature_names = processor.create_feature_matrix(processed_data)
        
        # Create target variables
        # For probability prediction, we use occurrence count as a proxy
        y_prob = np.array([
            min(q.get('occurrence_count', 1) / max(q.get('occurrence_count', 1) for q in questions), 1.0)
            for q in questions
        ])
        
        # For type classification
        y_type = np.array([q.get('question_type', 0) for q in questions])
        
        # For difficulty classification
        y_difficulty = np.array([q.get('difficulty', 1) for q in questions])
        
        # Split data
        X_train, X_test, y_prob_train, y_prob_test = train_test_split(
            X, y_prob, test_size=0.2, random_state=42
        )
        
        _, _, y_type_train, y_type_test = train_test_split(
            X, y_type, test_size=0.2, random_state=42
        )
        
        _, _, y_diff_train, y_diff_test = train_test_split(
            X, y_difficulty, test_size=0.2, random_state=42
        )
        
        # Train probability regressor
        self.probability_regressor = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        self.probability_regressor.fit(X_train, y_prob_train)
        
        # Train type classifier
        if len(np.unique(y_type)) > 1:
            self.type_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            )
            self.type_classifier.fit(X_train, y_type_train)
        
        # Train difficulty classifier
        if len(np.unique(y_difficulty)) > 1:
            self.difficulty_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            )
            self.difficulty_classifier.fit(X_train, y_diff_train)
        
        # Calculate metrics
        prob_pred = self.probability_regressor.predict(X_test)
        
        # Convert to binary for classification metrics
        prob_binary = (y_prob_test > 0.5).astype(int)
        prob_pred_binary = (prob_pred > 0.5).astype(int)
        
        metrics = {
            'accuracy': float(accuracy_score(prob_binary, prob_pred_binary)),
            'r2_score': float(self.probability_regressor.score(X_test, y_prob_test))
        }
        
        if self.type_classifier:
            type_pred = self.type_classifier.predict(X_test)
            metrics['type_accuracy'] = float(accuracy_score(y_type_test, type_pred))
        
        if self.difficulty_classifier:
            diff_pred = self.difficulty_classifier.predict(X_test)
            metrics['difficulty_accuracy'] = float(accuracy_score(y_diff_test, diff_pred))
        
        # Cross-validation score
        cv_scores = cross_val_score(
            self.probability_regressor, X, y_prob, 
            cv=min(5, len(questions)), scoring='r2'
        )
        metrics['cv_mean'] = float(np.mean(cv_scores))
        metrics['cv_std'] = float(np.std(cv_scores))
        
        # Update model info
        self.model_accuracy = metrics['accuracy']
        self.model_version = f"1.0.{datetime.now().strftime('%Y%m%d%H%M')}"
        self._is_loaded = True
        
        # Save model
        self._save_model()
        
        logger.info(f"Training completed. Accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
    
    def predict(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate predictions for questions and topics
        
        Args:
            processed_data: Preprocessed prediction data
            
        Returns:
            Prediction results
        """
        logger.info("Generating predictions...")
        
        questions = processed_data['questions']
        topics = processed_data['topics']
        
        # Prepare features
        from models.data_processor import DataProcessor
        processor = DataProcessor()
        
        results = {
            'topic_probabilities': {},
            'question_probabilities': {},
            'predicted_types': {},
            'predicted_difficulties': {},
            'type_distribution': {},
            'difficulty_distribution': {}
        }
        
        if not questions:
            return results
        
        try:
            X, _ = processor.create_feature_matrix(processed_data)
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            # Return default probabilities
            for q in questions:
                q_id = q.get('id', str(id(q)))
                results['question_probabilities'][q_id] = 0.5
                results['predicted_types'][q_id] = 'Theory'
                results['predicted_difficulties'][q_id] = 'Medium'
            for t in topics:
                t_id = t.get('id', str(id(t)))
                results['topic_probabilities'][t_id] = 0.5
            return results
        
        # Predict probabilities
        if self.probability_regressor:
            probabilities = self.probability_regressor.predict(X)
            probabilities = np.clip(probabilities, 0, 1)
        else:
            # Use heuristic-based prediction if model not trained
            probabilities = self._heuristic_probabilities(questions)
        
        # Map predictions to question IDs
        type_map_reverse = {0: 'Theory', 1: 'Objective', 2: 'Calculation', 
                           3: 'Practical', 4: 'Case Study'}
        difficulty_map_reverse = {0: 'Easy', 1: 'Medium', 2: 'Hard'}
        
        for i, q in enumerate(questions):
            q_id = q.get('id', str(i))
            results['question_probabilities'][q_id] = float(probabilities[i])
            
            # Predict type
            if self.type_classifier:
                type_pred = self.type_classifier.predict(X[i:i+1])[0]
                results['predicted_types'][q_id] = type_map_reverse.get(type_pred, 'Theory')
            else:
                results['predicted_types'][q_id] = type_map_reverse.get(
                    q.get('question_type', 0), 'Theory'
                )
            
            # Predict difficulty
            if self.difficulty_classifier:
                diff_pred = self.difficulty_classifier.predict(X[i:i+1])[0]
                results['predicted_difficulties'][q_id] = difficulty_map_reverse.get(diff_pred, 'Medium')
            else:
                results['predicted_difficulties'][q_id] = difficulty_map_reverse.get(
                    q.get('difficulty', 1), 'Medium'
                )
        
        # Calculate topic probabilities
        topic_question_probs = {}
        for i, q in enumerate(questions):
            topic = q.get('topic', 'Unknown')
            if topic not in topic_question_probs:
                topic_question_probs[topic] = []
            topic_question_probs[topic].append(probabilities[i])
        
        # Map topic probabilities
        for t in topics:
            t_id = t.get('id', str(id(t)))
            t_name = t.get('name', '')
            
            if t_name in topic_question_probs:
                # Average probability of questions in this topic
                base_prob = np.mean(topic_question_probs[t_name])
            else:
                base_prob = 0.3
            
            # Factor in lecturer emphasis
            emphasis_factor = t.get('emphasis', 5) / 10
            frequency_factor = min(t.get('frequency', 0) / 10, 1)
            
            # Combined probability
            final_prob = (base_prob * 0.4 + emphasis_factor * 0.4 + frequency_factor * 0.2)
            results['topic_probabilities'][t_id] = float(min(final_prob, 1.0))
        
        # Calculate distributions
        types_count = {}
        difficulties_count = {}
        
        for q_id, q_type in results['predicted_types'].items():
            types_count[q_type] = types_count.get(q_type, 0) + 1
        
        for q_id, q_diff in results['predicted_difficulties'].items():
            difficulties_count[q_diff] = difficulties_count.get(q_diff, 0) + 1
        
        total = len(questions) or 1
        results['type_distribution'] = {
            k: v / total for k, v in types_count.items()
        }
        results['difficulty_distribution'] = {
            k: v / total for k, v in difficulties_count.items()
        }
        
        logger.info(f"Generated predictions for {len(questions)} questions")
        
        return results
    
    def _heuristic_probabilities(self, questions: List[Dict]) -> np.ndarray:
        """Generate heuristic-based probabilities when model is not trained"""
        probabilities = []
        
        for q in questions:
            # Base probability
            prob = 0.5
            
            # Factor in occurrence count
            occurrence = q.get('occurrence_count', 1)
            prob += min(occurrence / 20, 0.3)
            
            # Factor in topic emphasis
            emphasis = q.get('topic_emphasis', 5)
            prob += (emphasis - 5) / 20
            
            # Factor in complexity
            complexity = q.get('complexity_score', 0.5)
            prob += (complexity - 0.5) / 5
            
            # Ensure within bounds
            prob = max(0.1, min(0.95, prob))
            probabilities.append(prob)
        
        return np.array(probabilities)

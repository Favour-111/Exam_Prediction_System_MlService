"""
Exam Question Prediction System - ML Microservice
FastAPI-based machine learning service for predicting exam questions
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import logging

from models.predictor import ExamPredictor
from models.data_processor import DataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Exam Question Prediction API",
    description="ML microservice for predicting exam questions using Random Forest",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
predictor = ExamPredictor()
data_processor = DataProcessor()


# Pydantic models for request/response
class QuestionInput(BaseModel):
    id: str
    text: str
    topic: Optional[str] = None
    topic_emphasis: float = 5.0
    lecturer_notes: Optional[str] = ""
    difficulty: Optional[str] = "Medium"
    question_type: Optional[str] = "Theory"
    occurrence_count: int = 1
    year: Optional[int] = None


class TopicInput(BaseModel):
    id: str
    name: str
    emphasis: float = 5.0
    frequency: int = 0
    keywords: Optional[List[str]] = []
    lecturer_notes: Optional[str] = ""


class TrainingRequest(BaseModel):
    course_id: str
    questions: List[Dict[str, Any]]
    topics: List[Dict[str, Any]]


class PredictionRequest(BaseModel):
    course_id: str
    questions: List[QuestionInput]
    topics: List[TopicInput]


class TopicPrediction(BaseModel):
    topic_id: str
    topic_name: str
    probability: float
    confidence: str


class QuestionPrediction(BaseModel):
    question_id: str
    question_text: str
    probability: float
    predicted_type: Optional[str] = None
    predicted_difficulty: Optional[str] = None


class PredictionResponse(BaseModel):
    success: bool
    model_version: str
    model_accuracy: Optional[float] = None
    topic_predictions: List[TopicPrediction]
    question_predictions: List[QuestionPrediction]
    type_distribution: Dict[str, float]
    difficulty_distribution: Dict[str, float]
    insights: List[str]


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Exam Question Prediction ML Service",
        "version": "1.0.0"
    }


@app.get("/api/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "OK",
        "model_loaded": predictor.is_model_loaded(),
        "version": predictor.get_model_version()
    }


@app.post("/api/train")
async def train_model(request: TrainingRequest):
    """
    Train the prediction model with provided data
    """
    try:
        logger.info(f"Training model for course: {request.course_id}")
        
        # Process training data
        processed_data = data_processor.process_training_data(
            request.questions,
            request.topics
        )
        
        # Train the model
        metrics = predictor.train(processed_data)
        
        logger.info(f"Training completed with accuracy: {metrics.get('accuracy', 0):.4f}")
        
        return {
            "success": True,
            "message": "Model trained successfully",
            "metrics": metrics,
            "model_version": predictor.get_model_version()
        }
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict", response_model=PredictionResponse)
async def generate_predictions(request: PredictionRequest):
    """
    Generate predictions for topics and questions
    """
    try:
        logger.info(f"Generating predictions for course: {request.course_id}")
        
        # Process input data
        questions_data = [q.dict() for q in request.questions]
        topics_data = [t.dict() for t in request.topics]
        
        processed_data = data_processor.process_prediction_data(
            questions_data,
            topics_data
        )
        
        # Generate predictions
        predictions = predictor.predict(processed_data)
        
        # Format topic predictions
        topic_predictions = []
        for topic_data in topics_data:
            prob = predictions['topic_probabilities'].get(topic_data['id'], 0.5)
            confidence = "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
            topic_predictions.append(TopicPrediction(
                topic_id=topic_data['id'],
                topic_name=topic_data['name'],
                probability=prob,
                confidence=confidence
            ))
        
        # Sort by probability
        topic_predictions.sort(key=lambda x: x.probability, reverse=True)
        
        # Format question predictions
        question_predictions = []
        for q_data in questions_data:
            prob = predictions['question_probabilities'].get(q_data['id'], 0.5)
            question_predictions.append(QuestionPrediction(
                question_id=q_data['id'],
                question_text=q_data['text'][:200],  # Truncate for response
                probability=prob,
                predicted_type=predictions['predicted_types'].get(q_data['id']),
                predicted_difficulty=predictions['predicted_difficulties'].get(q_data['id'])
            ))
        
        # Sort by probability
        question_predictions.sort(key=lambda x: x.probability, reverse=True)
        
        # Generate insights
        insights = generate_insights(topic_predictions, question_predictions)
        
        logger.info(f"Generated {len(topic_predictions)} topic predictions and {len(question_predictions)} question predictions")
        
        return PredictionResponse(
            success=True,
            model_version=predictor.get_model_version(),
            model_accuracy=predictor.get_model_accuracy(),
            topic_predictions=topic_predictions,
            question_predictions=question_predictions,
            type_distribution=predictions.get('type_distribution', {}),
            difficulty_distribution=predictions.get('difficulty_distribution', {}),
            insights=insights
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model/info")
async def get_model_info():
    """Get information about the current model"""
    return {
        "version": predictor.get_model_version(),
        "accuracy": predictor.get_model_accuracy(),
        "is_loaded": predictor.is_model_loaded(),
        "features": predictor.get_feature_names()
    }


def generate_insights(
    topic_predictions: List[TopicPrediction],
    question_predictions: List[QuestionPrediction]
) -> List[str]:
    """Generate human-readable insights from predictions"""
    insights = []
    
    # Top topics insight
    if topic_predictions:
        top_topics = [tp.topic_name for tp in topic_predictions[:3]]
        insights.append(f"Focus on: {', '.join(top_topics)} - these have the highest probability of appearing")
    
    # High confidence topics
    high_conf = [tp for tp in topic_predictions if tp.confidence == "High"]
    if high_conf:
        insights.append(f"{len(high_conf)} topics have high prediction confidence")
    
    # Question type distribution
    theory_qs = [qp for qp in question_predictions if qp.predicted_type == "Theory"]
    calc_qs = [qp for qp in question_predictions if qp.predicted_type == "Calculation"]
    
    if theory_qs:
        insights.append(f"Expect approximately {len(theory_qs)} theory-based questions")
    if calc_qs:
        insights.append(f"Prepare for {len(calc_qs)} calculation-based questions")
    
    # Difficulty distribution
    hard_qs = [qp for qp in question_predictions if qp.predicted_difficulty == "Hard"]
    if hard_qs:
        insights.append(f"{len(hard_qs)} questions predicted to be difficult - allocate extra study time")
    
    return insights


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

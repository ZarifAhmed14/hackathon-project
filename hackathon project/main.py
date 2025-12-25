from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Optional, List
import joblib
import numpy as np
from datetime import datetime

# Import database components
from database import (
    get_db, init_db, DatabaseManager,
    FarmerInput as DBFarmerInput,
    Recommendation as DBRecommendation
)

# Import weather service
from weather import WeatherService, get_weather_for_location

app = FastAPI(title="Fertilizer & Irrigation API with Weather Integration")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
@app.on_event("startup")
def startup_event():
    init_db()
    print("🚀 Server started with database and weather API support")

# Input schemas
class FarmerInputRequest(BaseModel):
    crop_type: str
    soil_ph: float
    soil_moisture: Optional[float] = None  # Now optional
    temperature: Optional[float] = None  # Now optional
    rainfall: Optional[float] = None  # Now optional
    farmer_name: Optional[str] = None
    location: Optional[str] = None  # City name for weather API

class WeatherRequest(BaseModel):
    location: str  # City name or "lat,lon"

class PredictWithWeatherRequest(BaseModel):
    crop_type: str
    soil_ph: float
    location: str  # Auto-fetch weather data
    farmer_name: Optional[str] = None

class FeedbackRequest(BaseModel):
    recommendation_id: int
    rating: Optional[int] = None
    was_helpful: Optional[bool] = None
    comments: Optional[str] = None

# Response schemas
class RecommendationResponse(BaseModel):
    id: int
    fertilizer_level: str
    irrigation_needed: bool
    confidence: float
    recommendations_text: str
    farmer_input_id: int
    created_at: datetime

class HistoryResponse(BaseModel):
    id: int
    crop_type: str
    soil_ph: float
    temperature: float
    rainfall: float
    created_at: datetime

# Load models
try:
    fertilizer_model = joblib.load("fertilizer_model.pkl")
    irrigation_model = joblib.load("irrigation_model.pkl")
    crop_encoder = joblib.load("crop_encoder.pkl")
    print("✅ Models loaded successfully")
except FileNotFoundError:
    print("⚠️ Models not found. Using mock predictions.")
    fertilizer_model = None
    irrigation_model = None
    crop_encoder = None

CROP_ENCODING = {
    "rice": 0, "wheat": 1, "maize": 2, "potato": 3, "jute": 4,
    "cotton": 5, "coffee": 6, "sugarcane": 7
}

def preprocess_input(crop_type: str, soil_ph: float, soil_moisture: float, 
                     temperature: float, rainfall: float) -> np.ndarray:
    """Convert input to model features"""
    crop_encoded = CROP_ENCODING.get(crop_type.lower(), 0)
    return np.array([[crop_encoded, soil_ph, soil_moisture, temperature, rainfall]])

def generate_recommendation_text(fertilizer: str, irrigation: bool, 
                                crop_type: str, soil_moisture: float, rainfall: float) -> str:
    """Generate farmer-friendly advice"""
    advice = []
    if fertilizer == "High":
        advice.append(f"Apply HIGH fertilizer for {crop_type}.")
    elif fertilizer == "Medium":
        advice.append(f"Apply MEDIUM fertilizer for {crop_type}.")
    else:
        advice.append(f"Apply LOW fertilizer. Soil is balanced.")
    
    if irrigation:
        advice.append(f"Irrigation needed (moisture: {soil_moisture}%, rainfall: {rainfall}mm).")
    else:
        advice.append(f"No irrigation needed. Moisture adequate ({soil_moisture}%).")
    
    return " ".join(advice)

@app.get("/")
def home():
    return {
        "message": "Fertilizer & Irrigation API with Weather Integration",
        "version": "3.0",
        "endpoints": [
            "/predict", 
            "/predict-with-weather",
            "/weather",
            "/history", 
            "/feedback", 
            "/stats", 
            "/health"
        ]
    }

@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    """Health check with database status"""
    try:
        total_records = db.query(DBFarmerInput).count()
        return {
            "status": "healthy",
            "models_loaded": fertilizer_model is not None,
            "database_connected": True,
            "total_records": total_records,
            "weather_api": "OpenWeatherMap"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/weather")
def get_weather(request: WeatherRequest):
    """
    Get current weather data for a location
    """
    try:
        weather_data = get_weather_for_location(request.location)
        return weather_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-with-weather")
def predict_with_weather(data: PredictWithWeatherRequest, db: Session = Depends(get_db)):
    """
    Make prediction by auto-fetching weather data
    Farmer only needs to provide: crop, soil pH, and location
    """
    try:
        # Fetch weather data
        try:
            weather_data = get_weather_for_location(data.location)
        except Exception as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Could not fetch weather data: {str(e)}. Make sure OPENWEATHER_API_KEY is set."
            )
        
        # Use weather data to fill in missing parameters
        temperature = weather_data["temperature"]
        rainfall = weather_data["rainfall"]
        soil_moisture = weather_data["estimated_soil_moisture"]
        
        # Save input to database
        farmer_input_data = {
            'farmer_name': data.farmer_name,
            'location': data.location,
            'crop_type': data.crop_type,
            'soil_ph': data.soil_ph,
            'soil_moisture': soil_moisture,
            'temperature': temperature,
            'rainfall': rainfall
        }
        saved_input = DatabaseManager.save_farmer_input(db, farmer_input_data)
        
        # Make prediction
        features = preprocess_input(
            data.crop_type, data.soil_ph, soil_moisture, temperature, rainfall
        )
        
        if fertilizer_model and irrigation_model:
            fert_pred = fertilizer_model.predict(features)[0]
            fert_proba = fertilizer_model.predict_proba(features)[0]
            irrig_pred = irrigation_model.predict(features)[0]
            
            fertilizer_level = ["Low", "Medium", "High"][fert_pred]
            irrigation_needed = bool(irrig_pred)
            confidence = float(max(fert_proba))
        else:
            fertilizer_level = "Medium"
            irrigation_needed = soil_moisture < 40 or rainfall < 50
            confidence = 0.75
        
        recommendations_text = generate_recommendation_text(
            fertilizer_level, irrigation_needed, data.crop_type, soil_moisture, rainfall
        )
        
        # Save recommendation
        recommendation_data = {
            'fertilizer_level': fertilizer_level,
            'irrigation_needed': irrigation_needed,
            'confidence': confidence,
            'recommendations_text': recommendations_text
        }
        saved_rec = DatabaseManager.save_recommendation(
            db, saved_input.id, recommendation_data
        )
        
        return {
            "id": saved_rec.id,
            "fertilizer_level": fertilizer_level,
            "irrigation_needed": irrigation_needed,
            "confidence": confidence,
            "recommendations_text": recommendations_text,
            "weather_data": {
                "temperature": temperature,
                "rainfall": rainfall,
                "soil_moisture": soil_moisture,
                "humidity": weather_data["humidity"],
                "weather": weather_data["weather_description"]
            },
            "farmer_input_id": saved_input.id,
            "message": "Recommendation with live weather data"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict(data: FarmerInputRequest, db: Session = Depends(get_db)):
    """
    Make prediction with manual input or auto-fetch weather
    """
    try:
        # If location provided but weather data missing, fetch it
        if data.location and (data.temperature is None or data.rainfall is None):
            try:
                weather_data = get_weather_for_location(data.location)
                temperature = data.temperature or weather_data["temperature"]
                rainfall = data.rainfall or weather_data["rainfall"]
                soil_moisture = data.soil_moisture or weather_data["estimated_soil_moisture"]
            except:
                # If weather fetch fails, require manual input
                if data.temperature is None or data.rainfall is None or data.soil_moisture is None:
                    raise HTTPException(
                        status_code=400,
                        detail="Please provide temperature, rainfall, and soil_moisture manually, or ensure weather API is configured."
                    )
        else:
            # Use provided values
            if data.temperature is None or data.rainfall is None or data.soil_moisture is None:
                raise HTTPException(
                    status_code=400,
                    detail="Missing required fields: temperature, rainfall, soil_moisture"
                )
            temperature = data.temperature
            rainfall = data.rainfall
            soil_moisture = data.soil_moisture
        
        # Save input to database
        farmer_input_data = {
            'farmer_name': data.farmer_name,
            'location': data.location,
            'crop_type': data.crop_type,
            'soil_ph': data.soil_ph,
            'soil_moisture': soil_moisture,
            'temperature': temperature,
            'rainfall': rainfall
        }
        saved_input = DatabaseManager.save_farmer_input(db, farmer_input_data)
        
        # Make prediction
        features = preprocess_input(
            data.crop_type, data.soil_ph, soil_moisture, temperature, rainfall
        )
        
        if fertilizer_model and irrigation_model:
            fert_pred = fertilizer_model.predict(features)[0]
            fert_proba = fertilizer_model.predict_proba(features)[0]
            irrig_pred = irrigation_model.predict(features)[0]
            
            fertilizer_level = ["Low", "Medium", "High"][fert_pred]
            irrigation_needed = bool(irrig_pred)
            confidence = float(max(fert_proba))
        else:
            fertilizer_level = "Medium"
            irrigation_needed = soil_moisture < 40 or rainfall < 50
            confidence = 0.75
        
        recommendations_text = generate_recommendation_text(
            fertilizer_level, irrigation_needed, data.crop_type, soil_moisture, rainfall
        )
        
        # Save recommendation
        recommendation_data = {
            'fertilizer_level': fertilizer_level,
            'irrigation_needed': irrigation_needed,
            'confidence': confidence,
            'recommendations_text': recommendations_text
        }
        saved_rec = DatabaseManager.save_recommendation(
            db, saved_input.id, recommendation_data
        )
        
        return {
            "id": saved_rec.id,
            "fertilizer_level": fertilizer_level,
            "irrigation_needed": irrigation_needed,
            "confidence": confidence,
            "recommendations_text": recommendations_text,
            "farmer_input_id": saved_input.id,
            "message": "Recommendation saved to database"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history", response_model=List[HistoryResponse])
def get_history(limit: int = 10, db: Session = Depends(get_db)):
    """Get recent farmer input history"""
    try:
        history = DatabaseManager.get_farmer_history(db, limit)
        return [
            {
                "id": h.id,
                "crop_type": h.crop_type,
                "soil_ph": h.soil_ph,
                "temperature": h.temperature,
                "rainfall": h.rainfall,
                "created_at": h.created_at
            }
            for h in history
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommendations", response_model=List[RecommendationResponse])
def get_recommendations(limit: int = 10, db: Session = Depends(get_db)):
    """Get recent recommendations"""
    try:
        recs = DatabaseManager.get_recommendations_history(db, limit)
        return [
            {
                "id": r.id,
                "fertilizer_level": r.fertilizer_level,
                "irrigation_needed": r.irrigation_needed,
                "confidence": r.confidence,
                "recommendations_text": r.recommendations_text,
                "farmer_input_id": r.farmer_input_id,
                "created_at": r.created_at
            }
            for r in recs
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
def submit_feedback(feedback: FeedbackRequest, db: Session = Depends(get_db)):
    """Submit feedback on a recommendation"""
    try:
        rec = DatabaseManager.get_recommendation_by_id(db, feedback.recommendation_id)
        if not rec:
            raise HTTPException(status_code=404, detail="Recommendation not found")
        
        feedback_data = {
            'rating': feedback.rating,
            'was_helpful': feedback.was_helpful,
            'comments': feedback.comments
        }
        saved_feedback = DatabaseManager.save_feedback(
            db, feedback.recommendation_id, feedback_data
        )
        
        return {
            "message": "Feedback saved successfully",
            "feedback_id": saved_feedback.id
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
def get_statistics(db: Session = Depends(get_db)):
    """Get database statistics"""
    try:
        from sqlalchemy import func
        
        total_inputs = db.query(DBFarmerInput).count()
        total_recommendations = db.query(DBRecommendation).count()
        
        avg_confidence = db.query(func.avg(DBRecommendation.confidence)).scalar() or 0
        
        crop_stats = db.query(
            DBFarmerInput.crop_type,
            func.count(DBFarmerInput.id)
        ).group_by(DBFarmerInput.crop_type).all()
        
        return {
            'total_inputs': total_inputs,
            'total_recommendations': total_recommendations,
            'average_confidence': round(float(avg_confidence), 2),
            'crop_distribution': dict(crop_stats)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
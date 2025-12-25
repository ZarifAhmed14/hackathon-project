from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./farmer_data.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class FarmerInput(Base):
    """Store farmer input data"""
    __tablename__ = "farmer_inputs"
    
    id = Column(Integer, primary_key=True, index=True)
    farmer_name = Column(String, nullable=True)
    location = Column(String, nullable=True)
    crop_type = Column(String, nullable=False)
    soil_ph = Column(Float, nullable=False)
    soil_moisture = Column(Float, nullable=False)
    temperature = Column(Float, nullable=False)
    rainfall = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Recommendation(Base):
    """Store AI recommendations"""
    __tablename__ = "recommendations"
    
    id = Column(Integer, primary_key=True, index=True)
    farmer_input_id = Column(Integer, nullable=False)
    fertilizer_level = Column(String, nullable=False)
    irrigation_needed = Column(Boolean, nullable=False)
    confidence = Column(Float, nullable=False)
    recommendations_text = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class UserFeedback(Base):
    """Store user feedback on recommendations"""
    __tablename__ = "user_feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    recommendation_id = Column(Integer, nullable=False)
    rating = Column(Integer, nullable=True)  # 1-5 stars
    was_helpful = Column(Boolean, nullable=True)
    comments = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create all tables
def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully")

# Database dependency for FastAPI
def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Helper functions for CRUD operations
class DatabaseManager:
    """Manager class for database operations"""
    
    @staticmethod
    def save_farmer_input(db, data: dict):
        """Save farmer input to database"""
        farmer_input = FarmerInput(
            farmer_name=data.get('farmer_name'),
            location=data.get('location'),
            crop_type=data['crop_type'],
            soil_ph=data['soil_ph'],
            soil_moisture=data['soil_moisture'],
            temperature=data['temperature'],
            rainfall=data['rainfall']
        )
        db.add(farmer_input)
        db.commit()
        db.refresh(farmer_input)
        return farmer_input
    
    @staticmethod
    def save_recommendation(db, farmer_input_id: int, recommendation: dict):
        """Save recommendation to database"""
        rec = Recommendation(
            farmer_input_id=farmer_input_id,
            fertilizer_level=recommendation['fertilizer_level'],
            irrigation_needed=recommendation['irrigation_needed'],
            confidence=recommendation['confidence'],
            recommendations_text=recommendation['recommendations_text']
        )
        db.add(rec)
        db.commit()
        db.refresh(rec)
        return rec
    
    @staticmethod
    def save_feedback(db, recommendation_id: int, feedback: dict):
        """Save user feedback"""
        fb = UserFeedback(
            recommendation_id=recommendation_id,
            rating=feedback.get('rating'),
            was_helpful=feedback.get('was_helpful'),
            comments=feedback.get('comments')
        )
        db.add(fb)
        db.commit()
        db.refresh(fb)
        return fb
    
    @staticmethod
    def get_farmer_history(db, limit: int = 10):
        """Get recent farmer inputs"""
        return db.query(FarmerInput).order_by(FarmerInput.created_at.desc()).limit(limit).all()
    
    @staticmethod
    def get_recommendations_history(db, limit: int = 10):
        """Get recent recommendations"""
        return db.query(Recommendation).order_by(Recommendation.created_at.desc()).limit(limit).all()
    
    @staticmethod
    def get_recommendation_by_id(db, rec_id: int):
        """Get specific recommendation"""
        return db.query(Recommendation).filter(Recommendation.id == rec_id).first()
    
    @staticmethod
    def get_stats(db):
        """Get database statistics"""
        total_inputs = db.query(FarmerInput).count()
        total_recommendations = db.query(Recommendation).count()
        total_feedback = db.query(UserFeedback).count()
        
        # Average confidence
        avg_confidence = db.query(func.avg(Recommendation.confidence)).scalar() or 0
        
        # Crop distribution
        from sqlalchemy import func
        crop_stats = db.query(
            FarmerInput.crop_type,
            func.count(FarmerInput.id)
        ).group_by(FarmerInput.crop_type).all()
        
        return {
            'total_inputs': total_inputs,
            'total_recommendations': total_recommendations,
            'total_feedback': total_feedback,
            'average_confidence': round(avg_confidence, 2),
            'crop_distribution': dict(crop_stats)
        }

if __name__ == "__main__":
    # Initialize database when run directly
    init_db()
    print("Database initialized!")
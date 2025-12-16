from sqlalchemy import create_engine, Column, Integer, Float, DateTime, String, Boolean, ForeignKey, Text, CheckConstraint, Index
from sqlalchemy.orm import sessionmaker, scoped_session, relationship, declarative_base
from sqlalchemy.pool import StaticPool, QueuePool, NullPool
from contextlib import contextmanager
from datetime import datetime, timezone
import os
import logging

logger = logging.getLogger(__name__)


# SQLAlchemy 1.4/2.0 compatible declarative base
Base = declarative_base()

# Enhanced database configuration with Supabase support
DB_URL = os.getenv('DATABASE_URL', 'sqlite:///data/floodingnaque.db')

# Supabase-specific: Handle connection string format
# Use pg8000 driver for better Windows compatibility (pure Python, no compilation needed)
if DB_URL.startswith('postgres://'):
    DB_URL = DB_URL.replace('postgres://', 'postgresql+pg8000://', 1)
elif DB_URL.startswith('postgresql://') and '+' not in DB_URL.split('://')[0]:
    # Use pg8000 driver for better Windows compatibility
    DB_URL = DB_URL.replace('postgresql://', 'postgresql+pg8000://', 1)

# Connection pool settings for better performance and reliability
if DB_URL.startswith('sqlite'):
    # SQLite-specific settings
    engine = create_engine(
        DB_URL,
        echo=False,
        connect_args={'check_same_thread': False},
        poolclass=StaticPool,  # SQLite works best with StaticPool
        pool_pre_ping=True,  # Verify connections before using
    )
else:
    # PostgreSQL/Supabase settings
    # Use NullPool for serverless/cloud environments to avoid connection issues
    is_supabase = 'supabase' in DB_URL.lower()
    engine = create_engine(
        DB_URL,
        echo=False,
        pool_size=5 if not is_supabase else 1,  # Smaller pool for Supabase free tier
        max_overflow=10 if not is_supabase else 2,
        pool_recycle=300 if is_supabase else 3600,  # Shorter recycle for cloud
        pool_timeout=30,  # Connection timeout to prevent leaks
        pool_pre_ping=True,  # Check connection health
        poolclass=QueuePool,
    )

Session = sessionmaker(bind=engine)
# Use scoped_session for thread-safe session management
db_session = scoped_session(Session)


class WeatherData(Base):
    """Enhanced weather data model with validation and additional fields."""
    __tablename__ = 'weather_data'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Weather measurements with validation constraints
    temperature = Column(
        Float, 
        nullable=False,
        info={'description': 'Temperature in Kelvin'}
    )
    humidity = Column(
        Float, 
        nullable=False,
        info={'description': 'Humidity percentage (0-100)'}
    )
    precipitation = Column(
        Float, 
        nullable=False,
        info={'description': 'Precipitation in mm'}
    )
    
    # Additional weather parameters
    wind_speed = Column(Float, info={'description': 'Wind speed in m/s'})
    pressure = Column(Float, info={'description': 'Atmospheric pressure in hPa'})
    
    # Location information
    location_lat = Column(Float, info={'description': 'Latitude'})
    location_lon = Column(Float, info={'description': 'Longitude'})
    
    # Metadata
    source = Column(
        String(50), 
        default='OWM',
        info={'description': 'Data source: OWM, Weatherstack, Manual'}
    )
    timestamp = Column(
        DateTime, 
        nullable=False,
        index=True,
        info={'description': 'Measurement timestamp'}
    )
    created_at = Column(
        DateTime(timezone=True), 
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        info={'description': 'Record creation time'}
    )
    updated_at = Column(
        DateTime(timezone=True), 
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        info={'description': 'Last update time'}
    )
    
    # Relationships
    predictions = relationship('Prediction', back_populates='weather_data', cascade='all, delete-orphan')
    
    # Table constraints
    __table_args__ = (
        CheckConstraint('temperature >= 173.15 AND temperature <= 333.15', name='valid_temperature'),
        CheckConstraint('humidity >= 0 AND humidity <= 100', name='valid_humidity'),
        CheckConstraint('precipitation >= 0', name='valid_precipitation'),
        CheckConstraint('wind_speed IS NULL OR wind_speed >= 0', name='valid_wind_speed'),
        CheckConstraint('pressure IS NULL OR (pressure >= 870 AND pressure <= 1085)', name='valid_pressure'),
        CheckConstraint('location_lat IS NULL OR (location_lat >= -90 AND location_lat <= 90)', name='valid_latitude'),
        CheckConstraint('location_lon IS NULL OR (location_lon >= -180 AND location_lon <= 180)', name='valid_longitude'),
        Index('idx_weather_timestamp', 'timestamp'),
        Index('idx_weather_location', 'location_lat', 'location_lon'),
        Index('idx_weather_created', 'created_at'),
        {'comment': 'Weather data measurements from various sources'}
    )
    
    def __repr__(self):
        return f"<WeatherData(id={self.id}, temp={self.temperature}, humidity={self.humidity}, precip={self.precipitation}, timestamp={self.timestamp})>"
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'temperature': self.temperature,
            'humidity': self.humidity,
            'precipitation': self.precipitation,
            'wind_speed': self.wind_speed,
            'pressure': self.pressure,
            'location_lat': self.location_lat,
            'location_lon': self.location_lon,
            'source': self.source,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class Prediction(Base):
    """Flood prediction records with audit trail."""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    weather_data_id = Column(Integer, ForeignKey('weather_data.id', ondelete='SET NULL'), index=True)
    
    # Prediction results
    prediction = Column(Integer, nullable=False, info={'description': '0=no flood, 1=flood'})
    risk_level = Column(Integer, info={'description': '0=Safe, 1=Alert, 2=Critical'})
    risk_label = Column(String(50), info={'description': 'Safe/Alert/Critical'})
    confidence = Column(Float, info={'description': 'Prediction confidence (0-1)'})
    
    # Model information
    model_version = Column(Integer, info={'description': 'Model version used'})
    model_name = Column(String(100), default='flood_rf_model')
    
    # Metadata
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    
    # Relationships
    weather_data = relationship('WeatherData', back_populates='predictions')
    alerts = relationship('AlertHistory', back_populates='prediction', cascade='all, delete-orphan')
    
    __table_args__ = (
        CheckConstraint('prediction IN (0, 1)', name='valid_prediction'),
        CheckConstraint('risk_level IS NULL OR risk_level IN (0, 1, 2)', name='valid_risk_level'),
        CheckConstraint('confidence IS NULL OR (confidence >= 0 AND confidence <= 1)', name='valid_confidence'),
        Index('idx_prediction_risk', 'risk_level'),
        Index('idx_prediction_model', 'model_version'),
        {'comment': 'Flood prediction history for analytics and audit'}
    )
    
    def __repr__(self):
        return f"<Prediction(id={self.id}, prediction={self.prediction}, risk={self.risk_label})>"


class AlertHistory(Base):
    """Alert delivery history and tracking."""
    __tablename__ = 'alert_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(Integer, ForeignKey('predictions.id', ondelete='CASCADE'), index=True)
    
    # Alert details
    risk_level = Column(Integer, nullable=False)
    risk_label = Column(String(50), nullable=False)
    location = Column(String(255), info={'description': 'Location name'})
    recipients = Column(Text, info={'description': 'JSON array of recipients'})
    message = Column(Text, info={'description': 'Alert message content'})
    
    # Delivery tracking
    delivery_status = Column(String(50), info={'description': 'delivered/failed/pending'})
    delivery_channel = Column(String(50), info={'description': 'web/sms/email'})
    error_message = Column(Text, info={'description': 'Error details if delivery failed'})
    
    # Metadata
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    delivered_at = Column(DateTime, info={'description': 'Actual delivery timestamp'})
    
    # Relationships
    prediction = relationship('Prediction', back_populates='alerts')
    
    __table_args__ = (
        Index('idx_alert_risk', 'risk_level'),
        Index('idx_alert_status', 'delivery_status'),
        {'comment': 'Alert delivery tracking and history'}
    )
    
    def __repr__(self):
        return f"<AlertHistory(id={self.id}, risk={self.risk_label}, status={self.delivery_status})>"


class ModelRegistry(Base):
    """Model version registry and metadata."""
    __tablename__ = 'model_registry'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(Integer, unique=True, nullable=False, index=True)
    
    # Model file information
    file_path = Column(String(500), nullable=False)
    algorithm = Column(String(100), default='RandomForest')
    
    # Performance metrics
    accuracy = Column(Float)
    precision_score = Column(Float)
    recall_score = Column(Float)
    f1_score = Column(Float)
    roc_auc = Column(Float)
    
    # Training information
    training_date = Column(DateTime)
    dataset_size = Column(Integer)
    dataset_path = Column(String(500))
    
    # Model parameters (JSON stored as text)
    parameters = Column(Text, info={'description': 'JSON serialized model parameters'})
    feature_importance = Column(Text, info={'description': 'JSON serialized feature importance'})
    
    # Status
    is_active = Column(Boolean, default=False, index=True)
    notes = Column(Text, info={'description': 'Additional notes about this version'})
    
    # Metadata
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    created_by = Column(String(100), info={'description': 'User or system that created this'})
    
    __table_args__ = (
        CheckConstraint('accuracy IS NULL OR (accuracy >= 0 AND accuracy <= 1)', name='valid_accuracy'),
        CheckConstraint('precision_score IS NULL OR (precision_score >= 0 AND precision_score <= 1)', name='valid_precision'),
        CheckConstraint('recall_score IS NULL OR (recall_score >= 0 AND recall_score <= 1)', name='valid_recall'),
        CheckConstraint('f1_score IS NULL OR (f1_score >= 0 AND f1_score <= 1)', name='valid_f1'),
        {'comment': 'Model version tracking and performance registry'}
    )
    
    def __repr__(self):
        return f"<ModelRegistry(version={self.version}, accuracy={self.accuracy}, active={self.is_active})>"


def init_db():
    """Initialize database tables with enhanced error handling."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized successfully")
        
        # Log table creation
        tables = Base.metadata.tables.keys()
        logger.info(f"Created tables: {', '.join(tables)}")
        
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

@contextmanager
def get_db_session():
    """Context manager for database sessions."""
    session = db_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        db_session.remove()  # Remove session from registry for scoped_session

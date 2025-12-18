from sqlalchemy import create_engine, Column, Integer, Float, DateTime, String, Boolean, ForeignKey, Text, CheckConstraint, Index, event
from sqlalchemy.orm import sessionmaker, scoped_session, relationship, declarative_base
from sqlalchemy.pool import StaticPool, QueuePool, NullPool
from contextlib import contextmanager
from datetime import datetime, timezone
import os
import logging
import time

logger = logging.getLogger(__name__)

# Pool connection metrics
pool_metrics = {
    'checkouts': 0,
    'checkins': 0,
    'invalidated': 0,
    'last_checkout_time': None,
}


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

# Pool monitoring events for PostgreSQL/Supabase connections
if not DB_URL.startswith('sqlite'):
    @event.listens_for(engine, "checkout")
    def receive_checkout(dbapi_conn, connection_record, connection_proxy):
        """Track when a connection is checked out from the pool."""
        pool_metrics['checkouts'] += 1
        pool_metrics['last_checkout_time'] = time.time()
        logger.debug(f"Connection checked out from pool (total: {pool_metrics['checkouts']})")

    @event.listens_for(engine, "checkin")
    def receive_checkin(dbapi_conn, connection_record):
        """Track when a connection is returned to the pool."""
        pool_metrics['checkins'] += 1
        logger.debug(f"Connection checked in to pool (total: {pool_metrics['checkins']})")

    @event.listens_for(engine, "invalidate")
    def receive_invalidate(dbapi_conn, connection_record, exception):
        """Track when a connection is invalidated."""
        pool_metrics['invalidated'] += 1
        if exception:
            logger.warning(f"Connection invalidated due to: {exception}")
        else:
            logger.debug(f"Connection invalidated (total: {pool_metrics['invalidated']})")

    @event.listens_for(engine, "connect")
    def receive_connect(dbapi_conn, connection_record):
        """Log new database connections."""
        logger.info("New database connection established")


def get_pool_status():
    """Get current connection pool status and metrics."""
    if DB_URL.startswith('sqlite'):
        return {'status': 'sqlite_static_pool', 'metrics': None}
    
    pool = engine.pool
    return {
        'pool_size': pool.size(),
        'checked_out': pool.checkedout(),
        'overflow': pool.overflow(),
        'checked_in': pool.checkedin(),
        'metrics': pool_metrics.copy(),
    }


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
        info={'description': 'Data source: OWM, Weatherstack, Meteostat, Manual'}
    )
    station_id = Column(
        String(50),
        nullable=True,
        info={'description': 'Weather station ID (for Meteostat data)'}
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
    # Soft delete support
    is_deleted = Column(Boolean, default=False, nullable=False, index=True)
    deleted_at = Column(DateTime(timezone=True), nullable=True)
    
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
        Index('idx_weather_location_time', 'location_lat', 'location_lon', 'timestamp'),  # Composite index for common query patterns
        Index('idx_weather_created', 'created_at'),
        Index('idx_weather_source', 'source'),
        Index('idx_weather_active', 'is_deleted'),  # Index for filtering active records
        {'comment': 'Weather data measurements from various sources including Meteostat'}
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
            'station_id': self.station_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'is_deleted': self.is_deleted,
        }
    
    def soft_delete(self):
        """Mark record as deleted without removing from database."""
        self.is_deleted = True
        self.deleted_at = datetime.now(timezone.utc)
    
    def restore(self):
        """Restore a soft-deleted record."""
        self.is_deleted = False
        self.deleted_at = None


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
    
    # Soft delete support
    is_deleted = Column(Boolean, default=False, nullable=False, index=True)
    deleted_at = Column(DateTime(timezone=True), nullable=True)
    
    __table_args__ = (
        CheckConstraint('prediction IN (0, 1)', name='valid_prediction'),
        CheckConstraint('risk_level IS NULL OR risk_level IN (0, 1, 2)', name='valid_risk_level'),
        CheckConstraint('confidence IS NULL OR (confidence >= 0 AND confidence <= 1)', name='valid_confidence'),
        Index('idx_prediction_risk', 'risk_level'),
        Index('idx_prediction_model', 'model_version'),
        Index('idx_prediction_active', 'is_deleted'),
        {'comment': 'Flood prediction history for analytics and audit'}
    )
    
    def __repr__(self):
        return f"<Prediction(id={self.id}, prediction={self.prediction}, risk={self.risk_label})>"
    
    def soft_delete(self):
        """Mark record as deleted without removing from database."""
        self.is_deleted = True
        self.deleted_at = datetime.now(timezone.utc)
    
    def restore(self):
        """Restore a soft-deleted record."""
        self.is_deleted = False
        self.deleted_at = None


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
    
    # Soft delete support
    is_deleted = Column(Boolean, default=False, nullable=False, index=True)
    deleted_at_field = Column('deleted_at', DateTime(timezone=True), nullable=True)
    
    __table_args__ = (
        Index('idx_alert_risk', 'risk_level'),
        Index('idx_alert_status', 'delivery_status'),
        Index('idx_alert_active', 'is_deleted'),
        {'comment': 'Alert delivery tracking and history'}
    )
    
    def __repr__(self):
        return f"<AlertHistory(id={self.id}, risk={self.risk_label}, status={self.delivery_status})>"
    
    def soft_delete(self):
        """Mark record as deleted without removing from database."""
        self.is_deleted = True
        self.deleted_at_field = datetime.now(timezone.utc)
    
    def restore(self):
        """Restore a soft-deleted record."""
        self.is_deleted = False
        self.deleted_at_field = None


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

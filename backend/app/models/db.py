from sqlalchemy import create_engine, Column, Integer, Float, DateTime
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
from contextlib import contextmanager
import os

Base = declarative_base()
engine = create_engine(os.getenv('DATABASE_URL', 'sqlite:///floodingnaque.db'), echo=False)
Session = sessionmaker(bind=engine)
# Use scoped_session for thread-safe session management
db_session = scoped_session(Session)

class WeatherData(Base):
    __tablename__ = 'weather_data'
    id = Column(Integer, primary_key=True)
    temperature = Column(Float)
    humidity = Column(Float)
    precipitation = Column(Float)
    timestamp = Column(DateTime)

def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)

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

from sqlalchemy import Column, Integer, Float, String, DateTime
from datetime import datetime, timezone
from .database import Base

class PredictionLog(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, Index=True)
    engine_id = Column(Integer, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    predicted_rul = Column(Float)

    reconstruction_mse = Column(Float)
    is_anomaly = Column(String)

    rag_diagnostics = Column(String, nullable=True)
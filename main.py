from fastapi import FastAPI, Depends
from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import joblib
import numpy as np
from pydantic import BaseModel

# --- 1. Database 설정 (database.py 분량) ---
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    id = Column(Integer, primary_key=True, index=True)
    sepal_length = Column(Float)
    sepal_width = Column(Float)
    petal_length = Column(Float)
    petal_width = Column(Float)
    result = Column(String)

Base.metadata.create_all(bind=engine)

# --- 2. ML 모델 로드 ---
model = joblib.load("iris_model.pkl")
iris_names = ['setosa', 'versicolor', 'virginica']

# --- 3. FastAPI 앱 및 스키마 ---
app = FastAPI()

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- 4. 엔드포인트 ---
@app.post("/predict")
def predict_and_store(data: IrisInput, db: Session = Depends(get_db)):
    # A. 모델 예측
    features = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    prediction = model.predict(features)
    species = iris_names[prediction[0]]

    # B. DB 저장
    db_log = PredictionLog(
        sepal_length=data.sepal_length,
        sepal_width=data.sepal_width,
        petal_length=data.petal_length,
        petal_width=data.petal_width,
        result=species
    )
    db.add(db_log)
    db.commit()
    
    return {"species": species, "log_id": db_log.id}

@app.get("/logs")
def get_logs(db: Session = Depends(get_db)):
    return db.query(PredictionLog).all()
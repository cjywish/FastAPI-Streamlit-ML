from fastapi import FastAPI, Depends
from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import joblib
import numpy as np
from pydantic import BaseModel

# --- 1. Database 설정 ---
# SQLite 데이터베이스 파일 경로 설정 (현재 폴더의 test.db)
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

# DB 엔진 생성 (connect_args는 SQLite에서 멀티스레드 사용을 위한 설정)
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})

# DB 세션을 생성하는 클래스 (실제 작업 시 이 클래스로 인스턴스 생성)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# DB 테이블 모델을 만들기 위한 기본 클래스
Base = declarative_base()

# DB에 저장될 테이블 구조 정의
class PredictionLog(Base):
    __tablename__ = "prediction_logs"  # 데이터베이스 내 테이블 이름
    id = Column(Integer, primary_key=True, index=True) # 고유 번호 (기본키)
    sepal_length = Column(Float)  # 꽃받침 길이
    sepal_width = Column(Float)   # 꽃받침 너비
    petal_length = Column(Float)  # 꽃잎 길이
    petal_width = Column(Float)    # 꽃잎 너비
    result = Column(String)       # 예측된 붓꽃 종 이름

# 코드가 실행될 때 정의된 테이블들을 실제로 DB 파일에 생성
Base.metadata.create_all(bind=engine)

# --- 2. ML 모델 로드 ---
# 학습된 머신러닝 모델 파일을 불러옴 (서버 시작 시 한 번만 실행)
model = joblib.load("iris_model.pkl")
# 예측 숫자 결과(0, 1, 2)를 실제 이름으로 매핑하기 위한 리스트
iris_names = ['setosa', 'versicolor', 'virginica']

# --- 3. FastAPI 앱 및 스키마 ---
app = FastAPI()

# 사용자가 API로 보낼 데이터 형식을 정의 (Pydantic 모델)
# 이 형식에 맞지 않는 데이터가 들어오면 자동으로 422 에러 반환
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# DB 세션을 생성하고 사용이 끝나면 자동으로 닫아주는 의존성 함수
def get_db():
    db = SessionLocal()
    try:
        yield db  # API 함수에 DB 세션 제공
    finally:
        db.close() # 요청 처리가 끝나면 반드시 연결 종료

# --- 4. 엔드포인트(API 경로) 정의 ---

# [POST] 예측 수행 및 저장 API
@app.post("/predict")
def predict_and_store(data: IrisInput, db: Session = Depends(get_db)):
    """
    1. 사용자의 입력을 받아 ML 모델로 예측합니다.
    2. 입력값과 결과값을 SQLite DB에 저장합니다.
    """
    # A. 모델 예측을 위한 데이터 가공 (2차원 배열 형태 필요)
    features = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    prediction = model.predict(features) # 모델 예측 실행
    species = iris_names[prediction[0]]   # 숫자를 붓꽃 이름으로 변환

    # B. DB 저장용 객체 생성
    db_log = PredictionLog(
        sepal_length=data.sepal_length,
        sepal_width=data.sepal_width,
        petal_length=data.petal_length,
        petal_width=data.petal_width,
        result=species
    )
    db.add(db_log)    # DB 대기열에 추가
    db.commit()       # 실제 DB 파일에 저장(커밋)
    db.refresh(db_log) # 저장된 후 생성된 id 등을 업데이트
    
    return {"species": species, "log_id": db_log.id}

# [GET] 저장된 전체 로그 조회 API
@app.get("/logs")
def get_logs(db: Session = Depends(get_db)):
    """
    DB에 저장된 모든 예측 로그를 리스트 형식으로 반환합니다.
    """
    return db.query(PredictionLog).all()
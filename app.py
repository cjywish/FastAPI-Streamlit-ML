import streamlit as st
import requests
import pandas as pd

# --- 페이지 설정 ---
# 브라우저 탭 제목과 화면 레이아웃(wide: 넓게) 설정
st.set_page_config(page_title="Iris 분류 및 로그 시스템", layout="wide")

# 메인 화면 제목 및 설명
st.title("🌸 Iris 꽃 분류 서비스 (ML + FastAPI + DB)")
st.markdown("""
이 시스템은 **FastAPI**를 통해 ML 모델 예측을 수행하고, 
모든 요청 결과를 **SQLite** DB에 자동으로 기록합니다.
""")

# --- 좌측 사이드바: 데이터 입력 ---
st.sidebar.header("Input Features")
# 슬라이더를 통해 꽃의 4가지 특성값(Feature)을 입력받음
# 형식: st.sidebar.slider("라벨", 최소값, 최대값, 기본값)
sepal_l = st.sidebar.slider("Sepal Length", 4.0, 8.0, 5.0)
sepal_w = st.sidebar.slider("Sepal Width", 2.0, 4.5, 3.0)
petal_l = st.sidebar.slider("Petal Length", 1.0, 7.0, 1.5)
petal_w = st.sidebar.slider("Petal Width", 0.1, 2.5, 0.2)

# 예측 버튼 클릭 시 로직 시작
if st.sidebar.button("Predict & Save"):
    # FastAPI 백엔드로 보낼 데이터를 JSON 형식(딕셔너리)으로 준비
    # 주의: 키(Key) 이름이 FastAPI의 Pydantic 모델(IrisInput) 필드명과 일치해야 함
    payload = {
        "sepal_length": sepal_l,
        "sepal_width": sepal_w,
        "petal_length": petal_l,
        "petal_width": petal_w
    }
    
    # 1. FastAPI 예측 API 호출
    with st.spinner("예측 중..."): # 요청 처리 중 로딩 애니메이션 표시
        try:
            # POST 방식으로 데이터 전송 (FastAPI 주소 확인 필요)
            response = requests.post("http://127.0.0.1:8000/predict", json=payload)
            
            # 응답 코드가 200(성공)인 경우
            if response.status_code == 200:
                result = response.json() # 응답받은 JSON 데이터를 파이썬 딕셔너리로 변환
                # 결과 출력 (꽃 종류 및 DB 저장 ID)
                st.success(f"### 예측 결과: **{result['species']}**")
                st.info(f"DB 로그 ID {result['log_id']}번으로 성공적으로 저장되었습니다.")
            else:
                # 422(데이터 형식 오류) 등의 문제 발생 시
                st.error(f"서버 에러: {response.status_code} - 상세 내용을 확인하세요.")
        except requests.exceptions.ConnectionError:
            # FastAPI 서버가 실행 중이지 않을 때 발생
            st.error("API 서버(FastAPI)가 꺼져 있습니다. 서버를 먼저 실행하세요.")

# 화면 구분선
st.divider()

# --- 하단: DB 로그 확인 섹션 ---
st.subheader("📊 최근 예측 로그 (From SQLite)")
if st.button("로그 새로고침"):
    # 2. FastAPI 로그 조회 API 호출 (GET 방식)
    try:
        log_response = requests.get("http://127.0.0.1:8000/logs")
        
        if log_response.status_code == 200:
            logs = log_response.json() # 리스트 형태의 로그 데이터
            
            if logs:
                # 1. JSON 데이터를 판다스 데이터프레임으로 변환
                df = pd.DataFrame(logs)
                # 2. 최신순(ID 내림차순)으로 정렬하여 표로 출력
                st.table(df.sort_values(by="id", ascending=False))
            else:
                st.warning("아직 저장된 로그가 없습니다. 먼저 예측을 진행해 보세요.")
        else:
            st.error("로그를 불러올 수 없습니다.")
    except Exception as e:
        st.error(f"연결 오류: {e}")
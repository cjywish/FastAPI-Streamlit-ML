import streamlit as st
import requests
import pandas as pd

# predict iris 
# with fastapi , streamlit
st.set_page_config(page_title="Iris 분류 및 로그 시스템", layout="wide")

st.title("🌸 Iris 꽃 분류 서비스 (ML + FastAPI + DB)")
st.markdown("""
이 시스템은 **FastAPI**를 통해 ML 모델 예측을 수행하고, 
모든 요청 결과를 **SQLite** DB에 자동으로 기록합니다.
""")

# 좌측 사이드바: 데이터 입력
st.sidebar.header("Input Features")
sepal_l = st.sidebar.slider("Sepal Length", 4.0, 8.0, 5.0)
sepal_w = st.sidebar.slider("Sepal Width", 2.0, 4.5, 3.0)
petal_l = st.sidebar.slider("Petal Length", 1.0, 7.0, 1.5)
petal_w = st.sidebar.slider("Petal Width", 0.1, 2.5, 0.2)

# 예측 버튼
if st.sidebar.button("Predict & Save"):
    payload = {
        "sepal_length": sepal_l,
        "sepal_width": sepal_w,
        "petal_length": petal_l,
        "petal_width": petal_w
    }
    
    # 1. FastAPI 예측 API 호출
    with st.spinner("예측 중..."):
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        
    if response.status_code == 200:
        result = response.json()
        st.success(f"### 예측 결과: **{result['species']}**")
        st.info(f"DB 로그 ID {result['log_id']}번으로 성공적으로 저장되었습니다.")
    else:
        st.error("API 서버와 통신에 실패했습니다.")

st.divider()

# 하단: DB 로그 확인 섹션
st.subheader("📊 최근 예측 로그 (From SQLite)")
if st.button("로그 새로고침"):
    # 2. FastAPI 로그 조회 API 호출
    log_response = requests.get("http://127.0.0.1:8000/logs")
    if log_response.status_code == 200:
        logs = log_response.json()
        if logs:
            df = pd.DataFrame(logs)
            st.table(df.sort_values(by="id", ascending=False))
        else:
            st.write("아직 저장된 로그가 없습니다.")
    else:
        st.error("로그를 불러올 수 없습니다.")

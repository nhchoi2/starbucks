import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1️⃣ 모델 로드
model_path = "models/fatigue_predictor.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# 2️⃣ 원본 데이터의 컬럼 로드 (입력 데이터 변환을 위해)
processed_data_path = "data/processed_data.csv"
df_reference = pd.read_csv(processed_data_path)  # 컬럼 이름 가져오기
model_features = df_reference.drop(columns=["피로도값"]).columns  # 모델 학습에 사용한 컬럼

# 3️⃣ 페이지 제목 및 설명
st.title("📊 피로도 예측 대시보드")
st.write("사용자의 건강 데이터를 입력하면 AI가 피로도를 예측합니다.")

# 4️⃣ 사용자 입력 받기
col1, col2 = st.columns(2)

with col1:
    성별 = st.selectbox("성별", ["남성", "여성"])
    평균심박수 = st.slider("평균 심박수 (BPM)", 40, 120, 70)
    신체스트레스값 = st.slider("신체 스트레스 값", 0, 100, 50)
    운동량 = st.slider("하루 평균 운동량 (분)", 0, 180, 30)

with col2:
    정신스트레스값 = st.slider("정신 스트레스 값", 0, 100, 50)
    혈관연령 = st.slider("혈관 연령 (세)", 20, 80, 40)
    이상심박수 = st.slider("이상 심박수 (BPM)", 40, 150, 80)
    수면시간 = st.slider("하루 평균 수면 시간 (시간)", 3, 12, 7)

# 5️⃣ 입력 데이터 전처리
성별 = 0 if 성별 == "남성" else 1  # 남성 = 0, 여성 = 1

# 입력값을 DataFrame으로 변환 (960개 컬럼을 맞추기 위해)
input_data = pd.DataFrame([[평균심박수, 이상심박수, 신체스트레스값, 정신스트레스값, 혈관연령, 성별, 운동량, 수면시간]],
                          columns=["평균심박수", "이상심박수", "신체스트레스값", "정신스트레스값", "혈관연령", "성별", "운동량", "수면시간"])

# 6️⃣ 모델 학습 데이터의 컬럼을 기준으로 입력값 정렬 (부족한 컬럼은 0으로 채우기)
input_data = input_data.reindex(columns=model_features, fill_value=0)

# 7️⃣ 예측 실행
if st.button("🚀 피로도 예측하기"):
    prediction = model.predict(input_data)[0]  # 피로도 예측 값
    st.success(f"📊 예측된 피로도 값: {prediction:.2f}")

    # 피로도 수준에 따른 메시지
    if prediction < 3:
        st.info("🔹 피로도가 낮습니다! 현재 상태를 유지하세요.")
    elif prediction < 6:
        st.warning("⚠️ 피로도가 중간 수준입니다. 충분한 휴식이 필요합니다!")
    else:
        st.error("🚨 피로도가 높습니다! 즉시 휴식을 취하세요.")

    # ✅ 피로도 예측 결과 시각화 추가
    st.progress(int(prediction * 10))

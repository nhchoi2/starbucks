import streamlit as st

# 1️⃣ 페이지 제목 및 설명
st.title("🛌 맞춤형 휴식 가이드")
st.write("AI가 예측한 피로도 값을 바탕으로 사용자에게 맞춤형 휴식 가이드를 제공합니다.")

# 2️⃣ 사용자 피로도 값 입력
st.subheader("✅ 피로도 입력")
predicted_fatigue = st.slider("AI가 예측한 피로도 값 (0-10)", min_value=0.0, max_value=10.0, step=0.1, value=5.0)

# 3️⃣ 피로도 수준별 가이드
st.subheader("🛌 맞춤형 가이드")
if predicted_fatigue < 3.0:
    st.success("💪 피로도가 낮습니다! 현재 상태를 유지하세요.")
    st.markdown("""
    - **가벼운 운동:** 요가, 스트레칭
    - **활동:** 생산적인 하루를 계획하세요!
    - **건강 팁:** 충분한 수분 섭취
    """)
elif 3.0 <= predicted_fatigue < 6.0:
    st.warning("⚠️ 피로도가 중간 수준입니다. 적절한 휴식이 필요합니다.")
    st.markdown("""
    - **휴식:** 15~30분 동안 눈을 감고 편안히 쉬세요.
    - **스트레스 관리:** 간단한 명상이나 호흡 운동을 시도해 보세요.
    - **건강 팁:** 균형 잡힌 식사를 하세요.
    """)
else:
    st.error("🚨 피로도가 높습니다! 즉시 충분한 휴식이 필요합니다.")
    st.markdown("""
    - **충분한 수면:** 최소 7~8시간의 수면이 필요합니다.
    - **의료 상담:** 필요하다면 병원을 방문해 전문가 상담을 받으세요.
    - **건강 팁:** 무리한 신체 활동을 피하고, 스트레스를 줄이는 활동을 하세요.
    """)

# 4️⃣ 추가 추천 이미지 및 링크
st.subheader("🌟 추가 리소스")
st.markdown("""
- **운동 가이드:** [간단한 스트레칭 방법 보기](https://www.stretchingguide.com/)
- **명상 앱 추천:** [Calm](https://www.calm.com/), [Headspace](https://www.headspace.com/)
- **피로 관리 팁:** [건강 정보 블로그](https://www.healthline.com/)
""")

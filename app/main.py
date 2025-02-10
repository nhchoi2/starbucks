import streamlit as st
from PIL import Image

# 앱 기본 설정
st.set_page_config(page_title="개인 맞춤형 피로 관리", layout="wide")

# 타이틀
st.title("🏥 개인 맞춤형 피로 관리 AI")
st.write("피로도를 예측하고 맞춤형 휴식 가이드를 받아보세요!")

# 네비게이션 메뉴
menu = st.sidebar.radio("메뉴 선택", ["🏠 홈", "📊 대시보드", "🛌 추천 시스템"])

# 🏠 홈 화면
if menu == "🏠 홈":
    st.subheader("🔍 AI 기반 피로 관리 시스템")
    st.write("""
    이 앱은 AI를 활용하여 사용자의 피로도를 예측하고 맞춤형 휴식 가이드를 제공합니다.  
    왼쪽 사이드바에서 원하는 메뉴를 선택하세요!
    """)
    
    # 앱 설명 이미지 (선택)
    image_path = "assets/home_image.jpg"  # 추가할 경우
    try:
        image = Image.open(image_path)
        st.image(image, caption="AI 기반 피로 관리 시스템", width=500)
    except FileNotFoundError:
        st.warning("홈 화면 이미지를 찾을 수 없습니다.")

# 📊 대시보드 페이지로 이동
elif menu == "📊 대시보드":
    st.switch_page("pages/dashboard.py")

# 🛌 추천 시스템 페이지로 이동
elif menu == "🛌 추천 시스템":
    st.switch_page("pages/recommend.py")

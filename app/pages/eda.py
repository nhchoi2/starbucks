import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import matplotlib.font_manager as fm

# 📌 한글 폰트 설정 (폰트 적용)
font_path = "assets/font1.ttf"
fontprop = fm.FontProperties(fname=font_path, size=12)
plt.rc('font', family=fontprop.get_name())

# 1️⃣ 데이터 로드
st.title("📊 탐색적 데이터 분석 (EDA)")

file_path = "data/processed_data.csv"
df = pd.read_csv(file_path)

# 2️⃣ 기본 데이터 정보 확인
st.subheader("✅ 데이터 정보 확인")
# `df.info()` 출력
buffer = io.StringIO()  # 메모리 버퍼 생성
df.info(buf=buffer)     # `df.info()` 내용을 버퍼에 저장
info_str = buffer.getvalue()  # 버퍼 내용을 문자열로 가져오기

# 간략한 데이터 요약 정보 추가
st.write("✅ 데이터 정보 요약:")
st.markdown(f"""
- **총 행 개수:** {df.shape[0]}  
- **총 열 개수:** {df.shape[1]}  
- **첫 번째 컬럼:** {df.columns[0]}  
- **마지막 컬럼:** {df.columns[-1]}  
- **데이터 타입:** float64 ({len(df.select_dtypes(include=['float64']).columns)}개 컬럼)  
- **메모리 사용량:** {df.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB  
""")

# ✅ '평균심박수'부터 '행정동명_해밀동' 컬럼만 선택
columns_to_show = df.columns[df.columns.get_loc('평균심박수'):df.columns.get_loc('행정동명_해밀동') + 1]

st.subheader("✅ 데이터 통계 요약")
st.write(df[columns_to_show].describe())

# 3️⃣ 수치형 변수 분포 확인 (히스토그램, Seaborn 사용)
st.subheader("📊 수치형 변수 분포")
num_features = ['평균심박수', '이상심박수', '피로도값', '혈관연령', '정신스트레스값', '신체스트레스값']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()  # 2D 배열을 1D로 변환

for i, col in enumerate(num_features):
    sns.histplot(df[col], bins=30, kde=True, ax=axes[i])
    axes[i].set_title(col, fontproperties=fontprop)  # ✅ 한글 적용
    axes[i].set_xlabel("값", fontproperties=fontprop)
    axes[i].set_ylabel("빈도", fontproperties=fontprop)

plt.tight_layout()
st.pyplot(fig)  # ✅ 한글 적용된 히스토그램 출력

# 4️⃣ 변수 간 관계 분석 (상관관계 히트맵)
st.subheader("📈 변수 간 상관관계")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df[num_features].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
ax.set_title("변수 간 상관관계", fontproperties=fontprop)
ax.set_xticklabels(ax.get_xticklabels(), fontproperties=fontprop)
ax.set_yticklabels(ax.get_yticklabels(), fontproperties=fontprop)
st.pyplot(fig)



# 6️⃣ 범주형 변수 분포 확인 (행정동명)
st.subheader("🏡 행정동별 데이터 분포 (상위 10개)")

# ✅ '행정동명'이 포함된 컬럼 자동 검색
district_columns = df.filter(like="행정동명").sum().sort_values(ascending=False).head(10)

# ✅ 데이터프레임 변환
df_district = pd.DataFrame({'행정동명': district_columns.index, '수량': district_columns.values})

# ✅ 컬럼명 변환 (원핫 인코딩된 컬럼에서 '행정동명_' 제거)
df_district['행정동명'] = df_district['행정동명'].str.replace('행정동명_', '')

# ✅ 시각화
fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(data=df_district, y='행정동명', x='수량', palette="coolwarm", ax=ax)

# ✅ 한글 폰트 적용 (Y축 라벨 수동 설정)
ax.set_title("상위 10개 행정동 데이터 분포", fontproperties=fontprop)
ax.set_xlabel("수량", fontproperties=fontprop)
ax.set_ylabel("행정동명", fontproperties=fontprop)

# ✅ Y축 라벨(행정동명)에도 한글 폰트 적용
y_labels = [label.get_text() for label in ax.get_yticklabels()]
ax.set_yticklabels(y_labels, fontproperties=fontprop)

st.pyplot(fig)

# 7️⃣ 이상치 탐지 (Boxplot)
st.subheader("⚠️ 이상치 탐지 (Boxplot)")
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df[num_features], ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, fontproperties=fontprop)  # ✅ 가독성 개선
ax.set_title("이상치 탐지 (Boxplot)", fontproperties=fontprop)
ax.set_xlabel("변수명", fontproperties=fontprop)
ax.set_ylabel("값", fontproperties=fontprop)
st.pyplot(fig)


st.success("✅ 탐색적 데이터 분석 완료!")

print("📌 데이터 컬럼 확인:", df.columns.unique())

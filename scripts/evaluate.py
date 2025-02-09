import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 📌 한글 폰트 설정 (사용자 정의 폰트 적용)
import matplotlib.font_manager as fm
font_path = "assets/font1.ttf"  # 폰트 경로
fontprop = fm.FontProperties(fname=font_path, size=12)

# ✅ Matplotlib에서 기본 폰트를 변경하는 코드 (폰트 오류 해결)
plt.rcParams['font.family'] = fontprop.get_name()
plt.rcParams['axes.unicode_minus'] = False  # 마이너스(-) 기호 깨짐 방지

# 1️⃣ 데이터 로드
file_path = "data/processed_data.csv"
df = pd.read_csv(file_path)

# 2️⃣ 기본 데이터 정보 확인
print("✅ 데이터 정보 확인:")
print(df.info())

print("\n✅ 데이터 통계 요약:")
print(df.describe())

# 3️⃣ 수치형 변수 분포 확인 (히스토그램)
num_features = ['평균심박수', '이상심박수', '피로도값', '혈관연령', '정신스트레스값', '신체스트레스값']
df[num_features].hist(figsize=(12, 8), bins=30)
plt.suptitle("📊 수치형 변수 분포", fontproperties=fontprop)
plt.show()

# 4️⃣ 변수 간 관계 분석 (상관관계 히트맵)
plt.figure(figsize=(10, 8))
sns.heatmap(df[num_features].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("📈 변수 간 상관관계", fontproperties=fontprop)
plt.show()

# 5️⃣ 범주형 변수 분포 확인 (성별, 행정동명)
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x='성별', palette="coolwarm")
plt.title("📊 성별 분포", fontproperties=fontprop)
plt.show()

# 행정동명 분포 (상위 10개만 표시)
plt.figure(figsize=(12, 5))
sns.countplot(data=df, y=df['행정동명'].value_counts().index[:10], palette="coolwarm")
plt.title("🏡 상위 10개 행정동 분포", fontproperties=fontprop)
plt.show()

# 6️⃣ 이상치 탐지 (Boxplot)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[num_features])
plt.xticks(rotation=30)
plt.title("📌 이상치 탐지 (Boxplot)", fontproperties=fontprop)
plt.show()

print("✅ EDA 완료!")

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# 파일 경로
input_file = "data/stress_fatigue_data.csv"  # 원본 데이터 경로
output_file = "data/processed_data.csv"  # 전처리 완료 데이터 경로

# 1️⃣ 데이터 로드
df = pd.read_csv(input_file, encoding="utf-8-sig")

# 2️⃣ 수치형 및 범주형 컬럼 정의
numeric_features = ['평균심박수', '이상심박수', '피로도값', '혈관연령', '정신스트레스값', '신체스트레스값']
categorical_features_onehot = ['행정동명', '측정시간']
categorical_features_label = ['성별']  # 성별은 레이블 인코딩

# 3️⃣ 성별 레이블 인코딩 (남: 0, 여: 1)
df['성별'] = df['성별'].map({'M': 0, 'F': 1})

# 4️⃣ 수치형 데이터 전처리 (결측값 처리 + 정규화)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # 결측값 평균으로 대체
    ('scaler', StandardScaler())  # 정규화 (표준화)
])

# 5️⃣ 범주형 데이터 전처리 (원핫인코딩) → Sparse Matrix 문제 해결 (sparse=False)
categorical_transformer_onehot = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # 결측값 최빈값으로 대체
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Sparse Matrix 방지!
])

# 6️⃣ 전처리 파이프라인 구성
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat_onehot', categorical_transformer_onehot, categorical_features_onehot),
    ]
)

# 7️⃣ 전처리 적용
processed_data = preprocessor.fit_transform(df)

# 8️⃣ 새로운 컬럼 이름 생성 (원핫인코딩 결과 포함)
columns_numeric = numeric_features
columns_onehot = preprocessor.named_transformers_['cat_onehot'].named_steps['onehot'].get_feature_names_out(categorical_features_onehot)
columns_final = list(columns_numeric) + list(columns_onehot)

# 컬럼 개수 확인
print(f"✅ 컬럼 개수 확인: {len(columns_final)}, 변환된 데이터 셰이프: {processed_data.shape}")

# 9️⃣ 희소 행렬 방지: numpy 배열로 변환 후 DataFrame 생성
df_processed = pd.DataFrame(processed_data, columns=columns_final)

# 🔟 결과 저장
df_processed.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"✅ 데이터 전처리 완료! 결과 저장: {output_file}")

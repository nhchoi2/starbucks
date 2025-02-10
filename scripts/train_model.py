import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1️⃣ 데이터 로드
file_path = "data/processed_data.csv"
df = pd.read_csv(file_path)

# 2️⃣ 입력(X)과 타겟(y) 분리
target_column = "피로도값"  # 예측할 변수
X = df.drop(columns=[target_column])  # 입력 데이터 (피로도값 제외)
y = df[target_column]  # 타겟 데이터 (예측값)

# 3️⃣ 학습/테스트 데이터 분할 (80% 학습, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ 랜덤 포레스트 모델 생성 및 하이퍼파라미터 튜닝
param_grid = {
    'n_estimators': [50, 100, 200],  # 트리 개수 조정
    'max_depth': [None, 10, 20],  # 트리 깊이 조정
    'min_samples_split': [2, 5, 10],  # 분할 최소 샘플 개수 조정
    'min_samples_leaf': [1, 2, 4]  # 리프 노드의 최소 샘플 개수 조정
}

# GridSearchCV로 최적의 파라미터 찾기
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, 
                           scoring='r2', cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# 최적의 모델 선택
best_model = grid_search.best_estimator_
print(f"✅ 최적의 하이퍼파라미터: {grid_search.best_params_}")

# 5️⃣ 모델 평가
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"✅ 최적 모델 평가 결과:")
print(f"📌 MAE (평균 절대 오차): {mae:.3f}")
print(f"📌 RMSE (평균 제곱근 오차): {rmse:.3f}")
print(f"📌 R^2 Score (결정 계수): {r2:.3f}")

# 6️⃣ 모델 저장
model_path = "models/fatigue_predictor.pkl"
with open(model_path, "wb") as file:
    pickle.dump(best_model, file)

print(f"✅ 최적 모델 저장 완료! 저장 경로: {model_path}")

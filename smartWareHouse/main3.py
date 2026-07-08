"""
스마트 창고 출고 지연 예측 - 3-모델 앙상블 제출
LightGBM + XGBoost + RandomForest (모두 GroupKFold로 검증됨)
"""
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
 
# ===== 1. 데이터 =====
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
 
X = train.drop(['ID', 'layout_id', 'scenario_id', 'avg_delay_minutes_next_30m'], axis=1)
y = train['avg_delay_minutes_next_30m']
y_log = np.log1p(y)   # 타깃 로그 변환
 
X_test = test.drop(['ID', 'layout_id', 'scenario_id'], axis=1)
 
# ===== 2. 튜닝된 파라미터 =====
lgb_params = {
    'n_estimators': 1000, 'learning_rate': 0.0100, 'num_leaves': 140,
    'max_depth': 8, 'min_child_samples': 25, 'subsample': 0.9608,
    'colsample_bytree': 0.6607, 'reg_alpha': 1.2338, 'reg_lambda': 7.6823,
    'random_state': 42, 'verbose': -1
}
xgb_params = {
    'n_estimators': 1000, 'learning_rate': 0.0101, 'max_depth': 7,
    'min_child_weight': 3, 'subsample': 0.6587, 'colsample_bytree': 0.8854,
    'reg_alpha': 3.3612, 'reg_lambda': 2.6707, 'random_state': 42
}
 
# ===== 3. 전체 데이터로 세 모델 학습 =====
print("LightGBM 학습...")
m_lgb = LGBMRegressor(**lgb_params)
m_lgb.fit(X, y_log)
 
print("XGBoost 학습...")
m_xgb = XGBRegressor(**xgb_params)
m_xgb.fit(X, y_log)
 
print("RandomForest 학습...")
m_rf = RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42)
m_rf.fit(X, y_log)
 
# ===== 4. 예측 + 로그 역변환 + 앙상블 =====
p_lgb = np.expm1(m_lgb.predict(X_test))
p_xgb = np.expm1(m_xgb.predict(X_test))
p_rf  = np.expm1(m_rf.predict(X_test))
 
pred = (p_lgb + p_xgb + p_rf) / 3      # 3-모델 평균
pred = np.clip(pred, 0, None)          # 음수 방지
 
# ===== 5. 제출 파일 =====
submission = pd.DataFrame({
    'ID': test['ID'],
    'avg_delay_minutes_next_30m': pred
})
submission.to_csv('submission_ensemble.csv', index=False)
print(f"\n저장 완료: submission_ensemble.csv {submission.shape}")
print(submission.head())
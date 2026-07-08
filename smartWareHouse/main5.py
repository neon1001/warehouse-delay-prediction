import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
layout = pd.read_csv('layout_info.csv')

train = train.merge(layout, on='layout_id', how='left')
test = test.merge(layout, on='layout_id', how='left')

def add_features(df):
    df['order_per_packstation'] = df['order_inflow_15m'] / (df['pack_station_count'] + 1)
    df['order_per_robot']       = df['order_inflow_15m'] / (df['robot_total'] + 1)
    df['order_per_charger']     = df['order_inflow_15m'] / (df['charger_count'] + 1)
    df['active_robot_density']  = df['robot_active'] / (df['floor_area_sqm'] + 1)
    return df

train = add_features(train)
test = add_features(test)

train = pd.get_dummies(train, columns=['layout_type'])
test = pd.get_dummies(test, columns=['layout_type'])

y = train['avg_delay_minutes_next_30m']
X = train.drop(['ID', 'layout_id', 'scenario_id', 'avg_delay_minutes_next_30m'], axis=1)
X_test = test.drop(['ID', 'layout_id', 'scenario_id'], axis=1)

X_test = X_test.reindex(columns=X.columns, fill_value=0)
print(f"train 피처: {X.shape[1]} | test 피처: {X_test.shape[1]}")

lgb_params = {
    'n_estimators': 1000, 'learning_rate': 0.0100, 'num_leaves': 140,
    'max_depth': 8, 'min_child_samples': 25, 'subsample': 0.9608,
    'colsample_bytree': 0.6607, 'reg_alpha': 1.2338, 'reg_lambda': 7.6823,
    'random_state': 42, 'verbose': -1
}
model = LGBMRegressor(**lgb_params)
model.fit(X, np.log1p(y))          # 타깃 로그 변환

pred = np.expm1(model.predict(X_test))   # 로그 역변환
pred = np.clip(pred, 0, None)            # 음수 방지

submission = pd.DataFrame({
    'ID': test['ID'],
    'avg_delay_minutes_next_30m': pred
})
submission.to_csv('submission_final.csv', index=False)
print(f"\n저장 완료: submission_final.csv {submission.shape}")
print(submission.head())
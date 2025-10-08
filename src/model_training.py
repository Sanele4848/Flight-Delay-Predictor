import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor


def train_models(train, val):
    exclude = ['arr_del15', 'arr_flights', 'flight_volume_percentile', 'capacity_utilization', 'flight_volume_category',
               'carrier_name', 'airport_name', 'carrier_airport_combo', 'delay_rate', 'avg_delay_minutes']
    feature_cols = [col for col in train.columns if col not in exclude]

    categorical = ['carrier', 'airport', 'flight_volume_category', 'operational_stress_level']
    encoders = {}
    train_enc = train.copy()
    val_enc = val.copy()

    for col in categorical:
        enc = LabelEncoder()
        train_enc[col] = enc.fit_transform(train[col].astype(str))
        encoders[col] = enc
        val_enc[col] = pd.Series(-1, index=val.index)
        mask = val[col].astype(str).isin(enc.classes_)
        val_enc.loc[mask, col] = enc.transform(val[col].astype(str)[mask])

    dont_scale = ['year', 'month', 'holiday_period', 'peak_summer', 'winter_weather_season',
                  'diversion_occurred'] + categorical
    features_to_scale = [col for col in feature_cols if
                         col not in dont_scale and train[col].dtype in ['int64', 'float64', 'int32', 'float32']]

    scaler = RobustScaler()
    train_enc[features_to_scale] = scaler.fit_transform(train_enc[features_to_scale])
    val_enc[features_to_scale] = scaler.transform(val_enc[features_to_scale])

    train_enc['delay_rate'] = np.where(train_enc['arr_flights'] > 0, train_enc['arr_del15'] / train_enc['arr_flights'],
                                       0)
    val_enc['delay_rate'] = np.where(val_enc['arr_flights'] > 0, val_enc['arr_del15'] / val_enc['arr_flights'], 0)

    y_train = train_enc['delay_rate'].values
    y_val = val_enc['delay_rate'].values
    X_train = train_enc[feature_cols]
    X_val = val_enc[feature_cols]

    models = {}
    predictions = {}

    print('Training Ridge...')
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)
    models['Ridge'] = ridge
    predictions['ridge_train'] = np.clip(ridge.predict(X_train), 0, 1)
    predictions['ridge_val'] = np.clip(ridge.predict(X_val), 0, 1)

    print('Training Decision Tree...')
    dt = DecisionTreeRegressor(max_depth=10, min_samples_split=15, min_samples_leaf=8, random_state=42)
    dt.fit(X_train, y_train)
    models['Decision Tree'] = dt
    predictions['dt_train'] = np.clip(dt.predict(X_train), 0, 1)
    predictions['dt_val'] = np.clip(dt.predict(X_val), 0, 1)

    print('Training Random Forest...')
    rf = RandomForestRegressor(n_estimators=150, max_depth=12, min_samples_split=8, min_samples_leaf=4, random_state=42,
                               n_jobs=-1)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    predictions['rf_train'] = np.clip(rf.predict(X_train), 0, 1)
    predictions['rf_val'] = np.clip(rf.predict(X_val), 0, 1)

    print('Training KNN...')
    knn = KNeighborsRegressor(n_neighbors=25, weights='distance', n_jobs=-1)
    knn.fit(X_train, y_train)
    models['KNN'] = knn
    predictions['knn_train'] = np.clip(knn.predict(X_train), 0, 1)
    predictions['knn_val'] = np.clip(knn.predict(X_val), 0, 1)

    print('Training Extra Trees...')
    et = ExtraTreesRegressor(n_estimators=120, max_depth=10, min_samples_split=12, min_samples_leaf=6, random_state=42,
                             n_jobs=-1)
    et.fit(X_train, y_train)
    models['Extra Trees'] = et
    predictions['et_train'] = np.clip(et.predict(X_train), 0, 1)
    predictions['et_val'] = np.clip(et.predict(X_val), 0, 1)

    print('Training Gradient Boosting...')
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, min_samples_split=20,
                                   min_samples_leaf=10, subsample=0.8, random_state=42)
    gb.fit(X_train, y_train)
    models['Gradient Boosting'] = gb
    predictions['gb_train'] = np.clip(gb.predict(X_train), 0, 1)
    predictions['gb_val'] = np.clip(gb.predict(X_val), 0, 1)

    return models, predictions, feature_cols, encoders, scaler, features_to_scale
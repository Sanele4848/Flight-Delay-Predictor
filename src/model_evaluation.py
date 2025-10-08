import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, RobustScaler


def evaluate_models(models, predictions, train, val, test, feature_cols):
    model_map = {'Ridge': 'ridge', 'Decision Tree': 'dt', 'Random Forest': 'rf', 'KNN': 'knn', 'Extra Trees': 'et',
                 'Gradient Boosting': 'gb'}

    results_list = []
    best_val_mae = float('inf')
    best_name = ''

    y_train = np.where(train['arr_flights'] > 0, train['arr_del15'] / train['arr_flights'], 0)
    y_val = np.where(val['arr_flights'] > 0, val['arr_del15'] / val['arr_flights'], 0)

    for name, model in models.items():
        key = model_map[name]
        train_mae = mean_absolute_error(y_train, predictions[f'{key}_train'])
        val_mae = mean_absolute_error(y_val, predictions[f'{key}_val'])
        val_r2 = r2_score(y_val, predictions[f'{key}_val'])
        val_rmse = np.sqrt(mean_squared_error(y_val, predictions[f'{key}_val']))

        results_list.append(
            {'model': name, 'train_mae': train_mae, 'val_mae': val_mae, 'val_r2': val_r2, 'val_rmse': val_rmse})

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_name = name

    print(pd.DataFrame(results_list))
    print(f'\nBest: {best_name} (Val MAE: {best_val_mae:.6f})')

    best_model = models[best_name]

    categorical = ['carrier', 'airport', 'flight_volume_category', 'operational_stress_level']
    encoders = {}
    test_enc = test.copy()

    for col in categorical:
        enc = LabelEncoder()
        enc.fit(train[col].astype(str))
        encoders[col] = enc
        test_enc[col] = pd.Series(-1, index=test.index)
        mask = test[col].astype(str).isin(enc.classes_)
        test_enc.loc[mask, col] = enc.transform(test[col].astype(str)[mask])

    dont_scale = ['year', 'month', 'holiday_period', 'peak_summer', 'winter_weather_season',
                  'diversion_occurred'] + categorical
    features_to_scale = [col for col in feature_cols if
                         col not in dont_scale and train[col].dtype in ['int64', 'float64', 'int32', 'float32']]

    scaler = RobustScaler()
    scaler.fit(train[features_to_scale])
    test_enc[features_to_scale] = scaler.transform(test_enc[features_to_scale])

    y_test = np.where(test['arr_flights'] > 0, test['arr_del15'] / test['arr_flights'], 0)
    X_test = test_enc[feature_cols]

    y_test_pred = np.clip(best_model.predict(X_test), 0, 1)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    results = {'test_mae': test_mae, 'test_r2': test_r2, 'test_rmse': test_rmse}

    return best_model, best_name, results
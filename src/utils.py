import joblib
import pandas as pd
import os


def save_artifacts(model, scaler, encoders, feature_cols, features_to_scale, train, results):
    os.makedirs('models', exist_ok=True)

    carrier_names = dict(
        train[['carrier', 'carrier_name']].drop_duplicates().values) if 'carrier_name' in train.columns else {c: c for c
                                                                                                              in train[
                                                                                                                  'carrier'].unique()}
    airport_names = dict(
        train[['airport', 'airport_name']].drop_duplicates().values) if 'airport_name' in train.columns else {a: a for a
                                                                                                              in train[
                                                                                                                  'airport'].unique()}

    lookup = train.groupby(['carrier', 'airport', 'month']).agg(
        {'arr_del15': 'sum', 'arr_flights': 'sum'}).reset_index()
    lookup['delay_probability'] = lookup['arr_del15'] / lookup['arr_flights']

    if 'arr_delay' in train.columns:
        delay_data = train.groupby(['carrier', 'airport', 'month']).agg(
            {'arr_delay': 'sum', 'arr_flights': 'sum'}).reset_index()
        delay_data['avg_delay_minutes'] = delay_data['arr_delay'] / delay_data['arr_flights']
        lookup = lookup.merge(delay_data[['carrier', 'airport', 'month', 'avg_delay_minutes']],
                              on=['carrier', 'airport', 'month'], how='left')
        lookup['avg_delay_minutes'] = lookup['avg_delay_minutes'].fillna(11.5)
    else:
        lookup['avg_delay_minutes'] = 11.5

    stats = {'overall_delay_rate': float(train['arr_del15'].sum() / train['arr_flights'].sum()),
             'avg_delay_minutes': float(lookup['avg_delay_minutes'].mean()),
             'carriers': sorted(train['carrier'].unique().tolist()),
             'airports': sorted(train['airport'].unique().tolist()), 'months': sorted(train['month'].unique().tolist()),
             'min_flights': int(train['arr_flights'].min()), 'max_flights': int(train['arr_flights'].quantile(0.995)),
             'median_flights': int(train['arr_flights'].median())}

    joblib.dump(model, 'models/best_model.pkl')
    joblib.dump(scaler, 'models/robust_scaler.pkl')
    joblib.dump(encoders, 'models/label_encoders.pkl')
    joblib.dump(feature_cols, 'models/feature_columns.pkl')
    joblib.dump(carrier_names, 'models/carrier_names.pkl')
    joblib.dump(airport_names, 'models/airport_names.pkl')
    joblib.dump(lookup, 'models/ui_lookup_table.pkl')
    joblib.dump(stats, 'models/dataset_stats.pkl')
    joblib.dump(features_to_scale, 'models/features_to_scale.pkl')

    print('Artifacts saved to models/')
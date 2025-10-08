import pandas as pd
import numpy as np


def engineer_features(df):
    if 'arr_delay' in df.columns and 'arr_flights' in df.columns:
        df['avg_delay_minutes'] = df['arr_delay'] / df['arr_flights']
        df['avg_delay_minutes'] = df['avg_delay_minutes'].fillna(0)

    if all(col in df.columns for col in
           ['carrier_delay', 'weather_delay', 'nas_delay', 'security_delay', 'late_aircraft_delay', 'arr_delay']):
        total_delay = df['arr_delay'].replace(0, np.nan)
        df['carrier_delay_pct'] = (df['carrier_delay'] / total_delay * 100).fillna(0)
        df['weather_delay_pct'] = (df['weather_delay'] / total_delay * 100).fillna(0)
        df['nas_delay_pct'] = (df['nas_delay'] / total_delay * 100).fillna(0)
        df['security_delay_pct'] = (df['security_delay'] / total_delay * 100).fillna(0)
        df['late_aircraft_delay_pct'] = (df['late_aircraft_delay'] / total_delay * 100).fillna(0)

    for ct_col, flag_col in [('weather_ct', 'weather_impact_occurred'), ('nas_ct', 'system_congestion_occurred'),
                             ('late_aircraft_ct', 'cascade_delay_occurred'), ('carrier_ct', 'carrier_issues_occurred'),
                             ('security_ct', 'security_incident_occurred')]:
        if ct_col in df.columns:
            df[flag_col] = (df[ct_col] > 0).astype(int)

    leaky = ['carrier_ct', 'weather_ct', 'nas_ct', 'security_ct', 'late_aircraft_ct', 'carrier_delay', 'weather_delay',
             'nas_delay', 'security_delay', 'late_aircraft_delay']
    df = df.drop(columns=[col for col in leaky if col in df.columns], errors='ignore')

    baseline = df['year'].min()
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['years_since_baseline'] = df['year'] - baseline
    df['holiday_period'] = df['month'].isin([6, 7, 11, 12]).astype(int)
    df['peak_summer'] = df['month'].isin([6, 7, 8]).astype(int)
    df['winter_weather_season'] = df['month'].isin([12, 1, 2, 3]).astype(int)

    df['flight_volume_category'] = pd.cut(df['arr_flights'], bins=[0, 50, 200, 500, float('inf')],
                                          labels=['small', 'medium', 'large', 'major_hub'])
    df['flight_volume_percentile'] = df['arr_flights'].rank(pct=True)
    airport_max = df.groupby('airport')['arr_flights'].transform('max')
    df['capacity_utilization'] = df['arr_flights'] / airport_max

    df['cancellation_rate'] = np.where(df['arr_flights'] > 0, df['arr_cancelled'] / df['arr_flights'], 0)
    df['diversion_rate'] = np.where(df['arr_flights'] > 0, df['arr_diverted'] / df['arr_flights'], 0)
    df['diversion_occurred'] = (df['arr_diverted'] > 0).astype(int)
    df['total_disruption_rate'] = df['cancellation_rate'] + df['diversion_rate']
    df['operational_stress_level'] = pd.cut(df['total_disruption_rate'], bins=[0, 0.01, 0.05, 0.15, float('inf')],
                                            labels=['low', 'moderate', 'high', 'severe'])

    df['carrier_airport_combo'] = df['carrier'] + '_' + df['airport']
    carrier_rate = df.groupby('carrier')['arr_del15'].sum() / df.groupby('carrier')['arr_flights'].sum()
    df['carrier_historical_delay_rate'] = df['carrier'].map(carrier_rate)
    airport_rate = df.groupby('airport')['arr_del15'].sum() / df.groupby('airport')['arr_flights'].sum()
    df['airport_historical_delay_rate'] = df['airport'].map(airport_rate)

    seasonal = df.groupby(['carrier', 'airport', 'month']).agg({'arr_del15': 'sum', 'arr_flights': 'sum'}).reset_index()
    seasonal['seasonal_delay_rate'] = seasonal['arr_del15'] / seasonal['arr_flights']
    df = df.merge(seasonal[['carrier', 'airport', 'month', 'seasonal_delay_rate']], on=['carrier', 'airport', 'month'],
                  how='left')

    if all(col in df.columns for col in
           ['carrier_delay_pct', 'weather_delay_pct', 'nas_delay_pct', 'late_aircraft_delay_pct']):
        breakdown = df.groupby(['carrier', 'airport', 'month']).agg(
            {'carrier_delay_pct': 'mean', 'weather_delay_pct': 'mean', 'nas_delay_pct': 'mean',
             'security_delay_pct': 'mean', 'late_aircraft_delay_pct': 'mean'}).reset_index()
        breakdown.columns = ['carrier', 'airport', 'month', 'avg_carrier_pct', 'avg_weather_pct', 'avg_nas_pct',
                             'avg_security_pct', 'avg_late_aircraft_pct']
        df = df.merge(breakdown, on=['carrier', 'airport', 'month'], how='left')

    df['carrier_peak_risk'] = df['carrier_historical_delay_rate'] * df['peak_summer']
    df['carrier_winter_risk'] = df['carrier_historical_delay_rate'] * df['winter_weather_season']
    df['airport_holiday_risk'] = df['airport_historical_delay_rate'] * df['holiday_period']

    # Apply preprocessing steps
    from .data_preprocessing import cap_outliers, create_log_features, reduce_cardinality, remove_duplicate_features

    df = cap_outliers(df)
    df = create_log_features(df)
    df = reduce_cardinality(df)
    df = remove_duplicate_features(df)

    print(f'Features engineered: {df.shape}')
    return df
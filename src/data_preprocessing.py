import pandas as pd
import numpy as np


def load_and_clean_data(filepath):
    df = pd.read_csv(
        'C:\\Users\\smabu\\.cache\\kagglehub\\datasets\\sriharshaeedala\\airline-delay\\versions\\1\\Airline_Delay_Cause.csv')
    print(f'Data loaded: {df.shape}')

    if 'year' in df.columns:
        before = df.shape
        df = df[~df['year'].isin([2020, 2021])].copy()
        print(f'Removed COVID years: {before} -> {df.shape}')

    if 'arr_del15' in df.columns:
        target_missing = df['arr_del15'].isnull().sum()
        if target_missing > 0:
            df = df.dropna(subset=['arr_del15'])
            print(f'Dropped {target_missing} rows with missing targets')

    if 'arr_flights' in df.columns and df['arr_flights'].isnull().sum() > 0:
        df['arr_flights'] = df.groupby(['carrier', 'airport'])['arr_flights'].transform(lambda x: x.fillna(x.median()))
        df['arr_flights'] = df['arr_flights'].fillna(df['arr_flights'].median())

    for col in ['arr_cancelled', 'arr_diverted']:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    print(f'Missing values: {df.isnull().sum().sum()}')
    return df


def cap_outliers(df):
    delay_cols = ['arr_delay', 'carrier_delay', 'weather_delay', 'nas_delay', 'late_aircraft_delay']
    for col in delay_cols:
        if col in df.columns:
            cap_value = df[col].quantile(0.99)
            df[col] = df[col].clip(upper=cap_value)
    print('Capped delay columns at 99th percentile')

    if 'avg_delay_minutes' in df.columns:
        df['avg_delay_minutes'] = df['avg_delay_minutes'].clip(upper=120)
        print('Capped avg_delay_minutes at 120 minutes')

    return df


def create_log_features(df):
    skewed_cols = ['arr_flights', 'arr_del15', 'arr_cancelled', 'arr_diverted']
    for col in skewed_cols:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col])
    print('Created log transformed features')
    return df


def reduce_cardinality(df):
    if 'carrier_airport_combo' in df.columns:
        combo_counts = df['carrier_airport_combo'].value_counts()
        rare_combos = combo_counts[combo_counts < 10].index
        df.loc[df['carrier_airport_combo'].isin(rare_combos), 'carrier_airport_combo'] = 'Other'
        print(f'Grouped {len(rare_combos)} rare carrier_airport_combos into Other')
    return df


def remove_duplicate_features(df):
    duplicate_cols = ['carrier_delay_pct', 'weather_delay_pct', 'nas_delay_pct', 'security_delay_pct',
                      'late_aircraft_delay_pct']
    existing_dupes = [col for col in duplicate_cols if col in df.columns]
    if existing_dupes:
        df = df.drop(columns=existing_dupes)
        print(f'Dropped duplicate percentage columns: {existing_dupes}')
    return df
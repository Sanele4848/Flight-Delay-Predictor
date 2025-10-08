import joblib
import pandas as pd
import numpy as np


class FlightDelayPredictor:
    def __init__(self):
        self.model = joblib.load('models/best_model.pkl')
        self.scaler = joblib.load('models/robust_scaler.pkl')
        self.encoders = joblib.load('models/label_encoders.pkl')
        self.feature_cols = joblib.load('models/feature_columns.pkl')
        self.features_to_scale = joblib.load('models/features_to_scale.pkl')
        self.carrier_names = joblib.load('models/carrier_names.pkl')
        self.airport_names = joblib.load('models/airport_names.pkl')
        self.lookup = joblib.load('models/ui_lookup_table.pkl')
        self.stats = joblib.load('models/dataset_stats.pkl')

    def predict(self, carrier, airport, month, arr_flights=100):
        lookup_match = self.lookup[
            (self.lookup['carrier'] == carrier) & (self.lookup['airport'] == airport) & (self.lookup['month'] == month)]

        if len(lookup_match) > 0:
            delay_prob = float(lookup_match['delay_probability'].iloc[0])
            avg_delay = float(lookup_match['avg_delay_minutes'].iloc[0])
        else:
            input_data = self._build_features(carrier, airport, month, arr_flights)
            X = input_data[self.feature_cols]
            delay_prob = float(np.clip(self.model.predict(X)[0], 0, 1))
            avg_delay = self.stats['avg_delay_minutes']

        risk = 'Very Low' if delay_prob < 0.15 else 'Low' if delay_prob < 0.25 else 'Moderate' if delay_prob < 0.35 else 'High'

        return {'carrier': self.carrier_names.get(carrier, carrier),
                'airport': self.airport_names.get(airport, airport), 'month': month, 'delay_probability': delay_prob,
                'avg_delay_minutes': avg_delay, 'risk_level': risk, 'expected_delays_per_100': int(delay_prob * 100)}

    def _build_features(self, carrier, airport, month, arr_flights):
        data = pd.DataFrame(
            {'year': [2023], 'month': [month], 'carrier': [carrier], 'airport': [airport], 'arr_flights': [arr_flights],
             'arr_cancelled': [int(arr_flights * 0.02)], 'arr_diverted': [int(arr_flights * 0.003)],
             'arr_delay': [int(arr_flights * 12)]})

        # Create log features
        data['arr_flights_log'] = np.log1p(data['arr_flights'])
        data['arr_del15_log'] = np.log1p(data['arr_flights'] * 0.2)
        data['arr_cancelled_log'] = np.log1p(data['arr_cancelled'])
        data['arr_diverted_log'] = np.log1p(data['arr_diverted'])

        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        data['years_since_baseline'] = data['year'] - 2013
        data['holiday_period'] = data['month'].isin([6, 7, 11, 12]).astype(int)
        data['peak_summer'] = data['month'].isin([6, 7, 8]).astype(int)
        data['winter_weather_season'] = data['month'].isin([12, 1, 2, 3]).astype(int)

        data['cancellation_rate'] = data['arr_cancelled'] / data['arr_flights']
        data['diversion_rate'] = data['arr_diverted'] / data['arr_flights']
        data['diversion_occurred'] = (data['arr_diverted'] > 0).astype(int)
        data['total_disruption_rate'] = data['cancellation_rate'] + data['diversion_rate']
        data['operational_stress_level'] = 'moderate'

        for flag in ['weather_impact_occurred', 'system_congestion_occurred', 'cascade_delay_occurred',
                     'carrier_issues_occurred', 'security_incident_occurred']:
            data[flag] = 0

        carrier_avg = self.lookup[self.lookup['carrier'] == carrier]['delay_probability'].mean()
        airport_avg = self.lookup[self.lookup['airport'] == airport]['delay_probability'].mean()
        seasonal_avg = self.lookup[(self.lookup['carrier'] == carrier) & (self.lookup['airport'] == airport) & (
                    self.lookup['month'] == month)]['delay_probability'].mean()

        data['carrier_historical_delay_rate'] = carrier_avg if not pd.isna(carrier_avg) else 0.196
        data['airport_historical_delay_rate'] = airport_avg if not pd.isna(airport_avg) else 0.193
        data['seasonal_delay_rate'] = seasonal_avg if not pd.isna(seasonal_avg) else 0.199

        for col in ['avg_carrier_pct', 'avg_weather_pct', 'avg_nas_pct', 'avg_security_pct', 'avg_late_aircraft_pct']:
            data[col] = [37.48, 5.65, 19.42, 0.19, 34.60][
                ['avg_carrier_pct', 'avg_weather_pct', 'avg_nas_pct', 'avg_security_pct',
                 'avg_late_aircraft_pct'].index(col)]

        data['carrier_peak_risk'] = data['carrier_historical_delay_rate'] * data['peak_summer']
        data['carrier_winter_risk'] = data['carrier_historical_delay_rate'] * data['winter_weather_season']
        data['airport_holiday_risk'] = data['airport_historical_delay_rate'] * data['holiday_period']

        for col in ['carrier', 'airport', 'operational_stress_level']:
            if col in self.encoders:
                if data[col].iloc[0] in self.encoders[col].classes_:
                    data[col] = self.encoders[col].transform([data[col].iloc[0]])[0]
                else:
                    data[col] = -1

        cols_to_scale = [col for col in self.features_to_scale if col in data.columns]
        data[cols_to_scale] = self.scaler.transform(data[cols_to_scale])

        return data
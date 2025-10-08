import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import engineer_features
from src.model_training import train_models
from src.model_evaluation import evaluate_models
from src.utils import save_artifacts


def generate_research_report_data(train, val, test, best_model, feature_cols, results, encoders, scaler,
                                  features_to_scale):
    """Generate comprehensive statistics for research report"""
    print("\n" + "=" * 80)
    print("RESEARCH REPORT DATA GENERATION")
    print("=" * 80)

    # =========================================================================
    # 1. FEATURE IMPORTANCE ANALYSIS
    # =========================================================================
    print("\n1. FEATURE IMPORTANCE ANALYSIS")
    print("-" * 80)

    if hasattr(best_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 20 Most Important Features:")
        print(importance_df.head(20).to_string(index=False))

        # Save to CSV
        importance_df.to_csv('models/feature_importance.csv', index=False)
        print("\n✓ Saved to: models/feature_importance.csv")

    # =========================================================================
    # 2. SEASONAL PATTERNS
    # =========================================================================
    print("\n2. SEASONAL DELAY PATTERNS")
    print("-" * 80)

    # Calculate delay rate for each month
    full_data = pd.concat([train, val, test])
    full_data['delay_rate'] = np.where(
        full_data['arr_flights'] > 0,
        full_data['arr_del15'] / full_data['arr_flights'],
        0
    )

    monthly_stats = full_data.groupby('month').agg({
        'delay_rate': ['mean', 'std', 'count'],
        'arr_del15': 'sum',
        'arr_flights': 'sum',
        'avg_delay_minutes': 'mean'
    }).round(4)

    print("\nMonthly Delay Statistics:")
    print(monthly_stats)
    monthly_stats.to_csv('models/monthly_patterns.csv')
    print("\n✓ Saved to: models/monthly_patterns.csv")

    # =========================================================================
    # 3. CARRIER PERFORMANCE ANALYSIS
    # =========================================================================
    print("\n3. CARRIER PERFORMANCE ANALYSIS")
    print("-" * 80)

    carrier_stats = full_data.groupby('carrier').agg({
        'delay_rate': ['mean', 'std', 'count'],
        'arr_del15': 'sum',
        'arr_flights': 'sum',
        'avg_delay_minutes': 'mean'
    }).round(4)

    carrier_stats = carrier_stats.sort_values(('delay_rate', 'mean'))

    print("\nTop 10 Best Performing Carriers:")
    print(carrier_stats.head(10))

    print("\nTop 10 Worst Performing Carriers:")
    print(carrier_stats.tail(10))

    carrier_stats.to_csv('models/carrier_performance.csv')
    print("\n✓ Saved to: models/carrier_performance.csv")

    # =========================================================================
    # 4. AIRPORT PERFORMANCE ANALYSIS
    # =========================================================================
    print("\n4. AIRPORT PERFORMANCE ANALYSIS")
    print("-" * 80)

    airport_stats = full_data.groupby('airport').agg({
        'delay_rate': ['mean', 'std', 'count'],
        'arr_del15': 'sum',
        'arr_flights': 'sum',
        'avg_delay_minutes': 'mean'
    }).round(4)

    airport_stats = airport_stats.sort_values(('delay_rate', 'mean'))

    print("\nTop 10 Best Performing Airports:")
    print(airport_stats.head(10))

    print("\nTop 10 Worst Performing Airports:")
    print(airport_stats.tail(10))

    airport_stats.to_csv('models/airport_performance.csv')
    print("\n✓ Saved to: models/airport_performance.csv")

    # =========================================================================
    # 5. YEARLY TRENDS
    # =========================================================================
    print("\n5. YEARLY TREND ANALYSIS")
    print("-" * 80)

    yearly_stats = full_data.groupby('year').agg({
        'delay_rate': ['mean', 'std', 'count'],
        'arr_del15': 'sum',
        'arr_flights': 'sum',
        'avg_delay_minutes': 'mean'
    }).round(4)

    print("\nYear-over-Year Delay Rates:")
    print(yearly_stats)
    yearly_stats.to_csv('models/yearly_trends.csv')
    print("\n✓ Saved to: models/yearly_trends.csv")

    # =========================================================================
    # 6. DELAY CAUSE ATTRIBUTION
    # =========================================================================
    print("\n6. DELAY CAUSE ATTRIBUTION")
    print("-" * 80)

    # Calculate average percentages for each delay cause
    delay_causes = ['avg_carrier_pct', 'avg_weather_pct', 'avg_nas_pct',
                    'avg_security_pct', 'avg_late_aircraft_pct']

    if all(col in full_data.columns for col in delay_causes):
        cause_stats = full_data[delay_causes].mean().sort_values(ascending=False)
        print("\nAverage Delay Attribution (% of total delay minutes):")
        for cause, pct in cause_stats.items():
            print(f"  {cause.replace('avg_', '').replace('_pct', '').title()}: {pct:.2f}%")

    # =========================================================================
    # 7. OPERATIONAL STRESS ANALYSIS
    # =========================================================================
    print("\n7. OPERATIONAL STRESS LEVEL ANALYSIS")
    print("-" * 80)

    if 'operational_stress_level' in full_data.columns:
        stress_stats = full_data.groupby('operational_stress_level').agg({
            'delay_rate': 'mean',
            'arr_del15': 'sum',
            'arr_flights': 'sum'
        }).round(4)

        print("\nDelay Rate by Operational Stress Level:")
        print(stress_stats)
        stress_stats.to_csv('models/stress_level_analysis.csv')
        print("\n✓ Saved to: models/stress_level_analysis.csv")

    # =========================================================================
    # 8. ERROR ANALYSIS BY SEGMENTS
    # =========================================================================
    print("\n8. ERROR ANALYSIS BY SEGMENTS")
    print("-" * 80)

    # Encode and scale test data (same as in model_evaluation.py)
    categorical = ['carrier', 'airport', 'flight_volume_category', 'operational_stress_level']
    test_enc = test.copy()

    for col in categorical:
        if col in encoders:
            test_enc[col] = pd.Series(-1, index=test.index)
            mask = test[col].astype(str).isin(encoders[col].classes_)
            test_enc.loc[mask, col] = encoders[col].transform(test[col].astype(str)[mask])

    # Scale features
    test_enc[features_to_scale] = scaler.transform(test_enc[features_to_scale])

    # Get predictions on properly encoded test set
    X_test = test_enc[feature_cols]
    y_test = np.where(test['arr_flights'] > 0,
                      test['arr_del15'] / test['arr_flights'], 0)
    y_pred = np.clip(best_model.predict(X_test), 0, 1)

    # Create error analysis DataFrame
    test_analysis = test.copy()
    test_analysis['y_true'] = y_test
    test_analysis['y_pred'] = y_pred
    test_analysis['abs_error'] = np.abs(y_test - y_pred)

    # By flight volume
    print("\nError Analysis by Flight Volume:")
    volume_bins = [0, 50, 200, 500, float('inf')]
    volume_labels = ['Low (<50)', 'Medium (50-200)', 'High (200-500)', 'Major (>500)']
    test_analysis['volume_category'] = pd.cut(test_analysis['arr_flights'],
                                              bins=volume_bins, labels=volume_labels)

    volume_errors = test_analysis.groupby('volume_category').agg({
        'abs_error': ['mean', 'median', 'std', 'count']
    }).round(4)
    print(volume_errors)

    # By delay rate ranges
    print("\nError Analysis by Delay Rate Range:")
    delay_bins = [0, 0.15, 0.25, 0.35, 1.0]
    delay_labels = ['Very Low (<15%)', 'Moderate (15-25%)',
                    'High (25-35%)', 'Very High (>35%)']
    test_analysis['delay_category'] = pd.cut(test_analysis['y_true'],
                                             bins=delay_bins, labels=delay_labels)

    delay_errors = test_analysis.groupby('delay_category').agg({
        'abs_error': ['mean', 'median', 'std', 'count'],
        'y_true': 'mean',
        'y_pred': 'mean'
    }).round(4)
    print(delay_errors)

    test_analysis[['carrier', 'airport', 'month', 'y_true', 'y_pred',
                   'abs_error']].to_csv('models/test_predictions.csv', index=False)
    print("\n✓ Saved to: models/test_predictions.csv")

    # =========================================================================
    # 9. DATASET STATISTICS
    # =========================================================================
    print("\n9. OVERALL DATASET STATISTICS")
    print("-" * 80)

    overall_stats = {
        'Total Records': len(full_data),
        'Unique Carriers': full_data['carrier'].nunique(),
        'Unique Airports': full_data['airport'].nunique(),
        'Date Range': f"{full_data['year'].min()}-{full_data['year'].max()}",
        'Total Flights': full_data['arr_flights'].sum(),
        'Total Delays (15+ min)': full_data['arr_del15'].sum(),
        'Overall Delay Rate': (full_data['arr_del15'].sum() /
                               full_data['arr_flights'].sum()),
        'Avg Delay Duration (min)': full_data['avg_delay_minutes'].mean(),
        'Training Set Size': len(train),
        'Validation Set Size': len(val),
        'Test Set Size': len(test)
    }

    print("\nDataset Overview:")
    for key, value in overall_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")

    # =========================================================================
    # 10. SAMPLE PREDICTIONS FOR CASE STUDIES
    # =========================================================================
    print("\n10. SAMPLE PREDICTIONS FOR CASE STUDIES")
    print("-" * 80)

    # Find interesting examples
    test_analysis_sorted = test_analysis.sort_values('y_true', ascending=False)

    print("\nHigh Risk Routes (Top 5):")
    high_risk = test_analysis_sorted.head(5)[['carrier', 'airport', 'month',
                                              'y_true', 'y_pred', 'abs_error']]
    print(high_risk.to_string(index=False))

    test_analysis_sorted = test_analysis.sort_values('y_true', ascending=True)
    print("\nLow Risk Routes (Top 5):")
    low_risk = test_analysis_sorted.head(5)[['carrier', 'airport', 'month',
                                             'y_true', 'y_pred', 'abs_error']]
    print(low_risk.to_string(index=False))

    print("\n" + "=" * 80)
    print("DATA GENERATION COMPLETE!")
    print("=" * 80)
    print("\nAll analysis files saved to models/ directory")
    print("Use these results to update your research report with actual data.")

    return {
        'monthly_stats': monthly_stats,
        'carrier_stats': carrier_stats,
        'airport_stats': airport_stats,
        'yearly_stats': yearly_stats,
        'overall_stats': overall_stats
    }


def main():
    print('Loading data...')
    df = load_and_clean_data('data/Airline_Delay_Cause.csv')

    print('Engineering features...')
    df = engineer_features(df)

    print('Splitting data...')
    df = df.sort_values(['year', 'month'])
    train = df[df['year'].isin([2013, 2014, 2015, 2016, 2017, 2018])]
    val = df[df['year'].isin([2019])]
    test = df[df['year'].isin([2022, 2023])]

    print('Training models...')
    models, predictions, feature_cols, encoders, scaler, features_to_scale = train_models(train, val)

    print('Evaluating models...')
    best_model, best_name, results = evaluate_models(models, predictions, train, val, test, feature_cols)

    print('Saving artifacts...')
    save_artifacts(best_model, scaler, encoders, feature_cols, features_to_scale, train, results)

    print(f'\nBest Model: {best_name}')
    print(f'Test MAE: {results["test_mae"]:.6f}')
    print(f'Test R2: {results["test_r2"]:.4f}')

    # Generate comprehensive research report data
    print('\n\nGenerating research report data...')
    report_data = generate_research_report_data(
        train, val, test, best_model, feature_cols, results, encoders, scaler, features_to_scale
    )


if __name__ == '__main__':
    main()
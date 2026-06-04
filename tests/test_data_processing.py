# tests/test_data_processing.py
import pytest
import pandas as pd
from src.data_processing import DefensiveFeatureExtractor

def test_defensive_feature_extractor_valid_input():
    # Arrange dummy records
    data = {
        'TransactionStartTime': ['2018-12-24T16:30:13Z'],
        'Amount': [5000.0],
        'CustomerId': ['CustomerId_7343'],
        'FraudResult': [0],
        'Value': [5000]
    }
    df = pd.DataFrame(data)
    extractor = DefensiveFeatureExtractor()

    # Act
    transformed = extractor.transform(df)

    # Assert
    assert 'Hour' in transformed.columns
    assert 'Value' not in transformed.columns  # Multicolliniarity fix verification
    assert transformed['Hour'].iloc[0] == 16

def test_defensive_feature_extractor_missing_columns():
    invalid_data = {'Amount': [5000.0]}
    df = pd.DataFrame(invalid_data)
    extractor = DefensiveFeatureExtractor()

    with pytest.raises(KeyError):
        extractor.transform(df)
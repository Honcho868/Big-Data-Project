import pickle
import numpy as np
from itertools import product

# Load the model
with open('rf_clf.pkl', 'rb') as f:
    model = pickle.load(f)

def create_features(params):
    """Create feature vector based on input parameters"""
    features = []
    
    # Numerical features
    features.extend([
        params['lead_time'],  # shorter lead times are better
        params['stays_in_weekend_nights'],
        params['stays_in_week_nights'],
        params['adults'],
        params['children'],
        params['babies'],
        params['is_repeated_guest'],
        params['previous_cancellations'],
        params['previous_bookings_not_canceled'],
        params['booking_changes'],
        params['days_in_waiting_list'],
        0,  # adr (not required)
        params['required_car_parking_spaces'],
        params['total_of_special_requests']
    ])
    
    # Categorical features (one-hot encoded)
    categorical_features = {
        'meal': ['BB', 'FB', 'HB', 'SC', 'Undefined'],
        'market_segment': ['Direct', 'Corporate', 'Online TA', 'Offline TA/TO', 'Groups', 'Aviation'],
        'distribution_channel': ['Direct', 'Corporate', 'TA/TO', 'GDS'],
        'reserved_room_type': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
        'assigned_room_type': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K'],
        'deposit_type': ['No Deposit', 'Non Refund', 'Refundable'],
        'customer_type': ['Transient', 'Contract', 'Transient-Party', 'Group'],
        'country': ['PRT', 'GBR', 'USA', 'ESP', 'IRL', 'FRA', 'ROU', 'NOR', 'OMN', 'ARG', 'POL', 'DEU', 'BEL', 'CHE', 'CN', 'GRC', 'ITA', 'NLD', 'DNK', 'RUS', 'Other']
    }
    
    for feature, value in params.items():
        if feature in categorical_features:
            encoding = [0] * len(categorical_features[feature])
            try:
                idx = categorical_features[feature].index(value)
                encoding[idx] = 1
            except ValueError:
                encoding[0] = 1  # default to first category
            features.extend(encoding)
    
    return features

# Test cases that are likely to result in "Not Cancelled"
test_cases = [
    {
        # Case 1: Loyal customer with good history
        'lead_time': 7,  # Short lead time
        'stays_in_weekend_nights': 2,
        'stays_in_week_nights': 3,
        'adults': 2,
        'children': 0,
        'babies': 0,
        'is_repeated_guest': 1,  # Repeated guest
        'previous_cancellations': 0,  # No previous cancellations
        'previous_bookings_not_canceled': 5,  # Good history
        'booking_changes': 0,
        'days_in_waiting_list': 0,
        'required_car_parking_spaces': 0,
        'total_of_special_requests': 2,  # Some special requests
        'meal': 'BB',
        'market_segment': 'Direct',  # Direct booking
        'distribution_channel': 'Direct',
        'reserved_room_type': 'A',
        'assigned_room_type': 'A',  # Same as reserved
        'deposit_type': 'Non Refund',  # Non-refundable booking
        'customer_type': 'Transient',
        'country': 'PRT'  # Local guest
    },
    {
        # Case 2: Business traveler
        'lead_time': 14,
        'stays_in_weekend_nights': 0,
        'stays_in_week_nights': 4,
        'adults': 1,
        'children': 0,
        'babies': 0,
        'is_repeated_guest': 1,
        'previous_cancellations': 0,
        'previous_bookings_not_canceled': 3,
        'booking_changes': 1,
        'days_in_waiting_list': 0,
        'required_car_parking_spaces': 0,
        'total_of_special_requests': 1,
        'meal': 'BB',
        'market_segment': 'Corporate',  # Corporate booking
        'distribution_channel': 'Corporate',
        'reserved_room_type': 'D',
        'assigned_room_type': 'D',
        'deposit_type': 'Non Refund',
        'customer_type': 'Contract',  # Contract customer
        'country': 'GBR'
    }
]

print("Testing combinations likely to result in 'Not Cancelled':")
print("-" * 50)

for i, test_case in enumerate(test_cases, 1):
    features = create_features(test_case)
    prediction = model.predict([features])[0]
    probability = model.predict_proba([features])[0]
    
    print(f"\nTest Case {i}:")
    print(f"Prediction: {'Not Cancelled' if prediction == 0 else 'Cancelled'}")
    print(f"Confidence: {max(probability) * 100:.2f}%")
    print("\nKey Features:")
    for key, value in test_case.items():
        print(f"- {key}: {value}")
    print("-" * 50)

if __name__ == '__main__':
    print("\nRun this script to test different feature combinations") 
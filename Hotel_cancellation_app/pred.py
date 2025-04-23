from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained model
with open('rf_clf.pkl', 'rb') as f:
    model = pickle.load(f)

# Define categorical features and their possible values
CATEGORICAL_FEATURES = {
    'meal': ['BB', 'FB', 'HB', 'SC', 'Undefined'],  # 5
    'market_segment': ['Direct', 'Corporate', 'Online TA', 'Offline TA/TO', 'Groups', 'Aviation'],  # 6
    'distribution_channel': ['Direct', 'Corporate', 'TA/TO', 'GDS'],  # 4
    'reserved_room_type': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],  # 8
    'assigned_room_type': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K'],  # 10
    'deposit_type': ['No Deposit', 'Non Refund', 'Refundable'],  # 3
    'customer_type': ['Transient', 'Contract', 'Transient-Party', 'Group'],  # 4
    'country': ['PRT', 'GBR', 'USA', 'ESP', 'IRL', 'FRA', 'ROU', 'NOR', 'OMN', 'ARG', 'POL', 'DEU', 'BEL', 'CHE', 'CN', 'GRC', 'ITA', 'NLD', 'DNK', 'RUS', 'Other']  # 21
}  # Total categorical: 61 + numerical: 14 = 75 features

# Define numerical features in order
NUMERICAL_FEATURES = [
    'lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 
    'adults', 'children', 'babies', 'is_repeated_guest', 
    'previous_cancellations', 'previous_bookings_not_canceled',
    'booking_changes', 'days_in_waiting_list', 'adr',
    'required_car_parking_spaces', 'total_of_special_requests'
]

def one_hot_encode(field_name, value):
    """Convert a single categorical value to one-hot encoding"""
    categories = CATEGORICAL_FEATURES[field_name]
    encoding = [0] * len(categories)
    try:
        if field_name == 'country':
            value = value.upper()
            if value not in categories:
                value = 'Other'
        idx = categories.index(value)
        encoding[idx] = 1
    except ValueError:
        # If value not found, use first category as default
        encoding[0] = 1
    return encoding

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = request.form.to_dict()
        
        # Initialize features list
        features = []
        
        # Process numerical features first
        for field_name in NUMERICAL_FEATURES:
            if field_name == 'adr':  # Skip adr as it's not required
                features.append(0)  # Use 0 as default for adr
                continue
            try:
                value = float(form_data.get(field_name, 0))
                features.append(value)
            except ValueError:
                return render_template('index.html', 
                                    prediction_text=f'Error: {field_name} must be a number')
        
        # Process categorical features
        for field_name in CATEGORICAL_FEATURES:
            value = form_data.get(field_name, '')
            one_hot = one_hot_encode(field_name, value)
            features.extend(one_hot)
        
        # Print feature length for debugging
        print(f"Number of features: {len(features)}")
        
        # Make prediction
        prediction = model.predict([features])[0]
        result = "Cancelled" if prediction == 1 else "Not Cancelled"
        
        return render_template('index.html', 
                             prediction_text=f'Prediction: {result}')
    
    except Exception as e:
        return render_template('index.html', 
                             prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
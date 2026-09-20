import pandas as pd
import joblib

#load saved model
model = joblib.load('car_model.joblib')
model_columns = joblib.load('model_columns.joblib')

new_car = pd.DataFrame ( [ { 
    'make_year' : 2023, 
    'mileage_kmpl' : 18.0,
    'engine_cc' :2000,
    'fuel_type' : 'Petrol',
    'owner_count' : 1,
    'brand' : 'Toyota',
    'transmission': 'Automatic',
    'color' : 'Black',
    'service_history': 'Full',
    'accidents_reported': 0,
    'Insurance_valid':  'Yes'

} ] )
new_car_encoded = pd.get_dummies(new_car, dtype=int)
new_car_alligned = new_car_encoded.reindex(columns=model_columns, fill_value=0)

predicted_price = model.predict(new_car_alligned)[0]
print(f"Predicted Car Price: ${predicted_price}")     



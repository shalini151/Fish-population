import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Set Streamlit page config
st.set_page_config(page_title="Fish Population Predictor", layout="centered")

# Title
st.title("🐟 Fish Population Predictor")
st.write("Enter water temperature and pH to predict the fish population.")

# Sample data
data = {
    'water_temperature': [20, 22, 21, 20, 19, 18, 17, 16, 15, 14],
    'pH': [7.5, 7.6, 7.4, 7.3, 7.2, 7.1, 7.0, 6.9, 6.8, 6.7],
    'fish_population': [100, 120, 110, 130, 140, 150, 160, 170, 180, 190]
}
df = pd.DataFrame(data)

X = df[['water_temperature', 'pH']]
y = df['fish_population']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
rf_mse = mean_squared_error(y_test, rf_model.predict(X_test))

# Train Neural Network
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(2,)),
    Dense(32, activation='relu'),
    Dense(1)
])
nn_model.compile(optimizer='adam', loss='mean_squared_error')
nn_model.fit(X_train, y_train, epochs=100, verbose=0)
nn_mse = mean_squared_error(y_test, nn_model.predict(X_test).flatten())

# UI inputs
temp = st.number_input("Water Temperature (°C)", min_value=0.0, step=0.1)
ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, step=0.1)

# Predict
if st.button("Predict Fish Population"):
    input_df = pd.DataFrame([[temp, ph]], columns=['water_temperature', 'pH'])
    
    rf_prediction = rf_model.predict(input_df)[0]
    nn_prediction = nn_model.predict(input_df).flatten()[0]

    st.subheader("📊 Predictions:")
    st.write(f"🔹 Random Forest Prediction: **{rf_prediction:.2f}** fish")
    st.write(f"🔹 Neural Network Prediction: **{nn_prediction:.2f}** fish")
    st.write("---")
    st.write(f"🧪 Random Forest MSE: `{rf_mse:.2f}`")
    st.write(f"🧪 Neural Network MSE: `{nn_mse:.2f}`")

from flask import Flask, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

app = Flask(__name__)

# Sample dataset (static for now)
data = {
    'water_temperature': [20, 22, 21, 20, 19, 18, 17, 16, 15, 14],
    'pH': [7.5, 7.6, 7.4, 7.3, 7.2, 7.1, 7.0, 6.9, 6.8, 6.7],
    'fish_population': [100, 120, 110, 130, 140, 150, 160, 170, 180, 190]
}
df = pd.DataFrame(data)
X = df[['water_temperature', 'pH']]
y = df['fish_population']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
rf_mse = mean_squared_error(y_test, rf_model.predict(X_test))

# Train Neural Network model
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(2,)),
    Dense(32, activation='relu'),
    Dense(1)
])
nn_model.compile(optimizer='adam', loss='mean_squared_error')
nn_model.fit(X_train, y_train, epochs=100, verbose=0)
nn_mse = mean_squared_error(y_test, nn_model.predict(X_test).flatten())


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_rf = None
    prediction_nn = None
    if request.method == 'POST':
        try:
            temp = float(request.form['temperature'])
            ph = float(request.form['ph'])
            input_data = np.array([[temp, ph]])
            prediction_rf = rf_model.predict(input_data)[0]
            prediction_nn = nn_model.predict(input_data).flatten()[0]
        except Exception as e:
            prediction_rf = "Invalid input"
            prediction_nn = "Invalid input"

    return render_template('index.html',
                           prediction_rf=prediction_rf,
                           prediction_nn=prediction_nn,
                           mse_rf=rf_mse,
                           mse_nn=nn_mse)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

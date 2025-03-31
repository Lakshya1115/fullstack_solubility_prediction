from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle  # To save/load models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load dataset and train models
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/delaney_solubility_with_descriptors.csv')
y = df['logS']
X = df.drop('logS', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=100)

lr = LinearRegression().fit(X_train, y_train)
rf = RandomForestRegressor(max_depth=2, random_state=100).fit(X_train, y_train)

# Save models
pickle.dump(lr, open("linear_model.pkl", "wb"))
pickle.dump(rf, open("random_forest.pkl", "wb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # Receive JSON data from frontend
    features = np.array(data["features"]).reshape(1, -1)  # Convert to NumPy array

    # Load models
    lr_model = pickle.load(open("linear_model.pkl", "rb"))
    rf_model = pickle.load(open("random_forest.pkl", "rb"))

    # Make predictions
    lr_pred = lr_model.predict(features)[0]
    rf_pred = rf_model.predict(features)[0]

    return jsonify({"Linear Regression": lr_pred, "Random Forest": rf_pred})

if __name__ == "__main__":
    app.run(debug=True)

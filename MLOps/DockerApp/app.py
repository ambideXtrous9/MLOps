from flask import Flask, render_template, request
import pandas as pd
import mlflow
from mlflow import MlflowClient
import os 

# Set Google application credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'mlops-437407-225d42b6661e.json'


# Set MLflow tracking URI
mlflow.set_tracking_uri("http://34.47.170.249:5000/")
client = MlflowClient(tracking_uri="http://34.47.170.249:5000/")

# Load the model
model_name = "nyctaxi_tripduration_regressor"
stage = "Production"
model = mlflow.pyfunc.load_model(f"models:/{model_name}/{stage}")

app = Flask(__name__)

# Route to display the form
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None  # Initialize prediction variable
    if request.method == "POST":
        # Get form values
        form_data = {
            "PULocationID": int(request.form.get("PULocationID")),
            "DOLocationID": int(request.form.get("DOLocationID")),
            "trip_distance": float(request.form.get("trip_distance")),
            "fare_amount": float(request.form.get("fare_amount")),
            "extra": float(request.form.get("extra")),
            "mta_tax": float(request.form.get("mta_tax")),
            "tip_amount": float(request.form.get("tip_amount")),
            "tolls_amount": float(request.form.get("tolls_amount")),
            "improvement_surcharge": float(request.form.get("improvement_surcharge")),
            "total_amount": float(request.form.get("total_amount"))
        }

        # Create DataFrame for prediction
        trip_data = pd.DataFrame([form_data])

        # Make the prediction
        pred = model.predict(trip_data)
        prediction = pred[0]  # Get the first prediction result

    # Default values for the form
    defaults = {
        "PULocationID": 43,
        "DOLocationID": 151,
        "trip_distance": 1.01,
        "fare_amount": 5.5,
        "extra": 0.5,
        "mta_tax": 0.5,
        "tip_amount": 0.00,
        "tolls_amount": 0.0,
        "improvement_surcharge": 0.3,
        "total_amount": 6.80
    }

    return render_template("form.html", defaults=defaults, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)

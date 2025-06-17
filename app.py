from flask import Flask, request, render_template, send_file
from sklearn.datasets import load_breast_cancer
import csv
import datetime
import joblib
import os
import numpy as np
import pandas as pd
import io

app = Flask(__name__)

# Load model
model = joblib.load('model/model.pkl')

# Load dataset and feature names
data = load_breast_cancer()
feature_names = data.feature_names
df = pd.DataFrame(data.data, columns=feature_names)
sample_row = df.iloc[0].to_dict()


@app.route('/')
def home():
    return render_template('index.html', feature_names=feature_names, sample=sample_row)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_values = [float(request.form.get(name)) for name in feature_names]
        input_array = np.array(input_values).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        result = 'Benign' if prediction == 1 else 'Malignant'

        # Log the request
        log_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'prediction': result,
        }
        for name, value in zip(feature_names, input_values):
            log_entry[name] = value

        log_file = 'logs/predictions_log.csv'
        os.makedirs('logs', exist_ok=True)
        write_header = not os.path.exists(log_file)

        with open(log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp'] + list(feature_names) + ['prediction'])
            if write_header:
                writer.writeheader()
            writer.writerow(log_entry)

        return render_template('result.html', prediction=result)

    except Exception as e:
        return f"Error: {e}"


@app.route('/dashboard')
def dashboard():
    try:
        log_file = 'logs/predictions_log.csv'
        if not os.path.exists(log_file):
            return "No prediction logs found yet. Submit a prediction first."

        df = pd.read_csv(log_file)

        total = len(df)
        malignant = (df['prediction'] == 'Malignant').sum()
        benign = (df['prediction'] == 'Benign').sum()

        return render_template('dashboard.html',
                               total=total,
                               malignant=malignant,
                               benign=benign)
    except Exception as e:
        return f"Dashboard error: {e}"


@app.route('/chart')
def chart():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:
        log_file = 'logs/predictions_log.csv'
        if not os.path.exists(log_file):
            return "No prediction logs found yet. Submit a prediction first."

        df = pd.read_csv(log_file)
        if df.empty or 'prediction' not in df.columns:
            return "No valid prediction data found."

        counts = df['prediction'].value_counts()
        labels = counts.index
        sizes = counts.values

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        plt.title("Prediction Distribution")

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        return send_file(img, mimetype='image/png')
    except Exception as e:
        return f"Chart error: {e}"


if __name__ == '__main__':
    app.run(debug=True)

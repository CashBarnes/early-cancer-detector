# Early Cancer Detector

This project demonstrates a full-stack machine learning application built to simulate a breast
cancer diagnostic tool using open medical data. It includes a trained classification model, web
interface, logging, monitoring, and cloud deployment, making it suitable for portfolios in data
engineering, machine learning, and applied AI.

## Overview

- **Model:** Logistic Regression (scikit-learn)
- **Dataset:** Breast Cancer Wisconsin Diagnostic Dataset (via scikit-learn)
- **Tech Stack:** Python, Flask, HTML/CSS, matplotlib, scikit-learn
- **Hosting:** Render.com (free-tier)

## Features

- Web form with 30 input features for prediction
- Logistic regression model trained on real medical data
- Result page showing whether the prediction is benign or malignant
- Logging of each prediction to a CSV file
- Monitoring dashboard with prediction counts and a live pie chart

## Project Structure

```
early-cancer-detector/
├── app.py                  # Flask web app
├── train_model.py          # Script to train and save the model
├── model/model.pkl         # Trained model file
├── logs/                   # Prediction logs (CSV)
├── templates/              # HTML templates (form, result, dashboard)
├── static/styles.css       # CSS styles for layout
├── requirements.txt        # Python dependencies
├── Procfile                # Entry point for Render deployment
└── render.yaml             # Optional Render config file
```

## Setup Instructions (Local)

1. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Train and save the model:

```bash
python train_model.py
```

4. Start the Flask app:

```bash
python app.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

## Deployment

This project is configured to deploy on Render. To deploy:

- Push this repository to GitHub
- Create a new Web Service on https://render.com
- Use the following settings:
  - Build Command: `pip install -r requirements.txt`
  - Start Command: `gunicorn app:app`

You can optionally use the included `render.yaml` file for automatic configuration.

## Dataset Source

- Breast Cancer Wisconsin Diagnostic Dataset  
  [scikit-learn documentation](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset)

## Disclaimer

This application is for educational and demonstration purposes only. It is not intended for medical use or diagnosis. The model is trained on open-source data and is not approved for clinical decision-making. Do not use this tool to guide real-world medical care.


# Early Cancer Detector

This is a full-stack machine learning application built as a demo for a data-focused role at GRAIL. It demonstrates the use of logistic regression to classify breast cancer diagnoses using features derived from biopsy images. The project includes model training, a web interface, a logging and monitoring system, and cloud deployment.

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

## Acknowledgments

This project is inspired by GRAIL's mission to detect cancer early and demonstrates applied machine learning using open biomedical data.

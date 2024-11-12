# Sentiment Analysis Model - Continuous Training & Deployment Pipeline

This repository contains a continuous training and deployment pipeline for a sentiment analysis model using the `bert-base-uncased` model. The pipeline monitors model performance, triggers retraining when necessary, and deploys updates seamlessly. This setup includes both live and batch prediction capabilities.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Setup Instructions](#setup-instructions)
3. [Deployment](#deployment)
4. [API Usage](#api-usage)
5. [Model Monitoring & Retraining](#model-monitoring-and-retraining)

---

## 1. Prerequisites

Ensure the following are installed:

- Python 3.8+
- Virtual environment (optional but recommended)
- Pip (Python package manager)

Install the required dependencies by running:
```bash
pip install -r requirements.txt
```

## 2. Setup Instructions

### Prepare the Data

Ensure the training and validation datasets are available in the project directory with the following names:
- `train_sentiment_data.csv`
- `val_sentiment_data.csv`

Each file should contain two columns:
- `text` - input text data for sentiment analysis
- `label` - the sentiment label (e.g., Positive, Negative)

## 3. Deployment

To train the model locally and start the FastAPI service using Uvicorn, execute the following command:
```bash
python app.py
```
This script will:
- Load and prepare the data.
- Train the sentiment classification model.
- Save the trained model and tokenizer to the `./model` directory.
- Start the FastAPI service on localhost at port 8000.

## 4. API Usage

After starting the service, the following endpoints are available:

### a. Live Prediction

- **URL:** `POST http://localhost:8000/predict/live`
- **Description:** Provides a single text input for sentiment prediction.
- **Example Request:**
    ```json
    {
        "text": "I love this product!"
    }
    ```
- **Example Response:**
    ```json
    {
        "predictions": {
            "label": {
                "0": "Positive"
            },
            "score": {
                "0": 0.9875
            }
        }
    }
    ```

### b. Batch Prediction

- **URL:** `POST http://localhost:8000/predict/batch`
- **Description:** Accepts a list of text inputs and returns sentiment predictions for each.
- **Example Request:**
    ```json
    {
        "texts": ["This is terrible.", "I love this product!"]
    }
    ```
- **Example Response:**
    ```json
    {
        "predictions": {
            "label": {
                "0": "Negative",
                "1": "Positive"
            },
            "score": {
                "0": 0.8228,
                "1": 0.9875
            }
        }
    }
    ```

## 5. Model Monitoring & Retraining

This system includes a model staleness detection feature that checks model accuracy hourly using a background scheduler (`BackgroundScheduler`). If accuracy or F1-score drops below the predefined threshold (e.g., 80%), an automatic retraining process is triggered.

Notifications are sent via Gmail to alert the team of any retraining events. After retraining, the new model is deployed seamlessly, ensuring continuous and optimal model performance without downtime.

### Notes

- **Model Versioning:** MLflow is used to track, version, and log model artifacts, providing easy rollback in case of degraded model performance.
- **Cost Optimization:** Batch processing and model caching reduce computational overhead and latency for both live and batch predictions.

This setup provides a flexible, scalable, and efficient solution for continuous model training and deployment, suitable for local or cloud deployment environments.

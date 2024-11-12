import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import mlflow
import mlflow.pyfunc
import mlflow.transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd
from datasets import Dataset




# function to compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {"accuracy": accuracy, "f1": f1}


def load_data():
    train_data = pd.read_csv('./train_sentiment_data.csv')
    eval_data = pd.read_csv('./val_sentiment_data.csv')
    return train_data, eval_data

#  data preprocessing
def preprocess_data(data, tokenizer):
    encodings = tokenizer(data['text'].tolist(), padding=True, truncation=True, return_tensors="pt")
    encodings['labels'] = data['label'].tolist()
    return encodings

# Convert DataFrame to Dataset
def dataframe_to_dataset(dataframe, tokenizer):
    encodings = preprocess_data(dataframe, tokenizer)

    dataset_dict = {key: encodings[key].tolist() if hasattr(encodings[key], 'tolist') else encodings[key] for key in encodings}

    dataset = Dataset.from_dict(dataset_dict)
    return dataset

#  model training
def train_model(train_data, eval_data, model_name="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_encodings = dataframe_to_dataset(train_data, tokenizer)
    eval_encodings = dataframe_to_dataset(eval_data, tokenizer)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        logging_dir='./logs'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_encodings,
        eval_dataset=eval_encodings,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return model, tokenizer

# Log model with MLflow
def log_model(model, tokenizer):
    with mlflow.start_run() as run:
        model_output_dir = "./model"
        model.save_pretrained(model_output_dir)
        tokenizer.save_pretrained(model_output_dir)

        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
        mlflow.transformers.log_model(
            transformers_model=pipe,
            artifact_path="text_classifier",
            input_example="MLflow is great!"
        )

        # Capture and print the run ID
        run_id = run.info.run_id
        print(f"Model logged in run ID: {run_id}")
        return run_id

# Evaluate model performance
def evaluate_model(model, eval_data):
    raw_predictions = model.predict(eval_data['text'].tolist())
    predictions = [1 if raw_predictions['label'][i] == 'LABEL_1' else 0 for i in range(len(eval_data))] #(0 for Negative, 1 for Positive sentiment)
    true_labels = eval_data['label'].tolist()
    # accuracy and F1 score
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')

    metrics = {"accuracy": accuracy, "f1": f1}
    print(f"Evaluation metrics: {metrics}")
    return metrics

# Check if retraining is needed
def check_retraining_needed(metrics, threshold=0.8):
    return metrics['accuracy'] < threshold

# Send Gmail alert
def send_gmail_alert(subject, body, to_email):
    from_email = "your_email@gmail.com"
    password = "your_password"

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, password)
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()
        print("Email sent successfully")
    except Exception as e:
        print(f"Failed to send email: {e}")

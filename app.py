from apscheduler.schedulers.background import BackgroundScheduler
import uvicorn
from threading import Thread,Lock
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import mlflow
import mlflow.pyfunc
import mlflow.transformers
from train import load_data, train_model, evaluate_model, check_retraining_needed, log_model,send_gmail_alert


app = FastAPI()

current_model = None
model_lock = Lock()

class LivePredictionRequest(BaseModel):
    text: str

class BatchPredictionRequest(BaseModel):
    texts: List[str]

label_mapping = {"LABEL_1": "Positive", "LABEL_0": "Negative"}

# load the model from model_uri
def load_model_from_uri(model_uri):
    if model_uri:
        return mlflow.pyfunc.load_model(model_uri)
    else:
        raise ValueError("Model URI is not set")

# Live prediction endpoint
@app.post("/predict/live")
def predict_live(request: LivePredictionRequest):
  global current_model
  try:
      with model_lock:
        prediction = current_model.predict([request.text])
      prediction['label'] = [label_mapping[label] for label in prediction['label']]
      return {"prediction": prediction}
  except Exception as e:
      raise HTTPException(status_code=500, detail=str(e))

# Batch prediction endpoint
@app.post("/predict/batch")
def predict_batch(request: BatchPredictionRequest):
  global current_model
  try:
      # lock to ensure thread safety while accessing the model
      with model_lock:
        predictions = current_model.predict(request.texts)
      predictions['label'] = [label_mapping[label] for label in predictions['label']]
      return {"predictions": predictions}
  except Exception as e:
      raise HTTPException(status_code=500, detail=str(e))


def periodic_evaluation(model_uri):
  global current_model
  print("Running periodic evaluation...")
  train_data, eval_data = load_data()
  loaded_model = load_model_from_uri(model_uri)
  metrics = evaluate_model(loaded_model, eval_data)
  # Check if retraining is needed
  if check_retraining_needed(metrics):
      print("Retraining needed. Performance below threshold.")
      send_gmail_alert(
          subject="Model Performance Alert",
          body="Model performance has degraded. Retraining is required.",
          to_email="recipient_email@gmail.com"
      )
      # model Retraining
      model, tokenizer = train_model(train_data, eval_data)
      new_run_id = log_model(model, tokenizer)
      new_model_uri = f"runs:/{new_run_id}/text_classifier"
      # Reload the new model in a thread-safe manner
      with model_lock:
        current_model = load_model_from_uri(new_model_uri)

      print(f"Model retrained. New model loaded from URI: {new_model_uri}")
  else:
    print("Model performance is satisfactory.")


# Main function
def main():
  global current_model
  train_data, eval_data = load_data()
  model, tokenizer = train_model(train_data, eval_data)
  initial_run_id = log_model(model, tokenizer)
  model_uri = f"runs:/{initial_run_id}/text_classifier"

  # Load the initial model
  current_model = load_model_from_uri(model_uri)
  if current_model is None:
    print("Error: Failed to load the initial model.")
  else:
    print(f"Initial model loaded from URI: {model_uri}")

  #start the scheduler
  scheduler = BackgroundScheduler()
  scheduler.add_job(lambda: periodic_evaluation(model_uri), 'interval', hours=1)  #interval
  scheduler.start()

  try:
      while True:
          time.sleep(1)
  except (KeyboardInterrupt, SystemExit):
      scheduler.shutdown()


def run():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":

    thread = Thread(target=run)
    thread.start()
    main()
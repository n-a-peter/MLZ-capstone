import pickle
from typing import Dict, Any
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="diabetes-diagnosis")

# load the model with pickle
with open("model.bin", "rb") as f_in:
    pipeline = pickle.load(f_in)

@app.post("/predict")
def predict_status(patient: Dict[str, Any]):
    diagnosis = pipeline.predict_proba(patient)[:, 1]
    diagnosis = float(diagnosis[0])
    return {
        "probability of diagnosis": diagnosis,
        "diagnosed" : bool(diagnosis >= 0.5)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
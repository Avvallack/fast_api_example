from typing import Optional
from fastapi import FastAPI

from src.model_inference import inference_step
from src.dataclass import JsonDict, ReturnDict

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/prediction")
def get_prediction(item: JsonDict) -> ReturnDict:
    return inference_step(item)


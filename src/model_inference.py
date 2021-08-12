import pickle
import numpy as np

from src.dataclass import JsonDict, ReturnDict


MODEL = pickle.load(open('./data/model.pkl', 'rb'))
SCALER = pickle.load(open('./data/scaler.pkl', 'rb'))


def get_prediction(X, model=MODEL, scaler=SCALER):
    scaled_x = scaler.transform(X)
    prediction = model.predict(scaled_x)
    return ReturnDict(**{'prediction': prediction})


def parse_json(item: JsonDict):
    return np.array(list(item.dict().values())).reshape(1, -1)


def inference_step(json_string):
    X = parse_json(json_string)
    return get_prediction(X)

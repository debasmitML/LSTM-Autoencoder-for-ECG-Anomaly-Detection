import os

class Config:
    INPUT_PATH = os.environ.get("INPUT_PATH", "http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv")
    MODEL_WEIGHT_PATH = os.environ.get('MODEL_WEIGHT_PATH', './weight')
    DIRECTORY_SAVE_RESULT = os.environ.get("DIRECTORY_SAVE_RESULT", "./result")
    
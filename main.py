from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel


class Item(BaseModel):
    text: str



app = FastAPI()
classifier = pipeline("sentiment-analysis")



@app.get("/")
def root():
    return {"message": "Hello World"}


@app.post("/predict/")
def predict(item: Item):
    return classifier(item.text)[0]


@app.get("/info/")
def info():
    return {
        "model_name": classifier.model.config.name_or_path,
        "framework": "transformers",
        "version": transformers_version
    }

from fastapi import FastAPI, HTTPException
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
    if not item.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    return classifier(item.text)[0]

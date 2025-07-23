from fastapi import FastAPI
from pydantic import BaseModel
from .model import SentimentModel

app = FastAPI(title="Phrase Sentiment Analyzer")

model = SentimentModel()

class TextInput(BaseModel):
    text: str

@app.post("/predict")
async def predict_sentiment(input: TextInput):
    prediction = model.predict(input.text)
    return {"text": input.text, "sentiment": prediction}

@app.get("/")
async def root():
    return {"message": "Phrase Sentiment Analyzer API"}

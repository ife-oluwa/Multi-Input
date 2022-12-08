from fastapi import FastAPI, UploadFile
import numpy as np
from utils import tokenizer, get_sequences, model, index_to_class
from pydantic import BaseModel


class description(BaseModel):
    description: str


app = FastAPI(
    title="Product Catgerization API",
    description="A simple API that uses NLP, CV and MultiInput to classify products.",
    version="0.1"
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post('/predict/')
async def predictImageText(description: str = None, Image: UploadFile | None = None):
    padded_sentence = get_sequences(tokenizer, description)
    pred = model.predict(np.expand_dims(padded_sentence[0], axis=0))[0]
    pred_class = index_to_class[np.argmax(pred).astype('uint8')]
    return {
        "Image": Image,
        "Description": pred_class
    }

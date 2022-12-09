from fastapi import FastAPI, UploadFile
import numpy as np
from utils import tokenizer, get_sequences, model_txt, model_multi, index_to_class, image_preprocess, model_cv
from pydantic import BaseModel
from typing import Union
import shutil
import os
import cv2
import uvicorn


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
async def predictImageText(description: str = None, file: UploadFile | None = None):
    if not description:
        with open('image.jpg', "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        processed_image = image_preprocess('image.jpg', [224, 224])
        pred = model_cv.predict(np.expand_dims(processed_image[0], axis=0))[0]
        pred_class = index_to_class[np.argmax(pred).astype('uint8')]
        os.remove('image.jpg')
        return{
            "Image": file.filename,
            "Predicted Category": pred_class
        }
    elif not file:
        padded_sentence = get_sequences(tokenizer, description)
        pred = model_txt.predict(np.expand_dims(padded_sentence[0], axis=0))[0]
        pred_class = index_to_class[np.argmax(pred).astype('uint8')]
        return {
            "Description": description,
            "Predicted Category": pred_class
        }
    elif description and file:
        with open('image.jpg', "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        padded_sentence = get_sequences(tokenizer, description)
        processed_image = image_preprocess('image.jpg', [150, 150])
        os.remove('image.jpg')
        pred = model_multi.predict(
            [np.expand_dims(padded_sentence[0], axis=0), np.array([processed_image])])[0]
        pred_class = index_to_class[np.argmax(pred).astype('uint8')]
        return {
            "Image": file.filename,
            "Description": description,
            "Predicted Category": pred_class
        }

# if __name__ == '__main__':
#     uvicorn.run(app, host="0.0.0.0", port=80, debug)

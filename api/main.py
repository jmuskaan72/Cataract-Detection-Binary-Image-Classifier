from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import io

# Load the trained model
model = tf.keras.models.load_model("models/best_model_v2_vgg16.h5")
app = FastAPI()

# Image preprocessing function (adjust as per your model input)
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224)) 
    image_array = np.array(image) / 255.0  #normalization
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = preprocess_image(contents)
        prediction = model.predict(image)[0][0]
        #optimal threshold set at 0.6
        result = "Cataract" if prediction < 0.6 else "Normal"
        confidence = round(float(1 - prediction if result == "Cataract" else prediction)*100, 2)
        return JSONResponse(content={"prediction": result, "confidence": confidence})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

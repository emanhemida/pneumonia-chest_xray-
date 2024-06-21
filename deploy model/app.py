from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import logging
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize FastAPI app
app = FastAPI()

# Load your trained model
model = load_model('model.h5')
class_names = ['Normal', 'bacteria', 'virus']

def preprocess_image(img, target_size):
    """Resize and preprocess the image for the model."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    try:
        # Read the file's content into a BytesIO object
        img_bytes = io.BytesIO(await file.read())
        
        # Use PIL to open the image
        img = Image.open(img_bytes)
        img_array = preprocess_image(img, (224, 224))
        
        # Make prediction
        predictions = model.predict(img_array)

        predicted_class = np.argmax(predictions, axis=1)
        
        # Return the prediction
        predictions = {
            'class': class_names[predicted_class[0]],
            'confidence': float(predictions[0][predicted_class[0]])
        }
        return JSONResponse(content=predictions)
    except Exception as e:
        logging.debug(f"Error processing the file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing the file: {str(e)}")
        
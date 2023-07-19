from fastapi import FastAPI, UploadFile, File
import uvicorn
import tensorflow as tf
from PIL import Image
import numpy as np
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [

 "http://localhost",
 "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the saved model
model = tf.keras.models.load_model('DenseNet_Model.h5')

# Define class names
classNames = ['Potato_Early_blight', 'Potato_Healthy', 'Potato_Late_blight']


# Define a route for prediction
@app.post('/predict')
async def predict_image(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    with open(file.filename, 'wb') as buffer:
        buffer.write(await file.read())

    # Open the image using PIL
    image = Image.open(file.filename)

    # Preprocess the image
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Make predictions
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    return {
        'class': classNames[predicted_class],
        'confidence': float(confidence)
    }


# Run the server
if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)

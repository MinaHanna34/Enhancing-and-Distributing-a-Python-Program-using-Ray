from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from io import BytesIO
import numpy as np
import ray
import time  # Import the time module

app = Flask(__name__)

ray.init(address='auto')  # Connect to the Ray cluster

model = load_model('pneumonia_model.h5')  # Load the model

@ray.remote
def predict_image(img_bytes):
    img = image.load_img(BytesIO(img_bytes), target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return 'Pneumonia' if prediction[0][0] > 0.5 else 'Normal'

 
@app.route('/predict', methods=['POST'])
def predict():
    img_file = request.files['image']
    if not img_file:
        return jsonify({'error': 'No image provided. Please provide an image file.'}), 400
    if img_file.filename == '':
        return jsonify({'error': 'No image selected for uploading'}), 400

    img_bytes = img_file.read()

    start_time = time.time()  # Start time measurement
    result_id = predict_image.remote(img_bytes)
    result = ray.get(result_id)
    end_time = time.time()  # End time measurement

    duration = end_time - start_time  # Calculate the duration
    print(f"Prediction: {result}, Prediction Time: {duration:.2f} seconds")  # Debug output

    # Return the prediction result and the duration of the prediction
    return jsonify({
        'prediction': result,
        'prediction_time': f"{duration:.2f} seconds"
    })

if __name__ == '__main__':
    app.run(port=5001, debug=True)

from flask import Flask, request, render_template, send_from_directory
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Load the trained models
spiral_model = tf.keras.models.load_model('spiral_detection_model_vgg16.h5')
wave_model = tf.keras.models.load_model('wave_detection_model_vgg16.h5')

# Ensure the 'static' directory exists for saving uploads
if not os.path.exists('static'):
    os.makedirs('static')


def predict_image(model, image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return prediction[0][0]


def is_valid_image(file_path):
    try:
        img = load_img(file_path)
        return True
    except Exception as e:
        print(f"Image validation error: {e}")
        return False


def classify_image(file_path, file_type):
    # Predict with the relevant model
    if file_type == 'spiral':
        score = predict_image(spiral_model, file_path)
    elif file_type == 'wave':
        score = predict_image(wave_model, file_path)
    else:
        return "Invalid image type. Please upload either a spiral or wave image.", None

    # Determine prediction based on score
    if score > 0.5:
        result = 'Parkinson Detected'
    else:
        result = 'Healthy'

    return result, None


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    file_type = request.form['file_type']
    file_path = os.path.join('static', file.filename)
    file.save(file_path)

    if not is_valid_image(file_path):
        return "Uploaded file is not a valid image.", 400

    # Check the file type and predict using the appropriate model
    if file_type not in ['spiral', 'wave']:
        return "Invalid image type. Please upload either a spiral or wave image.", 400

    prediction, error = classify_image(file_path, file_type)
    if error:
        return error, 400

    return f'''
        <h2>Prediction Result</h2>
        <p>{prediction}</p>
        <img src="/static/{file.filename}" alt="Uploaded Image">
        <br>
        <a href="/">Go Back</a>
    '''


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, jsonify, request, render_template

from pyngrok import ngrok
from werkzeug.utils import secure_filename
import os
from keras.models import load_model
import numpy as np
import tensorflow as tf

from flask_ngrok import run_with_ngrok
from flask import Flask,request,render_template
import pickle
import numpy as np
# Load the trained model

model = load_model('WebApp/pneumonia.h5')

app = Flask(__name__,template_folder='WebApp/templates')
run_with_ngrok(app)
ngrok.set_auth_token('2M1S9O9HgAtEu392TPckxBrbxBA_3ocqMB5F1oFTHzN7acm4M')
# Set the path where uploaded files will be saved

UPLOAD_FOLDER = 'WebApp/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded file
    file = request.files['file']

    # Save the file to disk
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Load the image and preprocess it
    img = tf.keras.preprocessing.image.load_img(file_path, target_size=(128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Make a prediction
    prediction = model.predict(img_array)

    # Get the predicted label
    predicted_label = np.argmax(prediction)

    # Return the prediction as a JSON object
    return str(predicted_label)


# Define a route to render the HTML form
@app.route('/')
def home():
    return render_template('index.html')

app.run()

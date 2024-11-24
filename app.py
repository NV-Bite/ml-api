import os
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, jsonify, request
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess_input
from werkzeug.utils import secure_filename
import time

app = Flask(__name__)

# Load model
MODEL_PATH = "model/model_xception.keras"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

app.config["ALLOWED_EXTENSIONS"] = {"jpg", "jpeg", "png"}
app.config["UPLOAD_FOLDER"] = "static/uploads"


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


# Class labels
class_labels = ['Ayam Goreng', 'Burger', 'French Fries', 'Gado-Gado', 'Ikan Goreng',
                'Mie Goreng', 'Nasi Goreng', 'Nasi Padang', 'Pizza', 'Rawon',
                'Rendang', 'Sate', 'Soto Ayam']


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": {
            "code": 200,
            "message": "Success fetching the API",
        },
        "data": None
    }), 200


@app.route("/predict_image", methods=["POST", "GET"])
def predict_image():
    if request.method == "POST":
        image = request.files["image"]
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            start_predict_time = time.time()
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            img = load_img(image_path, target_size=(300, 300))
            img_array = np.asarray(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = xception_preprocess_input(
                img_array)
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions)
            predicted_class_label = class_labels[predicted_class_index]
            confidence = predictions[0][predicted_class_index] * \
                100
            confidence = round(confidence, 1)
            end_predict_time = time.time()
            predict_time = end_predict_time - start_predict_time
            return jsonify(
                {
                    "status": {
                        "code": 200,
                        "message": "Success",
                    },
                    "data": {
                        "predicted_class": predicted_class_label,
                        "confidence": float(confidence),
                        "predict_time": predict_time
                    }
                }), 200
        else:
            return jsonify({
                "status": {
                    "code": 400,
                    "message": "Bad Request",
                },
                "data": None
            }), 400
    else:
        return jsonify({
            "status": {
                "code": 405,
                "message": "Method Not Allowed",
            },
            "data": None
        }), 405


if __name__ == "__main__":
    app.run()

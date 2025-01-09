import os
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, jsonify, request
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess_input
import time
from google.cloud import storage
from google.cloud import secretmanager
import uuid
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting
import markdown
import json
from dotenv import load_dotenv
import threading
import re
import google.generativeai as genai


app = Flask(__name__)

app.config["ALLOWED_EXTENSIONS"] = {"jpg", "jpeg", "png"}
app.config["UPLOAD_FOLDER"] = "static/uploads"

# Create upload directory if it doesn't exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


model = None

load_dotenv()


def get_secret(secret_name, project_id=None):
    client = secretmanager.SecretManagerServiceClient()
    project_id = os.getenv('PROJECT_ID')
    if not project_id:
        raise ValueError("PROJECT_ID environment variable is not set.")
    secret_version = f'projects/{project_id}/secrets/{secret_name}/versions/latest'
    response = client.access_secret_version(name=secret_version)
    return response.payload.data.decode('UTF-8')


# Get credentials from Secret Manager
credentials_json = get_secret("GOOGLE_APPLICATION_CREDENTIALS")

# Parse file JSON
credentials = json.loads(credentials_json)

# Configure Google Cloud Storage
BUCKET_NAME = credentials['bucket_name']
MODEL_PATH = credentials['model_path']

# Configure Vertex AI
PROJECT_ID = credentials['project_id']
print(f"Project ID: {PROJECT_ID}")
LOCATION = credentials['location']
print(f"Location: {LOCATION}")
MODEL_ID = credentials['model_id']
print(f"Model ID: {MODEL_ID}")
SYSTEM_INSTRUCTION = credentials['system_instruction']
print(f"System Instruction: {SYSTEM_INSTRUCTION}")
KEY_API = credentials['key_api']
print(f"Key API: {KEY_API}")
CHATBOT = credentials['chatbot']
print(f"Chatbot: {CHATBOT}")


# Download model from GCS
def download_model_from_gcs():
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_PATH)
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model_xception.keras")
    blob.download_to_filename(model_path)
    print(f"Model downloaded from GCS and saved to {model_path}")
    return model_path


# Load the model
def load_model():
    global model
    if model is None:
        model_path = download_model_from_gcs()
        model = tf.keras.models.load_model(model_path, compile=False)


# Load the model once during initialization
load_model()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


# Class labels
class_labels = ['Ayam Goreng', 'Burger', 'French Fries', 'Gado-Gado', 'Ikan Goreng',
                'Mie Goreng', 'Nasi Goreng', 'Nasi Padang', 'Pizza', 'Rawon',
                'Rendang', 'Sate', 'Soto Ayam']


generation_config = {
    "max_output_tokens": 8000,
    "temperature": 1,
    "top_p": 0.95,
}

safety_settings = [
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
]


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": {
            "code": 200,
            "message": "Welcome to model api NV Bite🍃",
        },
        "data": None
    }), 200


@app.route("/predict_image", methods=["POST"])
def predict_image_with_gentext():
    if "image" not in request.files:
        return jsonify({
            "status": {
                "code": 400,
                "message": "No image provided",
            },
            "data": None
        }), 400

    image = request.files["image"]
    if image and allowed_file(image.filename):
        unique_filename = str(uuid.uuid4()) + secure_filename(image.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
        image.save(file_path)

        img_array = preprocess_image(file_path)

        start_predict_time = time.time()

        prediction = model.predict(img_array)
        predicted_class, confidence = decode_prediction(prediction)

        # Menghasilkan teks menggunakan Vertex AI
        generated_text = generate_text(predicted_class)

        end_predict_time = time.time()
        predict_time = end_predict_time - start_predict_time

        upload_to_gcs(file_path, BUCKET_NAME,
                      f"upload_image/{unique_filename}")

        os.remove(file_path)

        return jsonify({
            "status": {
                "code": 200,
                "message": "Success",
            },
            "data": {
                "predicted_class": predicted_class,
                "confidence": float(confidence),
                "GenText": generated_text,
                "predict_time": predict_time
            }
        }), 200
    else:
        return jsonify({
            "status": {
                "code": 400,
                "message": "Invalid file type",
            },
            "data": None
        }), 400


@app.route("/predict_streamlit", methods=["POST"])
def predict_image_no_gentext():
    if "image" not in request.files:
        return jsonify({
            "status": {
                "code": 400,
                "message": "No image provided",
            },
            "data": None
        }), 400

    image = request.files["image"]
    if image and allowed_file(image.filename):
        # Generate a unique filename
        unique_filename = str(uuid.uuid4()) + secure_filename(image.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
        image.save(file_path)  # Save the uploaded image

        # Preprocess the image
        img_array = preprocess_image(file_path)

        # Start prediction timing
        start_predict_time = time.time()

        # Predict the image
        prediction = model.predict(img_array)
        predicted_class, confidence = decode_prediction(prediction)

        # End prediction timing
        end_predict_time = time.time()
        predict_time = end_predict_time - start_predict_time

        # Return prediction results to the client
        response = jsonify({
            "status": {
                "code": 200,
                "message": "Success",
            },
            "data": {
                "predicted_class": predicted_class,
                "confidence": float(confidence),
                "predict_time": predict_time
            }
        })

        # Start GCS upload asynchronously
        def upload_to_gcs_async():
            upload_to_gcs(file_path, BUCKET_NAME,
                          f"upload_image/{unique_filename}")
            os.remove(file_path)  # Remove the image after upload

        threading.Thread(target=upload_to_gcs_async).start()

        return response
    else:
        return jsonify({
            "status": {
                "code": 400,
                "message": "Invalid file type",
            },
            "data": None
        }), 400


genai.configure(api_key=KEY_API)

# Create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
    system_instruction=CHATBOT,
)

chat_session = model.start_chat(
    history=[
        {
            "role": "user",
            "parts": [
                "berikan summary nya ",
            ],
        },
        {
            "role": "model",
            "parts": [
                "Tentu, berikut adalah ringkasan singkat tentang Rizky Yanuar Kristianto:\n\nRizky adalah seorang mahasiswa S1 Data Science berusia 21 tahun di Surabaya, dengan IPK 3,47. Ia memiliki pengalaman sebagai ketua penelitian dan anggota staf divisi advokasi mahasiswa. Rizky memiliki keahlian teknis dalam Python, Tableau, Microsoft Office, dan TensorFlow, serta keahlian tambahan di Data Mining, Deep Learning, Data Visualization, dan Manajemen Proyek. Ia juga memiliki beberapa proyek penting, termasuk prediksi depresi dari data ucapan, analisis sentimen penerima KIPK berdasarkan tweet, dan pengembangan aplikasi pelacak jejak karbon makanan NV-Bite. Rizky memiliki kemampuan analisis yang kuat dan antusiasme dalam bidang data science.\n",
            ],
        },
        {
            "role": "user",
            "parts": [
                "berikan project projectnya ",
            ],
        },
        {
            "role": "model",
            "parts": [
                "Rizky Yanuar Kristianto memiliki beberapa proyek penting, yaitu:\n1.  **Prediksi Depresi dari Data Ucapan**: Mengembangkan model prediktif untuk mengidentifikasi indikator depresi dari data ucapan, yang dipresentasikan pada simposium penelitian universitas.\n2.  **Analisis Sentimen Penerima KIPK Berdasarkan Tweet di Platform X**: Menganalisis sentimen tweet terkait penerima KIPK menggunakan NLP dan memvisualisasikan hasilnya.\n3.  **NV-Bite: Aplikasi Pelacak Jejak Karbon Makanan**: Membangun model pembelajaran mesin untuk pengenalan makanan dan estimasi jejak karbon, berkolaborasi dengan tim cloud dan mobile untuk integrasi aplikasi.\n",
            ],
        },
        {
            "role": "user",
            "parts": [
                "summary cvnya",
            ],
        },
        {
            "role": "model",
            "parts": [
                "Rizky Yanuar Kristianto adalah mahasiswa S1 Data Science di Surabaya dengan IPK 3,47, memiliki pengalaman sebagai ketua penelitian dan aktif dalam organisasi mahasiswa. Ia memiliki keahlian teknis seperti Python, Tableau, dan TensorFlow, serta keahlian tambahan dalam Data Mining dan Deep Learning. Rizky juga memiliki pengalaman proyek dalam analisis data dan pengembangan aplikasi, menunjukkan kemampuan analisis dan pemecahan masalah yang kuat.\n",
            ],
        },
        {
            "role": "user",
            "parts": [
                "berikan dalam perpoin",
            ],
        },
        {
            "role": "model",
            "parts": [
                "Baik, berikut adalah poin-poin ringkasan CV Rizky Yanuar Kristianto:\n\n*   **Identitas:**\n    *   Nama: Rizky Yanuar Kristianto\n    *   Usia: 21 tahun\n    *   Domisili: Surabaya\n    *   Pendidikan: Mahasiswa S1 Data Science, Universitas Negeri Surabaya (IPK: 3,47/4,0)\n*   **Pengalaman:**\n    *   Ketua Penelitian, Fakultas Matematika dan Ilmu Pengetahuan Alam, Universitas Negeri Surabaya\n    *   Anggota Staf, Divisi Advokasi dan Kesejahteraan Mahasiswa, Himpunan Mahasiswa Data Science, Universitas Negeri Surabaya\n*   **Proyek:**\n    *   Prediksi Depresi dari Data Ucapan\n    *   Analisis Sentimen Penerima KIPK Berdasarkan Tweet di Platform X\n    *   NV-Bite: Aplikasi Pelacak Jejak Karbon Makanan\n*   **Keahlian:**\n    *   Teknis: Python, Tableau, Microsoft Office, TensorFlow\n    *   Tambahan: Data Mining, Deep Learning, Data Visualization, Manajemen Proyek\n*   **Deskripsi Singkat:**\n    *   Mahasiswa S1 Data Science dengan keterampilan analisis yang kuat, kemampuan pemecahan masalah berbasis data, dan pemikiran kritis.\n    *   Berfokus pada penerapan pengetahuan akademik untuk menyelesaikan tantangan berbasis data.\n    *   Memiliki antusiasme tinggi dalam berkontribusi pada proyek-proyek berdampak, sambil terus mengembangkan keahlian di bidang data science.\n",
            ],
        },
    ]
)


@app.route("/chatbot", methods=["POST"])
def predict_website():
    data = request.get_json()
    message = data.get('message')
    print(f"Message: {message}")
    if not message:
        return jsonify({
            "status": {
                "code": 400,
                "message": "No message provided",
            },
            "data": None
        }), 400

    # Generate text using Vertex AI
    answer = chatbot(message)
    print(f"Generated Text: {answer}")
    message = {"answer": answer}
    return jsonify(message)


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(300, 300))
    img_array = img_to_array(img)  # Convert to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimension
    img_array = xception_preprocess_input(
        img_array)  # Preprocessing sesuai Xception
    return img_array


def decode_prediction(prediction):
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_class_index]  # Get name Class
    confidence = prediction[0][predicted_class_index] * \
        100  # Get confidence
    confidence = round(confidence, 1)
    return predicted_class, confidence


def upload_to_gcs(file_path, bucket_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file_path)
    print(f"File {file_path} uploaded to {destination_blob_name}.")


def generate_text(predicted_class):
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel(
        "gemini-1.5-flash",
        system_instruction=SYSTEM_INSTRUCTION,
    )
    responses = model.generate_content(
        [predicted_class],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )
    generated_text = ""
    for response in responses:
        generated_text += response.text
    # Convert markdown to HTML
    html_content = markdown.markdown(
        generated_text, extensions=["tables"])
    generated_text = html_content.replace("\n", "")
    return generated_text


def chatbot(input_text):
    response = chat_session.send_message(input_text)
    generated_text = response.text
    generated_text = re.sub(
        r'^(#{1})\s*(.*?)$', r'<h1>\2</h1>', generated_text, flags=re.MULTILINE)
    generated_text = re.sub(
        r'^(#{2})\s*(.*?)$', r'<h2>\2</h2>', generated_text, flags=re.MULTILINE)
    generated_text = re.sub(
        r'^(#{3})\s*(.*?)$', r'<h3>\2</h3>', generated_text, flags=re.MULTILINE)
    generated_text = re.sub(
        r'\*\*(.*?)\*\*', r'<strong>\1</strong>', generated_text)
    generated_text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', generated_text)
    generated_text = re.sub(
        r'^[\*\-]\s+(.*)$', r'<ul><li>\1</li></ul>', generated_text, flags=re.MULTILINE)
    generated_text = re.sub(
        r'^\s{2,}[\*\-]\s+(.*)$', r'<ul><li>\1</li></ul>', generated_text, flags=re.MULTILINE)
    generated_text = re.sub(r'\[(.*?)\]\((.*?)\)',
                            r'<a href="\2">\1</a>', generated_text)
    generated_text = re.sub(
        r'^>\s*(.*)$', r'<blockquote>\1</blockquote>', generated_text, flags=re.MULTILINE)
    generated_text = re.sub(r'(^|\n)([^\n]+)', r'<p>\2</p>', generated_text)
    return generated_text


if __name__ == "__main__":
    # Create upload folder if not exists
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

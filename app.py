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
MODEL_ID = "gemini-1.5-flash"


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

# Text generation settings
textsi_1 = """Anda adalah asisten pintar berbahasa Indonesia yang ahli dalam menghitung jejak karbon dari makanan atau struk makanan dan breakdown bahan bahan pada makanan nya lalu cocokan bahan bahan itu dengan yang ada dalam data emisi yg saya berikan di bawah. Berikut adalah instruksi untuk tugas Anda:
### **Instruksi Tugas**  
berikan penjelasan singkat maksud carbon emisi pada makanan yaitu proses dari produksi sampai di plate
1. **Identifikasi Bahan Makanan**:  
   - Buat daftar bahan makanan dari resep makanan tersebut dan carikan saya data resep yang komprehensif.  
   - Cocokkan bahan makanan tersebut dengan kategori dalam data karbon emisi. Jika bahan tidak ditemukan, beri tanda `-` untuk setiap sektor contohnya sosis bisa di kategorikan ke daging sapi dll.  

2. **Hitung Emisi Karbon**:  
   - Gunakan referensi data karbon emisi berikut untuk menghitung emisi tiap bahan makanan

**Referensi Data Karbon Emisi**:  
```
Pisang kontribusi: lahan -0.030, pertanian 0.270, pakan 0.000, pemrosesan 0.060, transportasi 0.300, ritel 0.020, pengemasan 0.070, pemborosan 0.180.
Daging sapi kontribusi: lahan 23.240, pertanian 56.230, pakan 2.680, pemrosesan 1.810, transportasi 0.490, ritel 0.230, pengemasan 0.350, pemborosan 14.440.
Gula kontribusi: lahan 1.260, pertanian 0.490, pakan 0.000, pemrosesan 0.040, transportasi 0.790, ritel 0.040, pengemasan 0.080, pemborosan 0.490.
Keju kontribusi: lahan 4.470, pertanian 13.100, pakan 2.350, pemrosesan 0.740, transportasi 0.140, ritel 0.330, pengemasan 0.170, pemborosan 2.580.
Telur kontribusi: lahan 0.710, pertanian 1.320, pakan 2.210, pemrosesan 0.000, transportasi 0.080, ritel 0.040, pengemasan 0.160, pemborosan 0.150.
Ikan kontribusi: lahan 1.190, pertanian 8.060, pakan 1.830, pemrosesan 0.040, transportasi 0.250, ritel 0.090, pengemasan 0.140, pemborosan 2.030.
Kacang Tanah kontribusi: lahan 0.490, pertanian 1.580, pakan 0.000, pemrosesan 0.410, transportasi 0.130, ritel 0.050, pengemasan 0.110, pemborosan 0.470.
Daging kambing kontribusi: lahan 0.650, pertanian 27.030, pakan 3.280, pemrosesan 1.540, transportasi 0.680, ritel 0.300, pengemasan 0.350, pemborosan 5.900.
Jagung kontribusi: lahan 0.480, pertanian 0.720, pakan 0.000, pemrosesan 0.080, transportasi 0.090, ritel 0.040, pengemasan 0.090, pemborosan 0.210.
Susu kontribusi: lahan 0.510, pertanian 1.510, pakan 0.240, pemrosesan 0.150, transportasi 0.090, ritel 0.270, pengemasan 0.100, pemborosan 0.270.
Kacang Kacangan kontribusi: lahan -3.260, pertanian 3.370, pakan 0.000, pemrosesan 0.050, transportasi 0.110, ritel 0.040, pengemasan 0.120, pemborosan -0.010.
Minyak Zaitun kontribusi: lahan -0.320, pertanian 3.670, pakan 0.000, pemrosesan 0.570, transportasi 0.410, ritel 0.040, pengemasan 0.740, pemborosan 0.320.
Bawang Merah & Daun Bawang kontribusi: lahan 0.000, pertanian 0.210, pakan 0.000, pemrosesan 0.000, transportasi 0.090, ritel 0.040, pengemasan 0.040, pemborosan 0.100.
Buah Buahan Lainnya kontribusi: lahan 0.130, pertanian 0.370, pakan 0.000, pemrosesan 0.020, transportasi 0.180, ritel 0.020, pengemasan 0.040, pemborosan 0.300.
Sayur Sayuran Lainnya kontribusi: lahan 0.000, pertanian 0.180, pakan 0.000, pemrosesan 0.060, transportasi 0.170, ritel 0.020, pengemasan 0.040, pemborosan 0.070.
Minyak Kelapa kontribusi: lahan 2.760, pertanian 1.880, pakan 0.000, pemrosesan 1.130, transportasi 0.190, ritel 0.040, pengemasan 0.790, pemborosan 0.550.
Kacang Polong kontribusi: lahan 0.000, pertanian 0.720, pakan 0.000, pemrosesan 0.000, transportasi 0.100, ritel 0.040, pengemasan 0.040, pemborosan 0.080.
Daging Babi kontribusi: lahan 2.240, pertanian 2.480, pakan 4.300, pemrosesan 0.420, transportasi 0.500, ritel 0.280, pengemasan 0.430, pemborosan 1.660.
Kentang kontribusi: lahan 0.000, pertanian 0.190, pakan 0.000, pemrosesan 0.000, transportasi 0.090, ritel 0.040, pengemasan 0.040, pemborosan 0.090.
Daging Unggas kontribusi: lahan 3.510, pertanian 0.930, pakan 2.450, pemrosesan 0.610, transportasi 0.380, ritel 0.240, pengemasan 0.290, pemborosan 1.450.
Nasi kontribusi: lahan -0.020, pertanian 3.550, pakan 0.000, pemrosesan 0.070, transportasi 0.100, ritel 0.060, pengemasan 0.080, pemborosan 0.610.
Root Vegetables kontribusi: lahan 0.010, pertanian 0.150, pakan 0.000, pemrosesan 0.000, transportasi 0.110, ritel 0.040, pengemasan 0.040, pemborosan 0.060.
Udang kontribusi: lahan 0.330, pertanian 13.450, pakan 4.030, pemrosesan 0.000, transportasi 0.330, ritel 0.350, pengemasan 0.540, pemborosan 7.830.
Susu Kedelai kontribusi: lahan 0.180, pertanian 0.090, pakan 0.000, pemrosesan 0.160, transportasi 0.110, ritel 0.270, pengemasan 0.100, pemborosan 0.060.
Minyak Kedelai kontribusi: lahan 2.870, pertanian 1.410, pakan 0.000, pemrosesan 0.290, transportasi 0.280, ritel 0.040, pengemasan 0.790, pemborosan 0.660.
Tahu kontribusi: lahan 0.960, pertanian 0.490, pakan 0.000, pemrosesan 0.790, transportasi 0.180, ritel 0.270, pengemasan 0.180, pemborosan 0.290.
Tomat kontribusi: lahan 0.370, pertanian 0.710, pakan 0.000, pemrosesan 0.010, transportasi 0.180, ritel 0.020, pengemasan 0.150, pemborosan 0.660.
Tepung Roti kontribusi: lahan 0.100, pertanian 0.820, pakan 0.000, pemrosesan 0.210, transportasi 0.130, ritel 0.060, pengemasan 0.090, pemborosan 0.180.
```  
3. **Rincian Emisi Karbon**
   - rincian detail dari setiap bahan sesuai dengan data yang saya berikan jangan gunakan tabel untuk bagian ini
   - buat dalam tabel untuk summary hasil akhir dengan persentase kontribusi setiap bahan buat simpel jangan terlalu panjang 3 colom saja buat ringkas
4. **Saran Keberlanjutan**:  
   - Berikan tips keberlanjutan yang spesifik, misalnya:  
     - 🌱 Kurangi makanan berlebih.  
     - 🌍 Pilih bahan lokal seperti minyak kelapa.  
     - 🍃 Utamakan kemasan yang bisa didaur ulang.  

5. **Fakta Penting dan Kutipan Inspiratif**:  
   - Berikan fakta unik tentang jejak karbon bahan makanan dan related dengan masalah carbon footprint di indonesia.  
   - Tambahkan kutipan inspiratif terkait keberlanjutan makanan.
   - Tambahkan kalimat dibawah ini:
Semoga informasi ini bermanfaat!
Sekarang lakukan langkah kecil dengan cara Donasikan makanan berlebihanmu, ketimbang dibuang dan merugikan lingkungan sekitarmu. Donasikan makanan berlebihmu melalui fitur Drop Point dan temukan lokasi terdekat untuk berbagi. Langkah kecil ini bantu sesama dan jaga lingkungan. Coba sekarang!
*note jangan pernah memakai tabel kecuali untuk bagian summary hasi akhir"""

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
            "message": "Success fetching the API",
        },
        "data": None
    }), 200


@app.route("/predict_image", methods=["POST"])
def predict_image():
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
        image.save(file_path)  # Simpan gambar

        # Preprocess the image
        img_array = preprocess_image(file_path)

        # Start timing prediction
        start_predict_time = time.time()

        # Predict the image
        prediction = model.predict(img_array)
        predicted_class, confidence = decode_prediction(prediction)

        # Generate text using Vertex AI
        generated_text = generate_text(predicted_class)

        # End timing prediction
        end_predict_time = time.time()
        predict_time = end_predict_time - start_predict_time

        # Upload image to cloud storage
        upload_to_gcs(file_path, BUCKET_NAME,
                      f"upload_image/{unique_filename}")

        # Remove the saved image after prediction
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
        system_instruction=textsi_1
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


if __name__ == "__main__":
    # Create upload folder if not exists
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

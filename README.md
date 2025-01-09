# NV-BITE Machine Learning API

Welcome to the NV-BITE Machine Learning API repository. This project provides an API for predicting waste categories using machine learning models. Below, you will find detailed information about the technology, tools, setup, and usage instructions to help you understand and run the project effectively.

---

## Tech Stack

- **TensorFlow**: Machine Learning library for building and deploying the model.
- **Flask**: Python framework for building the API.
- **NumPy**: Library for numerical and scientific computing.
- **Pillow**: Library for image processing.
- **Cloud Run**: Deployment platform for running APIs in the cloud.

---

## Tools

- [Google Cloud Platform](https://cloud.google.com/) for hosting and deployment.
- [Docker](https://docs.docker.com/manuals/) for containerization.
- [Postman](https://www.postman.com/) for API testing and documentation.

---

## Getting Started

To set up and run the NV-BITE Machine Learning API on your local machine, follow these steps:

### Prerequisites

Ensure you have the following installed on your machine:
- Python 3.8 or higher
- pip (Python package manager)
- Flask
- TensorFlow
- Docker (optional for containerized deployment)
- service gcp (especially service secret manager, gcs)

### Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/NV-Bite/ml-api.git
   cd ml-api
   ```

2. **Create and Configure Environment Variables**:
   Create a `.env.example` file in the root directory with the following content:
   ```env
   PROJECT_ID=your_project_id
   ```
3. **Set Up Secret Manager Variables**
   Store the following keys as secrets in your Secret Manager to securely handle sensitive information:
   ```json
   {
     "type": "service_account",
     "project_id": "your_project_id",
     "location": "your_location",
     "bucket_name": "your_bucket_name",
     "model_path": "your_model_path",
     "model_id": "your_model_id",
     "system_instruction": "your_instruction"
   }
   ```

3. **Install Dependencies**:
   Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the Server**:
   Run the Flask server locally:
   ```bash
   flask run
   ```
   The API will start at `http://127.0.0.1:5000`.

5. **Test the API**:
   Use Postman or any other API testing tool to send requests to the API.

---

## Endpoints

### 1. **Predict Endpoint**

- **URL**: `POST /predict_image`
- **Content-Type**: `multipart/form-data`

#### Request Body:
Upload an image file to be classified by the model:
```json
{
  "image": "sate-image.jpg"
}
```

#### Response:
```json
{
  "response": {
    "code": 200,
    "data": {
      "GenText": "<h2>Jejak Karbon Nasi Padang:</h2><h3>Mengerti Jejak Karbon Makanan:</h3><p>Jejak karbon makanan adalah total emisi gas rumah kaca yang dihasilkan dari proses produksi makanan, mulai dari penanaman bahan baku, pemanenan, pengolahan, penyimpanan, transportasi, pengemasan, hingga sampai di piring kita. Emisi ini diukur dalam satuan kilogram CO2 setara (CO2e)...."
      "confidence": 0.9954487681388855,
      "predict_time": 16.313554525375366,
      "label": "sate"
    },
    "status": {
        "code": 200,
        "message": "Success"
  }
}
```

---

## Deployment to Cloud Run

To deploy this API to Google Cloud Run:

1. **Build Docker Image**:
   ```bash
   docker build -t gcr.io/$PROJECT_ID/ml-api .
   ```

2. **Push Docker Image**:
   ```bash
   docker push gcr.io/$PROJECT_ID/ml-api
   ```

3. **Deploy to Cloud Run**:
   ```bash
   gcloud run deploy ml-api \
       --image gcr.io/$PROJECT_ID/ml-api \
       --platform managed \
       --region $LOCATION \
       --allow-unauthenticated
   ```

4. **Access the Deployed API**:
   The Cloud Run service will provide a URL for the API.

---

## API Documentation

Complete API documentation is available on Postman. You can view it [here](https://documenter.getpostman.com/view/39512380/2sAYHwL5qp).

Gentext helps provide context-sensitive text solutions.

---

## Troubleshooting

1. **Dependencies Not Installed**:
   Ensure you have the correct Python version and run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables Not Set**:
   Verify that the `.env` file exists and is correctly configured.

3. **API Not Starting**:
   Ensure Flask is installed and run the server using:
   ```bash
   flask run
   ```

4. **Deployment Issues**:
   Verify your Google Cloud configuration and Docker image build steps.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

Special thanks to the NV-BITE development team for their efforts in creating this application.

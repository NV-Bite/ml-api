# NV BITE MACHINE LEARNING API

```markdown
# Tech We Use in Capstone Project

- Tensorflow (Machine Learning Library)
- Flask (Python API Framework)
- numpy (Scientific Computing Library)
- Pillow (Image Processing Library)
- Cloud Run (API Deployment)
```

## Tools

- [Google Cloud Platform](https://cloud.google.com/)
- [Docker](https://docs.docker.com/manuals/)
- [Postman](https://www.postman.com/)

## Getting Started

To run this API on your local computer, follow these steps:

1. Clone this repository
   ```bash
    git clone https://github.com/NV-Bite/ml-api.git
    ```
2. Install the required depedencies
   ```bash
    pip install -r requirements.txt
    ```
3. Start the server
   ```bash
    flask run
    ```
4. The API will be running on http://127.0.0.1:5000 
5. You can test the API using Postman or any other API testing tool.

## Endpoints

- **Predict**
  <pre>POST /predict_image</pre>
  <pre>Content-Type: multipart/form-data</pre>

  Request Body:

  ```json
  {
    "file": "sate-image.jpg"
  }
  ```

  Response Body:

  ```json
  {
    "response": {
      "code": 200,
      "data": {
        "confidence": 0.9954487681388855,
        "label": "Organic-Waste"
      },
      "error": false,
      "message": "Waste successfully predicted!"
    }
  }
  ```
## API Documentation

We published our API documentation from Postman, you can view it [here](https://documenter.getpostman.com/view/39512380/2sAYHwL5qp)

# Gunakan image dasar yang sesuai
FROM python:3.10-slim

ENV PYTHONUNBUFFERED True
ENV APP_HOME /app

# Set working directory
WORKDIR $APP_HOME

# Salin seluruh kode aplikasi ke dalam container
COPY . ./

# Salin requirements.txt dan install dependencies
RUN pip install -r requirements.txt

# Jalankan aplikasi menggunakan gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
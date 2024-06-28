# activate conda environment and install requirements from requirements.txt
FROM python:3.10

WORKDIR /app

RUN pip install --upgrade pip

RUN apt-get update && apt-get install -y libhdf5-dev
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt .

RUN pip install --no-binary h5py h5py
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "app.py"]


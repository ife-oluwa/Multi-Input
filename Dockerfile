FROM python:3.10.6
WORKDIR /etc/easypanel/projects/multi-input/fastapi/code/
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip 
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --no-deps tensorflow-io
COPY . .
EXPOSE 80
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]

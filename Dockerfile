FROM python:3.10.6
WORKDIR /etc/easypanel/projects/multi-input/fastapi/code/
COPY requirements.txt requirements.txt
RUN pip install --ugrade pip 
RUN pip install -r requirements.txt
COPY . .
EXPOSE 80
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
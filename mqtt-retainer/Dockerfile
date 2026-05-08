FROM python:3.11-slim

WORKDIR /app

COPY mqtt-retainer/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY mqtt-retainer/app.py /app/app.py

CMD ["python", "/app/app.py"]

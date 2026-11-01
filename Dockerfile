# Учебный портал и упражнения в контейнере:
#   docker build -t mlcs .
#   docker run --rm -p 8000:8000 mlcs                     # портал: http://localhost:8000
#   docker run --rm -it mlcs python tools/check_exercise.py --list
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "start.py", "--host", "0.0.0.0", "--no-browser"]

FROM python:3.11-slim

WORKDIR /app

RUN pip install --upgrade pip setuptools

COPY . .

RUN pip install -e .

CMD ["python", "src/learning_to_plan/main.py"]

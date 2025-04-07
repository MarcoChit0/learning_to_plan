# Base image
FROM python:3.11-slim

# Environment config
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.2

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl build-essential git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Set working directory
WORKDIR /app

# Copy dependency files first (for caching)
COPY pyproject.toml poetry.lock* README.md ./

# Configure Poetry
RUN poetry config virtualenvs.create false

# Install only dependencies (skip installing the current project)
RUN poetry install --no-root --no-interaction --no-ansi

# Now copy the full project source
COPY . .

# Run main entrypoint
CMD ["python", "src/learning_to_plan/main.py"]
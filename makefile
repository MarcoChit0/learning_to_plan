# Makefile

APP_NAME = learning_to_plan
DOCKER_IMAGE = $(APP_NAME):latest
DOCKER_CONTAINER = $(APP_NAME)-container
ENV_FILE = .env
CONDA_ENV_NAME = learning_to_plan_env

# Create virtual environment using conda + install dependencies with Poetry
venv:
	command -v poetry >/dev/null 2>&1 || { \
		echo "Poetry not found. Installing..."; \
		curl -sSL https://install.python-poetry.org | python3 -; \
		export PATH="$$HOME/.local/bin:$$PATH"; \
	}
	conda create -y -n learning_to_plan_env python=3.11
	PYTHON_PATH=$$(conda run -n learning_to_plan_env which python) && \
	$$HOME/.local/bin/poetry env use $$PYTHON_PATH && \
	$$HOME/.local/bin/poetry install

# Docker image build
build:
	docker build -f .docker/Dockerfile -t $(DOCKER_IMAGE) .

rebuild:
	docker build --no-cache -f .docker/Dockerfile -t $(DOCKER_IMAGE) .

# Run main script inside container (if Docker is available)
run:
	docker run --rm -it --env-file $(ENV_FILE) $(DOCKER_IMAGE)

# Run main script in dev mode with volume (if Docker is available)
dev-run:
	docker run --rm --network=host -v $(PWD):/app -w /app $(DOCKER_IMAGE) poetry run python src/learning_to_plan/main.py

# Run main script locally (without Docker) from the Poetry environment
local-run:
	poetry run python src/learning_to_plan/main.py

# Open a bash shell in the container
shell:
	docker run --rm -it --env-file $(ENV_FILE) --entrypoint /bin/bash $(DOCKER_IMAGE)

# Clean up docker artifacts
clean:
	docker system prune -f

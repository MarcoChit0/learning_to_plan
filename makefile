# Project config
APP_NAME = learning-to-plan
DOCKER_IMAGE = $(APP_NAME):latest
DOCKER_CONTAINER = $(APP_NAME)-container
ENV_FILE = .env

build:
	docker build -f .docker/Dockerfile -t $(DOCKER_IMAGE) .

rebuild:
	docker build --no-cache -f .docker/Dockerfile -t $(DOCKER_IMAGE) .

run:
	docker run --rm -it --env-file $(ENV_FILE) $(DOCKER_IMAGE)

dev-run:
	docker run --rm --network=host -v $(PWD):/app -w /app $(DOCKER_IMAGE) poetry run python src/learning_to_plan/main.py


shell:
	docker run --rm -it --env-file $(ENV_FILE) --entrypoint /bin/bash $(DOCKER_IMAGE)

clean:
	docker system prune -f
SHELL := /bin/bash

# Docker config
DOCKER_IMAGE = learning_to_plan
DOCKER_TAG = latest
DOCKERFILE_PATH = .docker/Dockerfile

# Environment file (optional)
ENV_FILE = .env
ENV_VARS := $(shell [ -f $(ENV_FILE) ] && cat $(ENV_FILE) | xargs)

# Default target
default: docker-build

docker-build:
	docker build -f $(DOCKERFILE_PATH) -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-run:
	docker run -it --rm \
		--env-file $(ENV_FILE) \
		-v $$(pwd):/app \
		$(DOCKER_IMAGE):$(DOCKER_TAG)

docker-dev-run:
	docker run -it --rm \
		--env-file $(ENV_FILE) \
		-v $$(pwd):/app \
		--entrypoint "bash" \
		$(DOCKER_IMAGE):$(DOCKER_TAG)

docker-shell:
	docker run -it --rm \
		--env-file $(ENV_FILE) \
		-v $$(pwd):/app \
		--entrypoint /bin/bash \
		$(DOCKER_IMAGE):$(DOCKER_TAG)
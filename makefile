SHELL := /bin/bash

# Docker config
DOCKER_IMAGE = learning_to_plan
DOCKER_TAG = latest
DOCKERFILE_PATH = .docker/Dockerfile

# Default target
default: build

build:
	docker build -f $(DOCKERFILE_PATH) -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

run:
	docker run -it --rm \
		--env-file $(ENV_FILE) \
		-v $$(pwd):/app \
		$(DOCKER_IMAGE):$(DOCKER_TAG)

dev-run:
	docker run -it --rm \
		--env-file $(ENV_FILE) \
		-v $$(pwd):/app \
		--entrypoint "bash" \
		$(DOCKER_IMAGE):$(DOCKER_TAG)


shell:
	docker run -it --rm \
		--env-file $(ENV_FILE) \
		-v $$(pwd):/app \
		--entrypoint /bin/bash \
		$(DOCKER_IMAGE):$(DOCKER_TAG)

clean:
	-docker rmi $(DOCKER_IMAGE):$(DOCKER_TAG)
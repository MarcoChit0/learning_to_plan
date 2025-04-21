# Makefile

# --- Configuration ---
# Updated names to use underscores
CONDA_ENV_NAME = learning_to_plan
DOCKER_IMAGE_NAME = learning_to_plan_app
MAIN_SCRIPT = main.py # Path to your main script at the root
VAL_DIR = utils/VAL # Path to VAL submodule
VAL_BUILD_DIR = $(VAL_DIR)/build

# --- Default Target ---
# The default action when running 'make' will be to run using Conda.
.DEFAULT_GOAL := conda_run

# --- Phony Targets ---
# Declare targets that are not files.
.PHONY: all submodules build_val conda_env conda_run conda_clean docker_build docker_run docker_clean help

all: conda_run # 'make all' is equivalent to 'make conda_run'

help:
	@echo "Available commands:"
	@echo "  make submodules      Initialize/update Git submodules"
	@echo "  make conda_env       Create/update the Conda environment (${CONDA_ENV_NAME}) (includes submodules, build tools, project install)"
	@echo "  make build_val       Build the VAL submodule (runs *after* conda_env, uses tools from Conda env)"
	@echo "  make conda_run       Run the main script (${MAIN_SCRIPT}) using the Conda environment (default)"
	@echo "                       NOTE: Runs without arguments. For arguments, activate env first:"
	@echo "                       'conda activate ${CONDA_ENV_NAME}' then 'python ${MAIN_SCRIPT} --your-args'"
	@echo "  make conda_clean     Remove the Conda environment"
	@echo "  make docker_build    Build the Docker image (${DOCKER_IMAGE_NAME})"
	@echo "                       NOTE: Ensure Dockerfile initializes submodules and builds VAL!"
	@echo "  make docker_run      Run the main script using the Docker container"
	@echo "  make docker_clean    Remove the Docker image"
	@echo "  make help            Show this help message"

# --- Git Submodule Target ---
# Ensures submodules are cloned and updated
submodules:
	@echo ">>> Initializing/updating Git submodules..."
	@git submodule update --init --recursive

# --- Conda Targets ---
# Make conda_env depend on submodules target
# This target now installs Python, Pip, VAL build tools, and runs 'pip install -e .'
conda_env: submodules environment.yml
	@echo ">>> Creating/updating Conda environment '${CONDA_ENV_NAME}'..."
	@echo ">>> Installing Python, Pip, VAL build tools, and project dependencies..."
	@echo ">>> Make sure 'name: ${CONDA_ENV_NAME}' is set in environment.yml"
	@conda env create -f environment.yml || conda env update -n ${CONDA_ENV_NAME} -f environment.yml --prune
	@echo ">>> Conda environment ready. Activate with: conda activate ${CONDA_ENV_NAME}"

# --- Build VAL Target ---
# Builds the C++ VAL tools using CMake, AFTER conda_env is ready
# Runs commands within the conda environment to use its build tools
build_val: conda_env
	@echo ">>> Building VAL submodule in $(VAL_DIR) using Conda env '${CONDA_ENV_NAME}'..."
	@conda run -n ${CONDA_ENV_NAME} bash -c '\
	    echo ">>> Creating VAL build directory..." && \
	    mkdir -p $(VAL_BUILD_DIR) && \
	    echo ">>> Running CMake and Make for VAL..." && \
	    cd $(VAL_BUILD_DIR) && \
	    cmake .. && \
	    make -j$$(nproc) \
	'
	@echo ">>> VAL build complete. Executables should be in $(VAL_BUILD_DIR)"


# Note: This runs the script without arguments.
# For arguments, it's usually better to activate the environment manually first.
conda_run: conda_env
	@echo ">>> Running script '${MAIN_SCRIPT}' using Conda environment '${CONDA_ENV_NAME}'..."
	@conda run -n ${CONDA_ENV_NAME} python ${MAIN_SCRIPT}

conda_clean:
	@echo ">>> Removing Conda environment '${CONDA_ENV_NAME}'..."
	@conda env remove -n ${CONDA_ENV_NAME} --yes || echo "Conda environment '${CONDA_ENV_NAME}' not found or already removed."

# --- Docker Targets ---
# Docker build depends on the Dockerfile containing the right commands
docker_build: Dockerfile
	@echo ">>> Building Docker image '${DOCKER_IMAGE_NAME}'..."
	@echo ">>> Ensure your Dockerfile installs build tools, runs 'git submodule update --init --recursive', and builds VAL."
	@docker build -t ${DOCKER_IMAGE_NAME} .

# Runs the container using the CMD defined in your Dockerfile
docker_run: docker_build
	@echo ">>> Running script via Docker container '${DOCKER_IMAGE_NAME}'..."
	@docker run --rm ${DOCKER_IMAGE_NAME}

docker_clean:
	@echo ">>> Removing Docker image '${DOCKER_IMAGE_NAME}'..."
	@docker image rm ${DOCKER_IMAGE_NAME} || echo "Docker image '${DOCKER_IMAGE_NAME}' not found or already removed."
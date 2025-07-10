# Makefile

# --- Configuration ---
# Updated names to use underscores
CONDA_ENV_NAME = learning_to_plan
DOCKER_IMAGE_NAME = learning_to_plan_app
MAIN_SCRIPT = main.py# Path to your main script at the root
VAL_DIR = utils/VAL# Path to VAL submodule
DOWNWARD_DIR = utils/downward# Path to Downward submodule
VAL_BUILD_DIR = $(VAL_DIR)/build# Should evaluate to utils/VAL/build
ENV_FILE ?= .env # Default environment file name (can be overridden: make docker_bash ENV_FILE=my.env)

# --- Default Target ---
# The default action when running 'make' will be to run using Conda.
.DEFAULT_GOAL := conda_run

# --- Phony Targets ---
# Declare targets that are not files.
.PHONY: all submodules build_val conda_env conda_run dev_run_conda conda_clean docker_build dev_run_docker docker_bash docker_run docker_clean help

all: conda_run # 'make all' is equivalent to 'make conda_run'

help:
	@echo "Available commands:"
	@echo "  make submodules      Initialize/update Git submodules"
	@echo "  --- Conda Workflow ---"
	@echo "  make conda_env       Create/update the Conda environment (${CONDA_ENV_NAME})"
	@echo "  make build_val       Build the VAL submodule using Conda env tools"
	@echo "  make conda_run       Run the main script (${MAIN_SCRIPT}) using Conda (default, no args)"
	@echo "  make conda_clean     Remove the Conda environment"
	@echo "  --- Docker Workflow ---"
	@echo "  make docker_build    Build the Docker image (${DOCKER_IMAGE_NAME})"
	@echo "  make dev_run_docker  Ensures Docker image is built, shows how to run manually with args"
	@echo "  make docker_bash     Start an interactive bash shell inside the Docker container"
	@echo "  make docker_run      Run the main script using the Docker container (no args)"
	@echo "  make docker_clean    Remove the Docker image"
	@echo "  --- General ---"
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

# --- Build VAL Target (for Conda) ---
# Builds the C++ VAL tools using CMake, AFTER conda_env is ready
# Runs commands within the conda environment to use its build tools
build_val: conda_env
	@echo ">>> Building VAL submodule in $(VAL_DIR) using Conda env '${CONDA_ENV_NAME}'..."
	# Runs the VAL build process inside the specified conda environment.
	@conda run -n ${CONDA_ENV_NAME} bash -c '\
	    set -e; \
	    BUILD_DIR="$(VAL_BUILD_DIR)"; \
	    mkdir -p "$${BUILD_DIR}"; \
	    cd "$${BUILD_DIR}"; \
	    cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..; \
	    make -j$$(nproc); \
	'
	@echo ">>> VAL build complete. Executables should be in $(VAL_BUILD_DIR)"

# --- Build Downward Dependencies ---
build_downward: conda_env
	@echo ">>> Building Downward submodule in $(DOWNWARD_DIR) using Conda env '${CONDA_ENV_NAME}'..."
	@conda run -n ${CONDA_ENV_NAME} bash -c '\
	    set -e; \
	    cd "$(DOWNWARD_DIR)"; \
	    ./build.py \
	'
	@echo ">>> Downward build complete. Executables should be in $(DOWNWARD_DIR)/install"


# --- Conda Run Target (Default, No Args) ---
# Note: This runs the script without arguments.
# For arguments, it's usually better to use 'make dev_run_conda' and run manually.
conda_run: conda_env
	@echo ">>> Running script '${MAIN_SCRIPT}' using Conda environment '${CONDA_ENV_NAME}' (no arguments)..."
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

# --- Docker Development Run Target ---
# Ensures Docker image is built, then provides instructions to run manually
dev_run_docker: docker_build
	@echo ">>> Docker image '${DOCKER_IMAGE_NAME}' is built."
	@echo ">>> To run your script inside the container:"
	@echo "1. Start the container interactively (mounting current dir recommended for code changes):"
	@echo "   docker run --rm -it -v $$(pwd):/app ${DOCKER_IMAGE_NAME} bash"
	@echo "2. Inside the container's bash prompt, run your script:"
	@echo "   python ${MAIN_SCRIPT} [your arguments]"
	@echo "   (Note: VAL executables should be in /app/utils/VAL/build if built in Dockerfile)"

# --- Docker Bash Target ---
# Starts an interactive bash shell inside the container for development/debugging
# Replicates the old 'dev-run' functionality
docker_bash: docker_build
	@echo ">>> Starting interactive bash shell in container '${DOCKER_IMAGE_NAME}'..."
	@echo ">>> Mounting local directory $$(pwd) to /app"
	@docker run -it --rm \
		-v $$(pwd):/app \
		--entrypoint "bash" \
		${DOCKER_IMAGE_NAME}
		# Uncomment below line if you need to load environment variables from a file
		# --env-file $(ENV_FILE) \


# --- Docker Run Target (No Args) ---
# Runs the container using the CMD defined in your Dockerfile
docker_run: docker_build
	@echo ">>> Running script via Docker container '${DOCKER_IMAGE_NAME}' (no arguments)..."
	@echo ">>> For arguments, use 'make dev_run_docker' instructions or 'make docker_bash'."
	@docker run --rm ${DOCKER_IMAGE_NAME}

docker_clean:
	@echo ">>> Removing Docker image '${DOCKER_IMAGE_NAME}'..."
	@docker image rm ${DOCKER_IMAGE_NAME} || echo "Docker image '${DOCKER_IMAGE_NAME}' not found or already removed."
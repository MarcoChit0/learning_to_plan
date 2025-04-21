# Use the official Python 3.11 slim image as a base
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install git (for submodules) and C++ build tools (for VAL)
# Also upgrade pip and setuptools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        cmake \
        make \
        g++ \
        flex \
        bison \
    && \
    pip install --no-cache-dir --upgrade pip setuptools && \
    rm -rf /var/lib/apt/lists/*

# Copy the entire project context (including .git, .gitmodules, source code, pyproject.toml, etc.)
# into the working directory inside the container
COPY . .

# Initialize and update Git submodules recursively
# This ensures the code for submodules like VAL is checked out
RUN git submodule update --init --recursive

# Build the VAL submodule using CMake and Make
# Assumes VAL submodule is located at utils/VAL
# Creates executables like Validate inside /app/utils/VAL/build/
RUN mkdir -p /app/utils/VAL/build && \
    cd /app/utils/VAL/build && \
    cmake .. && \
    make -j$(nproc) # Use nproc to build in parallel using available cores

# Install the local package 'learning_to_plan' in editable mode (-e)
# Pip will read pyproject.toml to find dependencies and install them.
# Using --no-cache-dir to keep the image smaller
RUN pip install --no-cache-dir -e .

# Set the default command to execute when the container starts
# Runs your main application script.
CMD ["python", "main.py"]
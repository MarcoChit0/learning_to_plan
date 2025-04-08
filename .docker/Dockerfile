FROM continuumio/miniconda3

# Create working directory
WORKDIR /app

# Copy environment.yml first so Docker can cache this step if unchanged
COPY environment.yml .

# Create the conda environment
RUN conda env create -n learning_to_plan -f environment.yml

# Ensure shell will activate conda by default
SHELL ["/bin/bash", "-c"]

# Copy the rest of the project files
COPY . .

# Install project dependencies with poetry inside the conda environment
RUN conda run -n learning_to_plan poetry install

# You can specify a default command:
CMD ["conda", "run", "-n", "learning_to_plan", "python", "main.py"]
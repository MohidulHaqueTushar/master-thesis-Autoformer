IMAGE := autoformer            # Name of the Docker image

ROOT := $(shell dirname $(realpath $(firstword ${MAKEFILE_LIST}))) # Project root directory (where Makefile is located)

DOCKER_PARAMETERS := \        # Parameters for running Docker containers:
	--user $(shell id -u) \   #   Run as current user (user ID)
	--gpus all \              #   Enable all GPUs for the container
	-v ${ROOT}:/app \         #   Mount project root to /app inside container
	-w /app \                 #   Set working directory to /app inside container
	-e HOME=/tmp              #   Set HOME environment variable to /tmp

init:
	docker build -t ${IMAGE} .  # Build Docker image with tag "autoformer" from Dockerfile in current directory

get_dataset:
	mkdir -p dataset/ && \                 # Create 'dataset' directory if not exist
		make run_module module="python -m utils.download_data" && \ # Download data using Python module inside Docker
		unzip dataset/datasets.zip -d dataset/ && \                 # Unzip datasets.zip to 'dataset' directory
		mv dataset/all_six_datasets/* dataset && \                  # Move all files from extracted folder up to 'dataset'
		rm -r dataset/all_six_datasets dataset/__MACOSX             # Remove unneeded folders after extraction

run_module: .require-module
	docker run -i --rm ${DOCKER_PARAMETERS} \  # Run a command/module inside Docker container
		${IMAGE} ${module}                      # Uses value passed to 'module' variable

bash_docker:
	docker run -it --rm ${DOCKER_PARAMETERS} ${IMAGE}   # Start interactive bash session inside Docker container

.require-module:
ifndef module
	$(error module is required)   # Error if 'module' variable not provided to run_module
endif
# Created by: George Corrêa de Araújo (george.gcac@gmail.com)

# ==================================================================
# environment variables
# ------------------------------------------------------------------

CACHE_PATH = $(HOME)/.cache
DATASETS_PATH = $(HOME)/datasets/super_resolution
DOCKERFILE_CONTEXT = $(PWD)
DOCKERFILE = Dockerfile
WORK_DIR = $(PWD)
RUN_STRING = bash start_here.sh

# ==================================================================
# Docker settings
# ------------------------------------------------------------------

# add gnu screen session name to CONTAINER_NAME, so when doing `make run`
# from a different session it will run without issues
CONTAINER_NAME = sr-pytorch-$(USER)-$(shell echo $$STY | cut -d'.' -f2)
CONTAINER_FILE = sr-pytorch-$(USER).tar
DOCKER_RUN = docker run --gpus all
HOSTNAME = docker-$(shell hostname)
IMAGE_NAME = $(USER)/sr-pytorch
# username inside the container
USERNAME = $(USER)
WORK_PATH = /work

CACHE_MOUNT_STRING = --mount type=bind,source=$(CACHE_PATH),target=/home/$(USERNAME)/.cache
DATASET_MOUNT_STRING = --mount type=bind,source=$(DATASETS_PATH),target=/datasets
# needed with you want to interactively debug code with `ipdb`
PDB_MOUNT_STRING = --mount type=bind,source=$(HOME)/.pdbhistory,target=/home/$(USERNAME)/.pdbhistory
RUN_CONFIG_STRING = --name $(CONTAINER_NAME) --hostname $(HOSTNAME) --rm -it --dns 8.8.8.8 \
	--userns=host --ipc=host --ulimit memlock=-1 -w $(WORK_PATH) $(IMAGE_NAME):latest
# needed to send message to a telegram bot when finished execution
TELEGRAM_BOT_MOUNT_STRING = --mount type=bind,source=$(HOME)/Docker/telegram_bot_config,target=/home/$(USERNAME)/.config
WORK_MOUNT_STRING = --mount type=bind,source=$(WORK_DIR),target=$(WORK_PATH)

# ==================================================================
# Make commands
# ------------------------------------------------------------------

# Build image
# the given arguments during build are used to create a user inside docker image
# that have the same id as the local user. This is useful to avoid creating outputs
# as a root user, since all generated data will be owned by the local user
build:
	docker build \
		--build-arg GROUPID=$(shell id -g) \
		--build-arg GROUPNAME=$(shell id -gn) \
		--build-arg USERID=$(shell id -u) \
		--build-arg USERNAME=$(USERNAME) \
		-f $(DOCKERFILE) \
		--pull --no-cache --force-rm \
		-t $(IMAGE_NAME) \
		$(DOCKERFILE_CONTEXT)

	if (hash telegram-send 2>/dev/null); then \
	  telegram-send "Finished building $(IMAGE_NAME) image on $(shell hostname)."; \
	fi


# Remove the image
clean:
	docker rmi $(IMAGE_NAME)


# Load image from file
load:
	docker load -i $(CONTAINER_FILE)


# Kill running container
kill:
	docker kill $(CONTAINER_NAME)


# Run RUN_STRING inside container
run:
	$(DOCKER_RUN) \
		$(DATASET_MOUNT_STRING) \
		$(WORK_MOUNT_STRING) \
		$(CACHE_MOUNT_STRING) \
		$(PDB_MOUNT_STRING) \
		$(TELEGRAM_BOT_MOUNT_STRING) \
		$(RUN_CONFIG_STRING) \
		$(RUN_STRING)


# Save image to file
save:
	docker save -o $(CONTAINER_FILE) $(IMAGE_NAME)


# Start container by opening shell
start:
	$(DOCKER_RUN) \
		$(DATASET_MOUNT_STRING) \
		$(WORK_MOUNT_STRING) \
		$(CACHE_MOUNT_STRING) \
		$(PDB_MOUNT_STRING) \
		$(TELEGRAM_BOT_MOUNT_STRING) \
		$(RUN_CONFIG_STRING)


# Test image by printing some info
test:
	$(DOCKER_RUN) \
		$(RUN_CONFIG_STRING) \
		python -c 'import torch as t; print("Found", t.cuda.device_count(), "devices:"); [print (f"\t{t.cuda.get_device_properties(i)}") for i in range(t.cuda.device_count())]'

IMAGE_NAME = pyiris

# Build the Docker image
build:
	docker build -t $(IMAGE_NAME) .

# Run the container with args: make run ARGS="-i /etc/pyiris/input -o /etc/pyiris/output"
run:
	docker run --rm \
	  -v /etc/pyiris/input:/etc/pyiris/input \
	  -v /etc/pyiris/output:/etc/pyiris/output \
	  -v /etc/pyiris/log:/etc/pyiris/log \
	  $(IMAGE_NAME) $(ARGS)

# Rebuild without cache (if dependencies change)
rebuild:
	docker build --no-cache -t $(IMAGE_NAME) .

# Debug shell inside container
shell:
	docker run --rm -it \
	  -v /etc/pyiris/input:/etc/pyiris/input \
	  -v /etc/pyiris/output:/etc/pyiris/output \
	  -v /etc/pyiris/log:/etc/pyiris/log \
	  $(IMAGE_NAME) bash


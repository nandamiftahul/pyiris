FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libx11-6 \
    python3-tk \
    tk-dev \
    libtk8.6 \
    git \
    && rm -rf /var/lib/apt/lists/* 

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Workdir
WORKDIR /app

# Copy main script and config
COPY pyiris.py .
COPY pyiris.conf /etc/pyiris/pyiris.conf

# Copy pyirisgraph library
COPY plot/ /etc/pyiris/plot/

# Copy your custom hdf52mod folder
COPY hdf52mod/ /etc/pyiris/hdf52mod/

# Copy patches (if any) into pyart
COPY patches/ /tmp/patches/
RUN python - <<EOF
import pyart, shutil, pathlib
base = pathlib.Path(pyart.__file__).parent
shutil.copytree('/tmp/patches', base, dirs_exist_ok=True)
EOF

# Default entrypoint
ENTRYPOINT ["python", "pyiris.py"]

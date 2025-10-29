#!/usr/bin/env bash
set -euo pipefail

export DISPLAY=:0.0

IMAGE_NAME="pyiris"
DEST_DIR="$HOME/pyiris-docker"
ARCHIVE=""
USER="terrindo"

usage() {
  cat <<EOF
Usage: $(basename "$0") -a /path/to/pyiris-docker.tar.gz [options]

Options:
  -a, --archive PATH   (required) path to pyiris-docker.tar.gz
  -i, --image NAME     docker image name (default: ${IMAGE_NAME})
  -d, --dest DIR       extract project into this directory (default: ${DEST_DIR})
  -h, --help           show this help
EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -a|--archive) ARCHIVE="${2:-}"; shift 2;;
    -i|--image)   IMAGE_NAME="${2:-}"; shift 2;;
    -d|--dest)    DEST_DIR="${2:-}"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1"; usage; exit 1;;
  esac
done

if [[ -z "${ARCHIVE}" ]]; then
  echo "ERROR: --archive PATH.tar.gz is required"; usage; exit 1
fi
if [[ ! -f "${ARCHIVE}" ]]; then
  echo "ERROR: Archive not found: ${ARCHIVE}"; exit 1
fi

log() { echo -e "\n=== $* ==="; }

# ======================================
# Check OS
# ======================================
if ! grep -q "Linux Mint" /etc/os-release; then
  echo "ERROR: This installer is intended for Linux Mint 22 (Ubuntu 24.04 base)."
  exit 1
fi

# ======================================
# Install Docker if missing
# ======================================
if ! command -v docker >/dev/null 2>&1; then
  log "Installing Docker CE on Linux Mint 22"
  sudo apt-get update
  sudo apt-get install -y ca-certificates curl gnupg lsb-release

  # Add Docker’s official GPG key
  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  sudo chmod a+r /etc/apt/keyrings/docker.gpg

  # Add Docker repository (Ubuntu Noble → Mint 22 base)
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu noble stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

  sudo apt-get update
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

  sudo systemctl enable --now docker
  sudo usermod -aG docker "$USER" || true
  log "Docker installed. ⚠️ Please log out and back in (or run 'newgrp docker') to use Docker without sudo."
else
  log "Docker already installed"
fi

# ======================================
# Prepare host directories
# ======================================
log "Creating /etc/pyiris directories"
sudo mkdir -p /etc/pyiris/input /etc/pyiris/output /etc/pyiris/log /etc/pyiris/hdf52mod /etc/pyiris/temp
sudo chown -R "${USER}":"${USER}" /etc/pyiris
sudo chmod 755  /etc/pyiris/*


# ======================================
# Extract and build image
# ======================================
log "Extracting archive into ${DEST_DIR}"
mkdir -p "${DEST_DIR}"
tar -xzf "${ARCHIVE}" -C "${DEST_DIR}" --strip-components=1
cp "${DEST_DIR}"/pyiris.conf /etc/pyiris/
cp "${DEST_DIR}"/hdf52mod/hdf52mod.conf /etc/pyiris/hdf52mod/hdf52mod.conf

log "Building Docker image: ${IMAGE_NAME}"
docker build -t "${IMAGE_NAME}" "${DEST_DIR}"

# ======================================
# Install wrapper command
# ======================================
log "Installing wrapper /usr/local/bin/pyiris"
sudo tee /usr/local/bin/pyiris >/dev/null <<WRAP
#!/usr/bin/env bash
set -euo pipefail
docker run --rm \
  -v /etc/pyiris/input:/etc/pyiris/input \
  -v /etc/pyiris/output:/etc/pyiris/output \
  -v /etc/pyiris/log:/etc/pyiris/log \
  -v /etc/pyiris/temp:/etc/pyiris/temp \
  -v /etc/pyiris/hdf52mod:/etc/pyiris/hdf52mod \
  -v /etc/pyiris/hdf52mod/hdf52mod.conf:/etc/pyiris/hdf52mod/hdf52mod.conf \
  -v /etc/pyiris/pyiris.conf:/etc/pyiris/pyiris.conf \
  -v /srv/iris_data:/srv/iris_data \
  ${IMAGE_NAME} "\$@"
WRAP
sudo chmod +x /usr/local/bin/pyiris

log "Add usergroup "${USER}" into docker"
sudo usermod -aG docker "${USER}"

log "update docker group"
newgrp docker

log "Installation complete!"

cat <<'POST'
Usage:
  Put input files into:   /etc/pyiris/input
  Run processing with:    pyiris -i /etc/pyiris/input -o /etc/pyiris/output
  Outputs will be in:     /etc/pyiris/output
  Logs are in:            /etc/pyiris/log

Note: If you were just added to the 'docker' group, log out & back in before running 'pyiris'.
POST

PyIRIS – Dockerized Setup
=========================

This package contains the PyIRIS project with a Dockerfile, requirements,
and helper installer script.

It supports:
- AlmaLinux 9 (RHEL-compatible)

------------------------------------------------------------
Installation Steps (AlmaLinux9)
------------------------------------------------------------

1. Copy archive to target machine:
   scp pyiris-docker.tar.gz user@target:/home/user/

2. Run the installer script as root:
   chmod +x install_pyiris.sh
   ./install_pyiris_alma9.sh -a pyiris-docker.tar.gz

   Options:
     -i NAME      custom Docker image name (default: pyiris)
     -d DIR       extract project into a specific directory (default: ~/pyiris-docker)
     -v           use Docker named volumes instead of host bind-mounts
     -b DIR       custom base directory for host bind mounts (default: /etc/pyiris)

------------------------------------------------------------
Manual Installation Steps (AlmaLinux 9)
------------------------------------------------------------

1. Ensure Docker (Moby Engine) is installed:

   sudo dnf -y install dnf-plugins-core
   sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
   sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
   sudo systemctl enable --now docker

   (If using Podman instead of Docker, you can alias `docker` to `podman`.)

2. Copy archive to target machine:
   scp pyiris-docker.tar.gz user@almalinux:/home/user/

3. Extract archive:
   tar -xzvf pyiris-docker.tar.gz
   cd pyiris-docker

4. Run the installer script:
   chmod +x install_pyiris.sh
   ./install_pyiris.sh -a pyiris-docker.tar.gz

   Options are the same as on Ubuntu.

------------------------------------------------------------
Storage Layout
------------------------------------------------------------

By default, the installer creates three directories on the host:

  /etc/pyiris/input    → for input files
  /etc/pyiris/output   → for output results
  /etc/pyiris/log      → for logs

If you chose -v, these paths are replaced by Docker volumes:
  pyiris_input, pyiris_output, pyiris_log

------------------------------------------------------------
Usage
------------------------------------------------------------

After installation, the installer creates a wrapper command 'pyiris'.
You can now run PyIRIS just like a native CLI tool:

  pyiris -i /etc/pyiris/input -o /etc/pyiris/output

You can also include your configuration file:

  pyiris --conf /etc/pyiris/pyiris.conf -i /etc/pyiris/input -o /etc/pyiris/output

All arguments you pass to 'pyiris' are forwarded directly to pyiris.py
inside Docker.

------------------------------------------------------------
Tips
------------------------------------------------------------

- Logs are written to:
    /etc/pyiris/log

- If using Docker volumes, you can inspect contents with:
    docker run --rm -v pyiris_output:/data busybox ls /data

- To rebuild the image later:
    cd ~/pyiris-docker
    docker build -t pyiris .

------------------------------------------------------------
Quick Recap
------------------------------------------------------------

- Install once with: ./install_pyiris.sh -a pyiris-docker.tar.gz
- Put files into:    /etc/pyiris/input
- Run processing:    pyiris ...
- Get results from:  /etc/pyiris/output

#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="pyiris-watcher"
WATCHER_SCRIPT="/usr/local/bin/pyiris_watcher.sh"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

# Detect package manager
if command -v dnf &>/dev/null; then
    PKG_MGR="dnf"
    INSTALL_CMD="sudo dnf install -y inotify-tools"
elif command -v apt-get &>/dev/null; then
    PKG_MGR="apt-get"
    INSTALL_CMD="sudo apt-get update && sudo apt-get install -y inotify-tools"
else
    echo "❌ Neither dnf nor apt-get found. Install inotify-tools manually."
    exit 1
fi

# 1. Install inotify-tools if missing
if ! command -v inotifywait &>/dev/null; then
    echo "📦 Installing inotify-tools..."
    eval "$INSTALL_CMD"
else
    echo "✅ inotify-tools already installed."
fi

# 2. Create watcher script
echo "📝 Creating watcher script at $WATCHER_SCRIPT"
sudo tee "$WATCHER_SCRIPT" > /dev/null <<'EOF'
#!/usr/bin/env bash
WATCH_DIR="/etc/pyiris/input"
OUTPUT_DIR="/etc/pyiris/output"
LOG_DIR="/etc/pyiris/log"

mkdir -p "$WATCH_DIR" "$OUTPUT_DIR" "$LOG_DIR"

echo "Watching $WATCH_DIR for new files..."

inotifywait -m -e create --format "%f" "$WATCH_DIR" | while read NEWFILE; do
    echo "[$(date)] New file detected: $NEWFILE"
    pyiris -i "$WATCH_DIR/$NEWFILE" -o "$OUTPUT_DIR" >> "$LOG_DIR/watcher.log" 2>&1
done
EOF

sudo chmod +x "$WATCHER_SCRIPT"

# 3. Create systemd service file
echo "📝 Creating systemd service at $SERVICE_FILE"
CURRENT_USER=$(whoami)
sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=PyIRIS Folder Watcher
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
ExecStart=$WATCHER_SCRIPT
Restart=on-failure
RestartSec=5
User=$CURRENT_USER
Group=$CURRENT_USER

[Install]
WantedBy=multi-user.target
EOF

# 4. Reload systemd and enable service
echo "🔄 Reloading systemd..."
sudo systemctl daemon-reload
sudo systemctl enable --now "$SERVICE_NAME"

echo "✅ PyIRIS watcher service installed and started!"
echo "👉 Manage it with:"
echo "   sudo systemctl start $SERVICE_NAME"
echo "   sudo systemctl stop $SERVICE_NAME"
echo "   sudo systemctl restart $SERVICE_NAME"
echo "   journalctl -u $SERVICE_NAME -f   # View logs"

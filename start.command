#!/bin/zsh
# Double-click to open the Local MLX Stack UI. Runs locally only (127.0.0.1:8788).
cd "$(dirname "$0")" || exit 1
# Prefer a venv outside iCloud: iCloud marks .pth files hidden, which breaks editable installs.
if [ -x "$HOME/.venvs/mlx-stack/bin/python" ]; then
    PY="$HOME/.venvs/mlx-stack/bin/python"
elif [ -d .venv ]; then
    chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null
    PY=".venv/bin/python"
else
    echo "No venv found. See README Quick start."; read -k 1; exit 1
fi
(sleep 1; open "http://127.0.0.1:8788") &
exec "$PY" -m local_mlx_stack serve --port 8788

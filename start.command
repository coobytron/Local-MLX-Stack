#!/bin/zsh
# Double-click to open the Local MLX Stack UI. Runs locally only (127.0.0.1:8788).
cd "$(dirname "$0")" || exit 1
[ -d .venv ] || { echo "No .venv here. See README Quick start."; read -k 1; exit 1; }
# iCloud can mark .pth files hidden, which breaks the editable install.
chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null
(sleep 1; open "http://127.0.0.1:8788") &
exec .venv/bin/python -m local_mlx_stack serve --port 8788

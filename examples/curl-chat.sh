#!/usr/bin/env bash
set -euo pipefail
curl --fail --silent --show-error http://127.0.0.1:8788/chat \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Describe an efficient image-analysis pipeline for SDXL ControlNet.","role":"auto"}'
echo

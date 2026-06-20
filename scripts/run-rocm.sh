#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

docker compose build voicebox-rocm
docker compose up -d voicebox-rocm

echo "Voicebox ROCm container is starting on http://localhost:17494"

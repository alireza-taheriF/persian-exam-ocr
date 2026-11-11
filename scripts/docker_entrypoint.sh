#!/bin/bash
set -euo pipefail
echo "Persian Exam OCR: running pre-flight self-test..."
python /app/scripts/docker_self_test.py
echo "Starting web server on http://0.0.0.0:8000"
exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1

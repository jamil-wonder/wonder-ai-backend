#!/bin/sh
set -e

if [ "$ROLE" = "worker" ]; then
  exec python worker.py
else
  exec gunicorn -w 1 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:10000 --timeout 120
fi

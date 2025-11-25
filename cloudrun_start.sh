#!/bin/bash
exec uvicorn ragthrones.app.main:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --workers 1
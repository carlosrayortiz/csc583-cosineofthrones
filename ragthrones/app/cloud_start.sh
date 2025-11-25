#!/usr/bin/env bash
set -e

echo "======================================="
echo "🚀 Starting Cosinify – Cloud Run"
echo "======================================="

# Cloud Run always injects PORT env var
export PORT=${PORT:-8080}

echo "🔧 Using PORT = $PORT"
echo "📦 Starting Uvicorn..."

# IMPORTANT:
# --no-access-log avoids huge stdout logs
# --host 0.0.0.0 required for Cloud Run
# --workers 1 is required for CPUs < 4 on Cloud Run
# --forwarded-allow-ips="*" needed because Cloud Run uses load balancer
uvicorn ragthrones.app.main:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers 1 \
    --no-access-log \
    --forwarded-allow-ips="*"
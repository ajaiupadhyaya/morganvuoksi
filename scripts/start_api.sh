#!/bin/bash

# MorganVuoksi Elite Terminal - Production API Startup
# MISSION CRITICAL: Bloomberg-grade API service startup

set -e

echo "🚀 Starting MorganVuoksi Elite Terminal API..."
echo "📅 Startup Time: $(date)"
echo "🌍 Environment: ${ENVIRONMENT:-production}"
echo "⚙️  Workers: ${WORKERS:-4}"
echo "🔧 Python Version: $(python --version)"

# Set production defaults
export WORKERS=${WORKERS:-4}
export HOST=${HOST:-0.0.0.0}
export PORT=${PORT:-8000}
export LOG_LEVEL=${LOG_LEVEL:-info}
export MAX_REQUESTS=${MAX_REQUESTS:-1000}
export MAX_REQUESTS_JITTER=${MAX_REQUESTS_JITTER:-100}
export TIMEOUT=${TIMEOUT:-120}
export KEEPALIVE=${KEEPALIVE:-5}
export PRELOAD=${PRELOAD:-true}

# Wait for dependencies
echo "⏳ Waiting for database connection..."
python -c "
import asyncio
import asyncpg
import os
import sys
import time

async def wait_for_db():
    max_retries = 30
    retry_interval = 2
    
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        print('❌ DATABASE_URL not set')
        sys.exit(1)
    
    for attempt in range(max_retries):
        try:
            conn = await asyncpg.connect(db_url)
            await conn.execute('SELECT 1')
            await conn.close()
            print('✅ Database connection successful')
            return
        except Exception as e:
            if attempt < max_retries - 1:
                print(f'⏳ Database connection attempt {attempt + 1}/{max_retries} failed: {e}')
                time.sleep(retry_interval)
            else:
                print(f'❌ Database connection failed after {max_retries} attempts')
                sys.exit(1)

asyncio.run(wait_for_db())
"

echo "⏳ Waiting for Redis connection..."
python -c "
import redis
import os
import sys
import time

max_retries = 30
retry_interval = 2

redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = int(os.getenv('REDIS_PORT', 6379))
redis_password = os.getenv('REDIS_PASSWORD')

for attempt in range(max_retries):
    try:
        r = redis.Redis(host=redis_host, port=redis_port, password=redis_password)
        r.ping()
        print('✅ Redis connection successful')
        break
    except Exception as e:
        if attempt < max_retries - 1:
            print(f'⏳ Redis connection attempt {attempt + 1}/{max_retries} failed: {e}')
            time.sleep(retry_interval)
        else:
            print(f'❌ Redis connection failed after {max_retries} attempts')
            sys.exit(1)
"

# Initialize database if needed
echo "🗄️  Initializing database..."
python -c "
import asyncio
from database.models import create_all_tables
from sqlalchemy.ext.asyncio import create_async_engine
import os

async def init_db():
    try:
        engine = create_async_engine(os.getenv('DATABASE_URL'))
        create_all_tables(engine)
        print('✅ Database tables initialized')
    except Exception as e:
        print(f'⚠️  Database initialization warning: {e}')

asyncio.run(init_db())
"

# Pre-warm models and caches
echo "🤖 Pre-warming ML models..."
python -c "
import sys
sys.path.append('/app')
try:
    from src.ml.ecosystem import MLEcosystem
    from src.config import get_config
    
    config = get_config()
    ml_ecosystem = MLEcosystem(config)
    print('✅ ML ecosystem pre-warmed')
except Exception as e:
    print(f'⚠️  ML pre-warming warning: {e}')
"

# Health check before startup
echo "🏥 Running pre-startup health check..."
python -c "
import requests
import subprocess
import time
import signal
import os

# Start the API in the background for health check
api_process = subprocess.Popen([
    'uvicorn', 'backend.main:app',
    '--host', '0.0.0.0',
    '--port', '8000',
    '--workers', '1'
], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Wait for startup
time.sleep(10)

try:
    # Health check
    response = requests.get('http://localhost:8000/api/v1/health', timeout=10)
    if response.status_code == 200:
        print('✅ Pre-startup health check passed')
    else:
        print(f'❌ Health check failed with status {response.status_code}')
        raise Exception('Health check failed')
except Exception as e:
    print(f'❌ Pre-startup health check failed: {e}')
finally:
    # Clean shutdown
    api_process.terminate()
    api_process.wait()
"

# Configure logging
export PYTHONPATH="/app:$PYTHONPATH"

# Create log file
mkdir -p /app/logs
touch /app/logs/api.log

echo "🎯 Starting production API server..."
echo "📍 Listening on ${HOST}:${PORT}"
echo "👥 Using ${WORKERS} workers"
echo "📊 Max requests per worker: ${MAX_REQUESTS}"
echo "⏱️  Request timeout: ${TIMEOUT}s"
echo "❤️  Keepalive: ${KEEPALIVE}s"

# Start with Gunicorn for production
if [ "${ENVIRONMENT}" = "development" ]; then
    echo "🛠️  Running in development mode with hot reload..."
    uvicorn backend.main:app \
        --host $HOST \
        --port $PORT \
        --log-level $LOG_LEVEL \
        --reload \
        --reload-dir /app/src \
        --reload-dir /app/backend
else
    echo "🚀 Running in production mode with Gunicorn..."
    gunicorn backend.main:app \
        --bind $HOST:$PORT \
        --workers $WORKERS \
        --worker-class uvicorn.workers.UvicornWorker \
        --max-requests $MAX_REQUESTS \
        --max-requests-jitter $MAX_REQUESTS_JITTER \
        --timeout $TIMEOUT \
        --keepalive $KEEPALIVE \
        --preload \
        --access-logfile /app/logs/access.log \
        --error-logfile /app/logs/error.log \
        --log-level $LOG_LEVEL \
        --capture-output \
        --enable-stdio-inheritance \
        --worker-tmp-dir /dev/shm \
        --worker-connections 1000
fi
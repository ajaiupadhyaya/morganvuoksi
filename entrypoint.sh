#!/bin/bash

# Wait for any services if needed
# Example: wait-for-it.sh db:5432 -t 60

# Run database migrations if needed
# Example: python manage.py migrate

# Start the application
if [ "$ENVIRONMENT" = "production" ]; then
    # Production settings
    gunicorn src.api.app:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
else
    # Development settings
    python src/main.py
fi 
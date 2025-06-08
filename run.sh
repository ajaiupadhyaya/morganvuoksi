#!/bin/bash

# Create necessary directories
mkdir -p data models exports logs

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found. Please create one from .env.example"
    exit 1
fi

# Check if running in Docker
if [ -f /.dockerenv ]; then
    echo "Running in Docker container..."
    python src/main.py
else
    # Check if Docker is installed
    if command -v docker &> /dev/null; then
        echo "Starting with Docker..."
        docker-compose up --build
    else
        echo "Docker not found. Starting locally..."
        python src/main.py
    fi
fi 
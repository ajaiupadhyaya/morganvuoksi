#!/bin/bash

# MorganVuoksi Terminal Deployment Script
# Supports local, Docker, and cloud deployments

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TERMINAL_NAME="MorganVuoksi Terminal"
VERSION="1.0.0"
DEFAULT_PORT=8501

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if Docker is installed
    if command -v docker &> /dev/null; then
        print_success "Docker is installed"
    else
        print_warning "Docker not found. Some deployment options will be unavailable."
    fi
    
    # Check if docker-compose is installed
    if command -v docker-compose &> /dev/null; then
        print_success "Docker Compose is installed"
    else
        print_warning "Docker Compose not found. Some deployment options will be unavailable."
    fi
    
    # Check Python
    if command -v python3 &> /dev/null; then
        print_success "Python 3 is installed"
    else
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check if .env file exists
    if [ -f ".env" ]; then
        print_success ".env file found"
    else
        print_warning ".env file not found. Creating template..."
        create_env_template
    fi
}

# Function to create .env template
create_env_template() {
    cat > .env << EOF
# MorganVuoksi Terminal - Environment Configuration
# Copy this file and fill in your API keys

# Trading & Market Data
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
POLYGON_API_KEY=your_polygon_api_key_here

# Economic Data
FRED_API_KEY=your_fred_api_key_here

# AI & NLP
OPENAI_API_KEY=your_openai_api_key_here

# News & Sentiment
NEWS_API_KEY=your_newsapi_key_here
ALPHA_VANTAGE_API_KEY=your_alphavantage_key_here

# Optional: Redis (for caching)
# REDIS_URL=redis://localhost:6379
EOF
    print_success "Created .env template. Please edit it with your API keys."
}

# Function to deploy locally
deploy_local() {
    print_status "Deploying $TERMINAL_NAME locally..."
    
    # Check if virtual environment exists
    if [ ! -d ".venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv .venv
    fi
    
    # Activate virtual environment
    print_status "Activating virtual environment..."
    source .venv/bin/activate
    
    # Install dependencies
    print_status "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
    # Create necessary directories
    mkdir -p logs outputs models/saved_models
    
    # Start the terminal
    print_success "Starting $TERMINAL_NAME locally..."
    print_status "Terminal will be available at: http://localhost:$DEFAULT_PORT"
    print_status "Press Ctrl+C to stop"
    
    cd dashboard
    streamlit run terminal.py --server.port $DEFAULT_PORT --server.address 0.0.0.0
}

# Function to deploy with Docker
deploy_docker() {
    print_status "Deploying $TERMINAL_NAME with Docker..."
    
    # Build the Docker image
    print_status "Building Docker image..."
    docker build -t morganvuoksi-terminal:$VERSION .
    
    # Start with docker-compose
    print_status "Starting services with Docker Compose..."
    docker-compose up -d
    
    print_success "Docker deployment completed!"
    print_status "Terminal is available at: http://localhost:$DEFAULT_PORT"
    print_status "To view logs: docker-compose logs -f"
    print_status "To stop: docker-compose down"
}

# Function to deploy to cloud (placeholder)
deploy_cloud() {
    print_status "Cloud deployment options:"
    echo "1. Heroku"
    echo "2. AWS"
    echo "3. Google Cloud"
    echo "4. Azure"
    echo "5. DigitalOcean"
    
    read -p "Select cloud provider (1-5): " choice
    
    case $choice in
        1) deploy_heroku ;;
        2) deploy_aws ;;
        3) deploy_gcp ;;
        4) deploy_azure ;;
        5) deploy_digitalocean ;;
        *) print_error "Invalid choice" ;;
    esac
}

# Function to deploy to Heroku
deploy_heroku() {
    print_status "Deploying to Heroku..."
    
    # Check if Heroku CLI is installed
    if ! command -v heroku &> /dev/null; then
        print_error "Heroku CLI is required. Please install it first."
        return 1
    fi
    
    # Create Heroku app
    heroku create morganvuoksi-terminal-$(date +%s)
    
    # Set buildpacks
    heroku buildpacks:set heroku/python
    
    # Set environment variables
    if [ -f ".env" ]; then
        while IFS= read -r line; do
            if [[ $line =~ ^[A-Z_]+= ]]; then
                heroku config:set "$line"
            fi
        done < .env
    fi
    
    # Deploy
    git add .
    git commit -m "Deploy to Heroku"
    git push heroku main
    
    print_success "Deployed to Heroku!"
    heroku open
}

# Function to deploy to AWS
deploy_aws() {
    print_status "AWS deployment requires additional setup."
    print_status "Please refer to the documentation for AWS deployment instructions."
}

# Function to deploy to Google Cloud
deploy_gcp() {
    print_status "Google Cloud deployment requires additional setup."
    print_status "Please refer to the documentation for GCP deployment instructions."
}

# Function to deploy to Azure
deploy_azure() {
    print_status "Azure deployment requires additional setup."
    print_status "Please refer to the documentation for Azure deployment instructions."
}

# Function to deploy to DigitalOcean
deploy_digitalocean() {
    print_status "DigitalOcean deployment requires additional setup."
    print_status "Please refer to the documentation for DigitalOcean deployment instructions."
}

# Function to stop deployment
stop_deployment() {
    print_status "Stopping $TERMINAL_NAME..."
    
    # Stop Docker containers if running
    if docker-compose ps | grep -q "morganvuoksi-terminal"; then
        docker-compose down
        print_success "Docker containers stopped"
    fi
    
    # Kill any running Streamlit processes
    pkill -f streamlit || true
    print_success "Local processes stopped"
}

# Function to show status
show_status() {
    print_status "Checking $TERMINAL_NAME status..."
    
    # Check Docker containers
    if docker-compose ps | grep -q "morganvuoksi-terminal"; then
        print_success "Docker containers are running"
        docker-compose ps
    else
        print_warning "No Docker containers running"
    fi
    
    # Check local processes
    if pgrep -f streamlit > /dev/null; then
        print_success "Local Streamlit process is running"
        ps aux | grep streamlit | grep -v grep
    else
        print_warning "No local Streamlit process running"
    fi
}

# Function to show logs
show_logs() {
    print_status "Showing logs..."
    
    if docker-compose ps | grep -q "morganvuoksi-terminal"; then
        docker-compose logs -f
    else
        print_warning "No Docker containers running. Check local logs in logs/ directory."
    fi
}

# Function to show help
show_help() {
    echo "MorganVuoksi Terminal Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  local     Deploy locally with Python"
    echo "  docker    Deploy with Docker"
    echo "  cloud     Deploy to cloud platform"
    echo "  stop      Stop all deployments"
    echo "  status    Show deployment status"
    echo "  logs      Show deployment logs"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 local     # Deploy locally"
    echo "  $0 docker    # Deploy with Docker"
    echo "  $0 stop      # Stop all deployments"
}

# Main script
main() {
    echo "🚀 $TERMINAL_NAME Deployment Script v$VERSION"
    echo "================================================"
    
    # Check prerequisites
    check_prerequisites
    
    # Parse command line arguments
    case "${1:-help}" in
        local)
            deploy_local
            ;;
        docker)
            deploy_docker
            ;;
        cloud)
            deploy_cloud
            ;;
        stop)
            stop_deployment
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@" 
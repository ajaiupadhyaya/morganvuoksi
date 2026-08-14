#!/bin/bash

# MorganVuoksi Terminal - Enhanced Railway.app Deployment Script
# Version 2.0 - Optimized for Railway deployment with comprehensive checks

set -e  # Exit on any error

echo "🚀 MorganVuoksi Terminal - Railway.app Deployment"
echo "=" * 50

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Railway CLI is installed
    if ! command -v railway &> /dev/null; then
        log_warning "Railway CLI not found. Installing..."
        if command -v npm &> /dev/null; then
            npm install -g @railway/cli
            log_success "Railway CLI installed successfully"
        else
            log_error "Node.js/npm not found. Please install Node.js first."
            exit 1
        fi
    else
        log_success "Railway CLI found"
    fi
    
    # Check if user is logged in
    if ! railway whoami &> /dev/null; then
        log_info "Not logged into Railway. Please login..."
        railway login
    else
        log_success "Logged into Railway as $(railway whoami)"
    fi
    
    # Check if Docker is available (for local testing)
    if command -v docker &> /dev/null; then
        log_success "Docker found - local testing available"
    else
        log_warning "Docker not found - skipping local tests"
    fi
}

# Verify deployment files
verify_deployment_files() {
    log_info "Verifying deployment files..."
    
    local files=(
        "Dockerfile"
        "requirements-railway.txt"
        "railway.json"
        "startup.py"
        "dashboard/terminal.py"
    )
    
    for file in "${files[@]}"; do
        if [[ -f "$file" ]]; then
            log_success "$file exists"
        else
            log_error "$file not found"
            exit 1
        fi
    done
}

# Run dependency check
check_dependencies() {
    log_info "Running dependency check..."
    
    if python startup.py --check; then
        log_success "Dependency check passed"
    else
        log_warning "Some dependencies missing - deployment may have limited functionality"
    fi
}

# Test Docker build locally (optional)
test_docker_build() {
    if command -v docker &> /dev/null && [[ "${SKIP_LOCAL_TEST:-false}" != "true" ]]; then
        log_info "Testing Docker build locally..."
        
        if docker build -t morganvuoksi-test -f Dockerfile .; then
            log_success "Docker build successful"
            docker rmi morganvuoksi-test &> /dev/null || true
        else
            log_error "Docker build failed"
            exit 1
        fi
    else
        log_info "Skipping local Docker test"
    fi
}

# Deploy to Railway
deploy_to_railway() {
    log_info "Deploying to Railway..."
    
    # Link to existing project or create new one
    if [[ -n "$RAILWAY_PROJECT_ID" ]]; then
        log_info "Linking to existing project: $RAILWAY_PROJECT_ID"
        railway link "$RAILWAY_PROJECT_ID"
    else
        log_info "Creating new Railway project..."
        railway init
    fi
    
    # Set essential environment variables
    log_info "Setting up environment variables..."
    railway variables set STREAMLIT_SERVER_PORT=\$PORT
    railway variables set STREAMLIT_SERVER_ADDRESS=0.0.0.0
    railway variables set STREAMLIT_SERVER_HEADLESS=true
    railway variables set STREAMLIT_SERVER_ENABLE_CORS=false
    railway variables set STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
    railway variables set PYTHONUNBUFFERED=1
    railway variables set PYTHONDONTWRITEBYTECODE=1
    railway variables set LOG_LEVEL=INFO
    railway variables set ENVIRONMENT=production
    
    # Performance optimizations
    railway variables set OMP_NUM_THREADS=4
    railway variables set MKL_NUM_THREADS=4
    railway variables set NUMEXPR_NUM_THREADS=4
    
    # Streamlit optimizations
    railway variables set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
    railway variables set STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
    
    log_success "Environment variables configured"
    
    # Deploy the application
    log_info "Starting deployment..."
    if railway up --detach; then
        log_success "Deployment initiated successfully"
    else
        log_error "Deployment failed"
        exit 1
    fi
}

# Monitor deployment
monitor_deployment() {
    log_info "Monitoring deployment..."
    
    # Wait for deployment to complete
    local max_attempts=30
    local attempt=0
    
    while [[ $attempt -lt $max_attempts ]]; do
        if railway status --json > /tmp/railway_status.json 2>/dev/null; then
            local status=$(jq -r '.deployments[0].status' /tmp/railway_status.json 2>/dev/null || echo "unknown")
            
            case $status in
                "SUCCESS")
                    log_success "Deployment completed successfully!"
                    break
                    ;;
                "FAILED"|"CRASHED")
                    log_error "Deployment failed. Check Railway dashboard for details."
                    railway logs
                    exit 1
                    ;;
                "BUILDING"|"DEPLOYING")
                    log_info "Deployment in progress... (attempt $((attempt + 1))/$max_attempts)"
                    ;;
                *)
                    log_info "Deployment status: $status"
                    ;;
            esac
        else
            log_warning "Unable to check deployment status"
        fi
        
        sleep 10
        ((attempt++))
    done
    
    if [[ $attempt -eq $max_attempts ]]; then
        log_warning "Deployment monitoring timed out. Check Railway dashboard manually."
    fi
}

# Get deployment information
get_deployment_info() {
    log_info "Getting deployment information..."
    
    if railway status --json > /tmp/railway_status.json 2>/dev/null; then
        local url=$(jq -r '.deployments[0].url' /tmp/railway_status.json 2>/dev/null || echo "")
        local project_id=$(jq -r '.project.id' /tmp/railway_status.json 2>/dev/null || echo "")
        
        if [[ -n "$url" && "$url" != "null" ]]; then
            echo ""
            log_success "🌐 Deployment completed successfully!"
            echo -e "${GREEN}📱 Application URL: ${BLUE}$url${NC}"
            echo -e "${GREEN}🔧 Project ID: ${BLUE}$project_id${NC}"
            echo -e "${GREEN}📊 Dashboard: ${BLUE}https://railway.app/project/$project_id${NC}"
            echo ""
            
            # Test the deployment
            log_info "Testing deployment..."
            if curl -f -s "$url/_stcore/health" > /dev/null; then
                log_success "Health check passed - application is running!"
            else
                log_warning "Health check failed - application may still be starting up"
            fi
        else
            log_warning "Unable to retrieve deployment URL"
        fi
    else
        log_error "Unable to get deployment status"
    fi
}

# Cleanup function
cleanup() {
    rm -f /tmp/railway_status.json
}

# Main deployment flow
main() {
    trap cleanup EXIT
    
    echo "Starting Railway deployment process..."
    echo ""
    
    check_prerequisites
    verify_deployment_files
    check_dependencies
    test_docker_build
    deploy_to_railway
    monitor_deployment
    get_deployment_info
    
    echo ""
    log_success "🎉 Railway deployment process completed!"
    echo ""
    echo "Next steps:"
    echo "1. 🔑 Add your API keys in Railway dashboard > Variables"
    echo "2. 📊 Monitor application performance in Railway dashboard"
    echo "3. 🔧 Configure custom domain (optional)"
    echo "4. 📈 Set up monitoring and alerts"
    echo ""
    echo "For support:"
    echo "• Railway docs: https://docs.railway.app"
    echo "• Railway Discord: https://discord.gg/railway"
    echo ""
}

# Run main function
main "$@" 
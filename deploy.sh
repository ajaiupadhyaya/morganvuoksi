#!/bin/bash

# MorganVuoksi Elite Terminal - Production Deployment Script
# MISSION CRITICAL: Bloomberg-grade deployment automation
# ZERO DOWNTIME DEPLOYMENT

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${ENVIRONMENT:-production}
BACKUP_RETENTION_DAYS=${BACKUP_RETENTION_DAYS:-30}
HEALTH_CHECK_TIMEOUT=${HEALTH_CHECK_TIMEOUT:-300}
DOCKER_COMPOSE_FILE="docker-compose.production.yml"

# Print colored output
print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

print_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ❌ $1${NC}"
}

# Help function
show_help() {
    cat << EOF
MorganVuoksi Elite Terminal - Production Deployment

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    deploy          Full production deployment
    start           Start all services
    stop            Stop all services
    restart         Restart all services
    backup          Create system backup
    restore         Restore from backup
    update          Update services (zero downtime)
    status          Show system status
    logs            Show service logs
    health          Run health checks
    cleanup         Clean up old resources
    monitor         Start monitoring dashboard
    help            Show this help message

Options:
    --environment   Environment (production/staging) [default: production]
    --backup-dir    Backup directory [default: ./backups]
    --no-build      Skip building Docker images
    --force         Force operation without prompts
    --verbose       Verbose output

Examples:
    $0 deploy                           # Full production deployment
    $0 start --environment staging      # Start staging environment
    $0 backup --backup-dir /mnt/backup  # Create backup in custom directory
    $0 update --no-build               # Update without rebuilding images
    $0 logs api                        # Show API service logs
    $0 health                          # Run comprehensive health check

EOF
}

# Pre-deployment checks
run_pre_deployment_checks() {
    print_status "Running pre-deployment checks..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check environment file
    if [ ! -f .env ]; then
        print_warning ".env file not found, creating from template..."
        if [ -f .env.template ]; then
            cp .env.template .env
            print_warning "Please configure .env file and run deployment again"
            exit 1
        else
            print_error ".env.template not found"
            exit 1
        fi
    fi
    
    # Validate required environment variables
    required_vars=(
        "DB_PASSWORD"
        "REDIS_PASSWORD"
        "GRAFANA_PASSWORD"
        "API_KEY_ALPHA_VANTAGE"
    )
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ] && [ -z "$(grep "^$var=" .env)" ]; then
            print_error "Required environment variable $var is not set"
            exit 1
        fi
    done
    
    # Check system resources
    print_status "Checking system resources..."
    
    # Check memory (minimum 8GB)
    total_mem=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$total_mem" -lt 8 ]; then
        print_warning "System has less than 8GB RAM. Recommended: 16GB+"
    fi
    
    # Check disk space (minimum 50GB free)
    available_space=$(df / | awk 'NR==2{print $4}')
    available_gb=$((available_space / 1024 / 1024))
    if [ "$available_gb" -lt 50 ]; then
        print_warning "Less than 50GB free disk space available"
    fi
    
    print_success "Pre-deployment checks completed"
}

# Create backup
create_backup() {
    local backup_dir=${1:-./backups}
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_path="$backup_dir/morganvuoksi_backup_$timestamp"
    
    print_status "Creating system backup..."
    
    mkdir -p "$backup_path"
    
    # Backup database
    if docker-compose -f $DOCKER_COMPOSE_FILE ps timescaledb | grep -q "Up"; then
        print_status "Backing up database..."
        docker-compose -f $DOCKER_COMPOSE_FILE exec -T timescaledb pg_dump \
            -U morganvuoksi -d morganvuoksi > "$backup_path/database.sql"
    fi
    
    # Backup configurations
    print_status "Backing up configurations..."
    cp -r config/ "$backup_path/" 2>/dev/null || true
    cp .env "$backup_path/" 2>/dev/null || true
    cp docker-compose*.yml "$backup_path/" 2>/dev/null || true
    
    # Backup models and data
    print_status "Backing up models and data..."
    cp -r models/ "$backup_path/" 2>/dev/null || true
    cp -r data/ "$backup_path/" 2>/dev/null || true
    
    # Create backup manifest
    cat > "$backup_path/manifest.json" << EOF
{
    "timestamp": "$timestamp",
    "version": "1.0.0",
    "environment": "$ENVIRONMENT",
    "components": {
        "database": "$([ -f "$backup_path/database.sql" ] && echo "true" || echo "false")",
        "configurations": "$([ -d "$backup_path/config" ] && echo "true" || echo "false")",
        "models": "$([ -d "$backup_path/models" ] && echo "true" || echo "false")",
        "data": "$([ -d "$backup_path/data" ] && echo "true" || echo "false")"
    }
}
EOF
    
    # Compress backup
    tar -czf "$backup_path.tar.gz" -C "$backup_dir" "morganvuoksi_backup_$timestamp"
    rm -rf "$backup_path"
    
    print_success "Backup created: $backup_path.tar.gz"
    
    # Cleanup old backups
    find "$backup_dir" -name "morganvuoksi_backup_*.tar.gz" -mtime +$BACKUP_RETENTION_DAYS -delete 2>/dev/null || true
}

# Build Docker images
build_images() {
    if [ "$NO_BUILD" = "true" ]; then
        print_status "Skipping image build (--no-build specified)"
        return
    fi
    
    print_status "Building Docker images..."
    
    # Build main application image
    docker build -f Dockerfile.production -t morganvuoksi/terminal:latest .
    
    # Build frontend image
    docker build -f frontend/Dockerfile.production -t morganvuoksi/frontend:latest ./frontend/
    
    print_success "Docker images built successfully"
}

# Deploy services
deploy_services() {
    print_status "Deploying services..."
    
    # Pull external images
    docker-compose -f $DOCKER_COMPOSE_FILE pull
    
    # Start infrastructure services first
    print_status "Starting infrastructure services..."
    docker-compose -f $DOCKER_COMPOSE_FILE up -d \
        timescaledb redis prometheus grafana elasticsearch influxdb
    
    # Wait for infrastructure
    print_status "Waiting for infrastructure services..."
    sleep 30
    
    # Start Ray cluster
    print_status "Starting Ray cluster..."
    docker-compose -f $DOCKER_COMPOSE_FILE up -d ray-head ray-worker
    sleep 15
    
    # Start application services
    print_status "Starting application services..."
    docker-compose -f $DOCKER_COMPOSE_FILE up -d api frontend
    
    # Start monitoring and supporting services
    print_status "Starting monitoring services..."
    docker-compose -f $DOCKER_COMPOSE_FILE up -d \
        nginx kibana logstash jupyter backup healthcheck
    
    print_success "All services deployed"
}

# Health check
run_health_check() {
    print_status "Running comprehensive health check..."
    
    local services=("api" "frontend" "timescaledb" "redis" "ray-head")
    local failed_services=()
    
    for service in "${services[@]}"; do
        print_status "Checking $service..."
        
        # Check if container is running
        if ! docker-compose -f $DOCKER_COMPOSE_FILE ps "$service" | grep -q "Up"; then
            print_error "$service is not running"
            failed_services+=("$service")
            continue
        fi
        
        # Service-specific health checks
        case $service in
            "api")
                if ! curl -f http://localhost:8000/api/v1/health --max-time 10 &>/dev/null; then
                    print_error "API health check failed"
                    failed_services+=("$service")
                fi
                ;;
            "frontend")
                if ! curl -f http://localhost:3000 --max-time 10 &>/dev/null; then
                    print_error "Frontend health check failed"
                    failed_services+=("$service")
                fi
                ;;
            "timescaledb")
                if ! docker-compose -f $DOCKER_COMPOSE_FILE exec -T timescaledb pg_isready -U morganvuoksi &>/dev/null; then
                    print_error "Database health check failed"
                    failed_services+=("$service")
                fi
                ;;
            "redis")
                if ! docker-compose -f $DOCKER_COMPOSE_FILE exec -T redis redis-cli ping &>/dev/null; then
                    print_error "Redis health check failed"
                    failed_services+=("$service")
                fi
                ;;
        esac
    done
    
    if [ ${#failed_services[@]} -eq 0 ]; then
        print_success "All health checks passed"
        return 0
    else
        print_error "Health check failed for: ${failed_services[*]}"
        return 1
    fi
}

# Monitor deployment
monitor_deployment() {
    local timeout=${HEALTH_CHECK_TIMEOUT:-300}
    local elapsed=0
    local interval=10
    
    print_status "Monitoring deployment progress..."
    
    while [ $elapsed -lt $timeout ]; do
        if run_health_check &>/dev/null; then
            print_success "Deployment successful! All services are healthy."
            
            # Display access information
            echo ""
            echo "🎉 MorganVuoksi Elite Terminal is now running!"
            echo ""
            echo "📊 Services:"
            echo "   • Terminal:     http://localhost:3000"
            echo "   • API:          http://localhost:8000"
            echo "   • Docs:         http://localhost:8000/docs"
            echo "   • Grafana:      http://localhost:3001"
            echo "   • Prometheus:   http://localhost:9090"
            echo "   • Ray Dashboard: http://localhost:8265"
            echo "   • Kibana:       http://localhost:5601"
            echo "   • Jupyter:      http://localhost:8888"
            echo ""
            echo "🔐 Default credentials:"
            echo "   • Grafana:      admin / ${GRAFANA_PASSWORD:-admin}"
            echo "   • Jupyter:      Token in logs or set JUPYTER_TOKEN"
            echo ""
            
            return 0
        fi
        
        elapsed=$((elapsed + interval))
        print_status "Waiting for services to be ready... (${elapsed}s/${timeout}s)"
        sleep $interval
    done
    
    print_error "Deployment timeout after ${timeout}s"
    print_status "Checking individual service status..."
    run_health_check
    return 1
}

# Show system status
show_status() {
    print_status "System Status:"
    echo ""
    
    # Service status
    docker-compose -f $DOCKER_COMPOSE_FILE ps
    
    echo ""
    print_status "Resource Usage:"
    
    # CPU and Memory usage
    echo "CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
    echo "Memory Usage: $(free | grep Mem | awk '{printf("%.1f%%\n", $3/$2 * 100.0)}')"
    echo "Disk Usage: $(df / | awk 'NR==2{printf("%.1f%%\n", $5)}')"
    
    echo ""
    print_status "Docker Resources:"
    docker system df
}

# Show logs
show_logs() {
    local service=${1:-}
    local lines=${2:-100}
    
    if [ -z "$service" ]; then
        print_status "Available services:"
        docker-compose -f $DOCKER_COMPOSE_FILE config --services
        return
    fi
    
    print_status "Showing logs for $service (last $lines lines):"
    docker-compose -f $DOCKER_COMPOSE_FILE logs --tail=$lines -f "$service"
}

# Update services with zero downtime
update_services() {
    print_status "Starting zero-downtime update..."
    
    # Create backup before update
    create_backup
    
    # Pull latest images
    if [ "$NO_BUILD" != "true" ]; then
        build_images
    fi
    
    # Rolling update strategy
    local services=("api" "frontend")
    
    for service in "${services[@]}"; do
        print_status "Updating $service..."
        
        # Scale up new instance
        docker-compose -f $DOCKER_COMPOSE_FILE up -d --scale "$service"=2 "$service"
        sleep 30
        
        # Health check new instance
        if run_health_check &>/dev/null; then
            # Scale down old instance
            docker-compose -f $DOCKER_COMPOSE_FILE up -d --scale "$service"=1 "$service"
            print_success "$service updated successfully"
        else
            print_error "Update failed for $service, rolling back..."
            docker-compose -f $DOCKER_COMPOSE_FILE up -d --scale "$service"=1 "$service"
            exit 1
        fi
    done
    
    print_success "Zero-downtime update completed"
}

# Cleanup old resources
cleanup_resources() {
    print_status "Cleaning up old resources..."
    
    # Remove unused Docker images
    docker image prune -f
    
    # Remove unused Docker volumes
    docker volume prune -f
    
    # Remove unused Docker networks
    docker network prune -f
    
    # Clean up old logs
    find logs/ -name "*.log" -mtime +7 -delete 2>/dev/null || true
    
    print_success "Cleanup completed"
}

# Start monitoring dashboard
start_monitoring() {
    print_status "Starting monitoring dashboard..."
    
    # Ensure monitoring services are running
    docker-compose -f $DOCKER_COMPOSE_FILE up -d grafana prometheus
    
    # Wait for services
    sleep 10
    
    # Open monitoring URLs
    echo ""
    echo "📊 Monitoring Dashboard URLs:"
    echo "   • Grafana:      http://localhost:3001"
    echo "   • Prometheus:   http://localhost:9090"
    echo "   • Ray Dashboard: http://localhost:8265"
    echo ""
}

# Main deployment function
main_deploy() {
    print_status "🚀 Starting MorganVuoksi Elite Terminal Deployment"
    print_status "Environment: $ENVIRONMENT"
    print_status "Timestamp: $(date)"
    
    # Run pre-deployment checks
    run_pre_deployment_checks
    
    # Create backup
    create_backup
    
    # Build images
    build_images
    
    # Deploy services
    deploy_services
    
    # Monitor deployment
    monitor_deployment
    
    print_success "🎉 Deployment completed successfully!"
}

# Parse command line arguments
COMMAND=""
NO_BUILD=false
FORCE=false
VERBOSE=false
BACKUP_DIR="./backups"

while [[ $# -gt 0 ]]; do
    case $1 in
        deploy|start|stop|restart|backup|restore|update|status|logs|health|cleanup|monitor|help)
            COMMAND="$1"
            shift
            ;;
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --backup-dir)
            BACKUP_DIR="$2"
            shift 2
            ;;
        --no-build)
            NO_BUILD=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            set -x
            shift
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

# Load environment variables
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Execute command
case $COMMAND in
    deploy)
        main_deploy
        ;;
    start)
        print_status "Starting services..."
        docker-compose -f $DOCKER_COMPOSE_FILE up -d
        monitor_deployment
        ;;
    stop)
        print_status "Stopping services..."
        docker-compose -f $DOCKER_COMPOSE_FILE down
        print_success "Services stopped"
        ;;
    restart)
        print_status "Restarting services..."
        docker-compose -f $DOCKER_COMPOSE_FILE restart
        monitor_deployment
        ;;
    backup)
        create_backup "$BACKUP_DIR"
        ;;
    update)
        update_services
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs "${ARGS[0]}" "${ARGS[1]}"
        ;;
    health)
        run_health_check
        ;;
    cleanup)
        cleanup_resources
        ;;
    monitor)
        start_monitoring
        ;;
    help|"")
        show_help
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac 
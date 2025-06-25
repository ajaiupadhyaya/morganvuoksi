#!/bin/bash

# MorganVuoksi Elite Terminal - Quick Deployment Script
# Deploys the Bloomberg-grade terminal in minutes

set -e  # Exit on any error

echo "🚀 MorganVuoksi Elite Terminal - Quick Deploy"
echo "=============================================="
echo ""

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function for colored output
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

# Check system requirements
check_requirements() {
    print_status "Checking system requirements..."
    
    # Check Python
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
        print_success "Python found: $PYTHON_VERSION"
    elif command -v python &> /dev/null; then
        PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
        print_success "Python found: $PYTHON_VERSION"
    else
        print_error "Python not found. Please install Python 3.9+"
        exit 1
    fi
    
    # Check Node.js (for advanced UI)
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        print_success "Node.js found: $NODE_VERSION"
        HAS_NODE=true
    else
        print_warning "Node.js not found. Advanced UI will not be available."
        HAS_NODE=false
    fi
    
    # Check Docker (for production deployment)
    if command -v docker &> /dev/null; then
        DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | tr -d ',')
        print_success "Docker found: $DOCKER_VERSION"
        HAS_DOCKER=true
    else
        print_warning "Docker not found. Production deployment will not be available."
        HAS_DOCKER=false
    fi
    
    echo ""
}

# Menu for deployment options
show_menu() {
    echo "📋 Choose your deployment method:"
    echo ""
    echo "1) 🌐 Web-Optimized Version (Streamlit)"
    echo "   └─ Ready for Streamlit Cloud deployment"
    echo "   └─ Perfect for demos and testing"
    echo ""
    
    if [ "$HAS_NODE" = true ]; then
        echo "2) 🎨 Professional UI Version (Bloomberg Clone)"
        echo "   └─ Advanced Bloomberg Terminal interface"
        echo "   └─ 16-column professional layout"
        echo ""
    fi
    
    if [ "$HAS_DOCKER" = true ]; then
        echo "3) 🐳 Production Docker Deployment"
        echo "   └─ Full microservices architecture"
        echo "   └─ Enterprise-grade setup"
        echo ""
    fi
    
    echo "4) 📖 Show deployment URLs and instructions"
    echo "5) ❌ Exit"
    echo ""
}

# Deploy Streamlit version
deploy_streamlit() {
    print_status "Deploying Web-Optimized Terminal (Streamlit)..."
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    print_status "Activating virtual environment..."
    source venv/bin/activate
    
    # Install dependencies
    print_status "Installing dependencies..."
    pip install --upgrade pip
    
    if [ -f "requirements-web.txt" ]; then
        pip install -r requirements-web.txt
    elif [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        print_error "Requirements file not found!"
        exit 1
    fi
    
    print_success "Dependencies installed successfully!"
    echo ""
    
    # Start Streamlit app
    print_status "Starting Bloomberg Terminal..."
    echo ""
    print_success "🎉 Terminal starting at: http://localhost:8501"
    print_success "📊 Bloomberg-grade interface loading..."
    echo ""
    print_status "Press Ctrl+C to stop the terminal"
    echo ""
    
    streamlit run streamlit_app.py --server.headless true
}

# Deploy professional UI version
deploy_professional_ui() {
    print_status "Deploying Professional UI Version (Bloomberg Clone)..."
    
    if [ ! -d "provided" ]; then
        print_error "Professional UI folder 'provided' not found!"
        exit 1
    fi
    
    cd provided
    
    # Install Node.js dependencies
    print_status "Installing Node.js dependencies..."
    npm install
    
    print_success "Dependencies installed successfully!"
    echo ""
    
    # Start development server
    print_status "Starting Professional Bloomberg Terminal..."
    echo ""
    print_success "🎉 Terminal starting at: http://localhost:5173"
    print_success "💎 Professional Bloomberg interface loading..."
    print_success "⚡ Advanced features: Command palette (Ctrl+K), Function keys (F8, F9)"
    echo ""
    print_status "Press Ctrl+C to stop the terminal"
    echo ""
    
    npm run dev
}

# Deploy Docker version
deploy_docker() {
    print_status "Deploying Production Docker Setup..."
    
    # Check if .env file exists
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            print_status "Creating .env file from example..."
            cp .env.example .env
            print_warning "Please edit .env file with your API keys for full functionality"
        else
            print_status "Creating basic .env file..."
            cat > .env << EOF
# MorganVuoksi Terminal Environment Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Optional: Add your API keys for enhanced functionality
# ALPHA_VANTAGE_API_KEY=your_key_here
# POLYGON_API_KEY=your_key_here
# OPENAI_API_KEY=your_key_here
EOF
        fi
    fi
    
    # Deploy with Docker Compose
    print_status "Starting Docker containers..."
    
    if [ -f "docker-compose.production.yml" ]; then
        docker-compose -f docker-compose.production.yml up -d
        
        print_success "Production environment deployed successfully!"
        echo ""
        print_success "🎉 Services available at:"
        print_success "   📊 Bloomberg Terminal: http://localhost:3000"
        print_success "   🔧 API Gateway: http://localhost:8000"
        print_success "   📈 Monitoring: http://localhost:3001"
        print_success "   🤖 ML Cluster: http://localhost:8265"
        print_success "   📓 Jupyter: http://localhost:8888"
        
    else
        docker-compose up -d
        
        print_success "Development environment deployed successfully!"
        echo ""
        print_success "🎉 Bloomberg Terminal: http://localhost:8501"
    fi
    
    echo ""
    print_status "To stop: docker-compose down"
}

# Show deployment information
show_deployment_info() {
    echo ""
    print_success "🌐 DEPLOYMENT RESOURCES"
    echo "========================"
    echo ""
    
    echo "📋 Quick Deploy Options:"
    echo "• Streamlit Cloud: https://share.streamlit.io"
    echo "• Railway: https://railway.app"
    echo "• Render: https://render.com"
    echo "• Heroku: https://heroku.com"
    echo ""
    
    echo "📖 Documentation:"
    echo "• README.md - Complete deployment guide"
    echo "• DEPLOYMENT_CONFIRMATION.md - Verification details"
    echo "• PRODUCTION_DEPLOYMENT_COMPLETE.md - Production setup"
    echo ""
    
    echo "🎯 Project Structure:"
    echo "• streamlit_app.py - Web-optimized terminal"
    echo "• provided/ - Professional Bloomberg UI"
    echo "• frontend/ - Next.js production frontend"
    echo "• docker-compose.production.yml - Full production setup"
    echo ""
    
    echo "🔑 API Keys (Optional but recommended):"
    echo "• Alpha Vantage: https://www.alphavantage.co/support/#api-key"
    echo "• Polygon.io: https://polygon.io/pricing"
    echo "• OpenAI: https://platform.openai.com/api-keys"
    echo ""
    
    print_success "Ready to deploy your Bloomberg Terminal! 🚀"
}

# Main script execution
main() {
    check_requirements
    
    while true; do
        show_menu
        read -p "Select option [1-5]: " choice
        echo ""
        
        case $choice in
            1)
                deploy_streamlit
                break
                ;;
            2)
                if [ "$HAS_NODE" = true ]; then
                    deploy_professional_ui
                    break
                else
                    print_error "Node.js required for Professional UI. Please install Node.js."
                fi
                ;;
            3)
                if [ "$HAS_DOCKER" = true ]; then
                    deploy_docker
                    break
                else
                    print_error "Docker required for production deployment. Please install Docker."
                fi
                ;;
            4)
                show_deployment_info
                ;;
            5)
                print_status "Thanks for using MorganVuoksi Elite Terminal!"
                exit 0
                ;;
            *)
                print_warning "Invalid option. Please select 1-5."
                ;;
        esac
        echo ""
    done
}

# Run main function
main
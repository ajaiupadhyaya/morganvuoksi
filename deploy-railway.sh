#!/bin/bash

# MorganVuoksi Terminal - Railway.app Deployment Script
echo "🚀 Deploying MorganVuoksi Terminal to Railway.app..."

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Installing..."
    npm install -g @railway/cli
fi

# Login to Railway (if not already logged in)
echo "🔐 Logging into Railway..."
railway login

# Link to existing project or create new one
echo "🔗 Linking to Railway project..."
if [ -z "$RAILWAY_PROJECT_ID" ]; then
    echo "📝 Creating new Railway project..."
    railway init
else
    echo "🔗 Linking to existing project: $RAILWAY_PROJECT_ID"
    railway link $RAILWAY_PROJECT_ID
fi

# Set environment variables
echo "⚙️ Setting up environment variables..."
railway variables set STREAMLIT_SERVER_PORT=\$PORT
railway variables set STREAMLIT_SERVER_ADDRESS=0.0.0.0
railway variables set STREAMLIT_SERVER_HEADLESS=true
railway variables set STREAMLIT_SERVER_ENABLE_CORS=false
railway variables set STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Deploy the application
echo "🚀 Deploying application..."
railway up

# Get the deployment URL
echo "🌐 Getting deployment URL..."
DEPLOY_URL=$(railway status --json | jq -r '.deployments[0].url')
echo "✅ Deployment complete!"
echo "🌐 Your app is available at: $DEPLOY_URL"

# Optional: Set up custom domain
echo "💡 To set up a custom domain, run:"
echo "   railway domain add your-domain.com" 
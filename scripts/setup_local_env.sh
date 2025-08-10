#!/bin/bash
# Setup local development environment

echo "🚀 Setting up local development environment for Label Computation System"

# Check if .env exists
if [ -f .env ]; then
    echo "⚠️  .env file already exists. Backing up to .env.backup"
    cp .env .env.backup
fi

# Create .env.example if it doesn't exist
cat > .env.example << 'EOF'
# ClickHouse Configuration
CLICKHOUSE_HOST=your-endpoint.clickhouse.cloud
CLICKHOUSE_PORT=9440
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=your-password
CLICKHOUSE_DATABASE=quantx
CLICKHOUSE_SECURE=true

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Label Computation Settings
BATCH_CHUNK_SIZE=10000
PARALLEL_WORKERS=8
CACHE_TTL_SECONDS=3600

# Monitoring
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090

# Optional: Firestore
GCP_PROJECT_ID=
GOOGLE_APPLICATION_CREDENTIALS=

# Environment
DEBUG=true
ENVIRONMENT=development
EOF

echo "✅ Created .env.example"

# Try to get secrets from GitHub (requires gh CLI and proper auth)
echo ""
echo "📦 Attempting to fetch secrets from GitHub..."
echo "Note: This requires 'gh' CLI to be authenticated with proper permissions"

# Check if gh is authenticated
if gh auth status &>/dev/null; then
    echo "✅ GitHub CLI authenticated"
    
    # Create .env with actual values
    echo "# Auto-generated from GitHub Secrets" > .env
    echo "# Generated on $(date)" >> .env
    echo "" >> .env
    
    # Note: We can't actually retrieve secret VALUES from GitHub (they're write-only)
    # But we can set up the structure
    cat >> .env << 'EOF'
# ClickHouse Configuration
# IMPORTANT: Update these with actual values!
# You can get them from GitHub Actions logs or ask the repository admin
CLICKHOUSE_HOST=${CLICKHOUSE_HOST}
CLICKHOUSE_PORT=9440
CLICKHOUSE_USER=${CLICKHOUSE_USER}
CLICKHOUSE_PASSWORD=${CLICKHOUSE_PASSWORD}
CLICKHOUSE_DATABASE=quantx
CLICKHOUSE_SECURE=true

# Redis Configuration (local development)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Label Computation Settings
BATCH_CHUNK_SIZE=10000
PARALLEL_WORKERS=8
CACHE_TTL_SECONDS=3600

# Monitoring
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090

# Environment
DEBUG=true
ENVIRONMENT=development
EOF
    
    echo "⚠️  GitHub Secrets are write-only and cannot be retrieved directly"
    echo "📝 Please update .env with actual values from:"
    echo "   1. GitHub Actions logs (after running test-connection workflow)"
    echo "   2. Repository admin/owner"
    echo "   3. Your ClickHouse Cloud dashboard"
    
else
    echo "⚠️  GitHub CLI not authenticated. Creating template .env"
    cp .env.example .env
fi

# Install Python dependencies
echo ""
echo "📦 Installing Python dependencies..."
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
    echo "✅ Python dependencies installed"
else
    echo "❌ requirements.txt not found"
fi

# Check Redis
echo ""
echo "🔍 Checking Redis..."
if command -v redis-cli &> /dev/null; then
    if redis-cli ping &> /dev/null; then
        echo "✅ Redis is running"
    else
        echo "⚠️  Redis is not running. Start it with: redis-server"
    fi
else
    echo "⚠️  Redis not installed. Install with: brew install redis (Mac) or apt-get install redis (Linux)"
fi

# Check Docker
echo ""
echo "🔍 Checking Docker..."
if command -v docker &> /dev/null; then
    if docker ps &> /dev/null; then
        echo "✅ Docker is running"
    else
        echo "⚠️  Docker is not running. Please start Docker Desktop"
    fi
else
    echo "⚠️  Docker not installed. Download from https://docker.com"
fi

echo ""
echo "📋 Next steps:"
echo "1. Update .env with your actual ClickHouse credentials"
echo "2. Run: python scripts/create_tables.py"
echo "3. Start Redis: redis-server"
echo "4. Run tests: python scripts/run_tests.py"
echo "5. Start server: python main.py"
echo ""
echo "Or use Docker:"
echo "  docker-compose up -d"
echo ""
echo "✅ Setup script complete!"
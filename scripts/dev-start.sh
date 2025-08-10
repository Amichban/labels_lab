#!/bin/bash
# Development startup script for Label Computation System

set -e

echo "🚀 Starting Label Computation System in Development Mode"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}📦 Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies if requirements changed
if [ requirements.txt -nt venv/pyvenv.cfg ]; then
    echo -e "${YELLOW}📚 Installing dependencies...${NC}"
    pip install --upgrade pip
    pip install -r requirements.txt
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚙️  Creating .env from template...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}📝 Please edit .env file with your configuration${NC}"
fi

# Create necessary directories
mkdir -p logs temp

# Check if dependencies are running (optional)
echo -e "${YELLOW}🔍 Checking dependencies...${NC}"

# Check Redis (optional - will work without for development)
if command -v redis-cli &> /dev/null; then
    if redis-cli ping &> /dev/null; then
        echo -e "${GREEN}✅ Redis is running${NC}"
    else
        echo -e "${YELLOW}⚠️  Redis is not running (cache will be disabled)${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Redis CLI not found (install redis-tools for health checks)${NC}"
fi

# Start the development server
echo -e "${GREEN}🎯 Starting FastAPI development server...${NC}"
echo -e "${GREEN}📖 API Documentation: http://localhost:8000/docs${NC}"
echo -e "${GREEN}🏥 Health Check: http://localhost:8000/v1/health${NC}"
echo -e "${GREEN}📊 Metrics: http://localhost:8000/v1/metrics${NC}"
echo ""

# Set development environment
export DEBUG=true
export ENVIRONMENT=development

# Start with auto-reload
python -m uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --reload-dir src \
    --reload-dir config \
    --log-level info \
    --access-log
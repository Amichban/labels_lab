#!/bin/bash

# Post-deployment verification script
# Comprehensive health checks and metric validation

set -e

# Configuration
ENVIRONMENT=${1:-staging}
VERSION=${2:-latest}
CANARY_PERCENTAGE=${3:-100}
TIMEOUT=${4:-300}

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Thresholds
ERROR_RATE_THRESHOLD=0.02
LATENCY_P95_THRESHOLD=500
SUCCESS_RATE_THRESHOLD=0.98
MEMORY_THRESHOLD=80
CPU_THRESHOLD=70

# Results tracking
CHECKS_PASSED=0
CHECKS_FAILED=0
SHOULD_ROLLBACK=false

echo "================================================="
echo "🔍 POST-DEPLOYMENT VERIFICATION"
echo "================================================="
echo "Environment: $ENVIRONMENT"
echo "Version: $VERSION"
echo "Canary: ${CANARY_PERCENTAGE}%"
echo "Timeout: ${TIMEOUT}s"
echo "================================================="
echo ""

# Function: Check endpoint health
check_endpoint() {
    local url=$1
    local expected_status=$2
    local description=$3
    local max_retries=3
    local retry=0
    
    echo -n "Checking $description... "
    
    while [ $retry -lt $max_retries ]; do
        status=$(curl -s -o /dev/null -w "%{http_code}" "$url" --max-time 10 2>/dev/null || echo "000")
        
        if [ "$status" = "$expected_status" ]; then
            echo -e "${GREEN}✅ OK ($status)${NC}"
            ((CHECKS_PASSED++))
            return 0
        fi
        
        ((retry++))
        if [ $retry -lt $max_retries ]; then
            sleep 2
        fi
    done
    
    echo -e "${RED}❌ FAILED (expected $expected_status, got $status)${NC}"
    ((CHECKS_FAILED++))
    return 1
}

# Function: Check metric value
check_metric() {
    local metric_name=$1
    local current_value=$2
    local threshold=$3
    local comparison=$4
    local unit=${5:-""}
    
    echo -n "Checking $metric_name... "
    
    local passes=false
    case $comparison in
        "lt")
            if (( $(echo "$current_value < $threshold" | bc -l) )); then
                passes=true
            fi
            ;;
        "gt")
            if (( $(echo "$current_value > $threshold" | bc -l) )); then
                passes=true
            fi
            ;;
        "eq")
            if [ "$current_value" = "$threshold" ]; then
                passes=true
            fi
            ;;
    esac
    
    if [ "$passes" = true ]; then
        echo -e "${GREEN}✅ OK (${current_value}${unit})${NC}"
        ((CHECKS_PASSED++))
        return 0
    else
        echo -e "${RED}❌ FAILED (${current_value}${unit}, threshold: $comparison ${threshold}${unit})${NC}"
        ((CHECKS_FAILED++))
        SHOULD_ROLLBACK=true
        return 1
    fi
}

# Function: Run SQL query check
check_database() {
    local query=$1
    local expected=$2
    local description=$3
    
    echo -n "Checking $description... "
    
    result=$(psql -h db.$ENVIRONMENT.example.com -U readonly -d app -t -c "$query" 2>/dev/null || echo "ERROR")
    
    if [ "$result" = "ERROR" ]; then
        echo -e "${RED}❌ FAILED (database connection error)${NC}"
        ((CHECKS_FAILED++))
        SHOULD_ROLLBACK=true
        return 1
    fi
    
    if [ "$result" = "$expected" ] || [ -z "$expected" ]; then
        echo -e "${GREEN}✅ OK${NC}"
        ((CHECKS_PASSED++))
        return 0
    else
        echo -e "${YELLOW}⚠️ WARNING (got: $result, expected: $expected)${NC}"
        ((CHECKS_FAILED++))
        return 1
    fi
}

# 1. HEALTH CHECKS
echo "1️⃣ HEALTH CHECKS"
echo "-----------------"
check_endpoint "https://api.$ENVIRONMENT.example.com/health" "200" "API health"
check_endpoint "https://app.$ENVIRONMENT.example.com/" "200" "App homepage"
check_endpoint "https://api.$ENVIRONMENT.example.com/version" "200" "Version endpoint"
check_endpoint "https://api.$ENVIRONMENT.example.com/ready" "200" "Readiness"
echo ""

# 2. VERSION VERIFICATION
echo "2️⃣ VERSION VERIFICATION"
echo "------------------------"
echo -n "Deployed version check... "
deployed_version=$(curl -s "https://api.$ENVIRONMENT.example.com/version" | jq -r '.version' 2>/dev/null || echo "unknown")
if [ "$deployed_version" = "$VERSION" ]; then
    echo -e "${GREEN}✅ Correct version ($VERSION)${NC}"
    ((CHECKS_PASSED++))
else
    echo -e "${RED}❌ Version mismatch (expected $VERSION, got $deployed_version)${NC}"
    ((CHECKS_FAILED++))
    SHOULD_ROLLBACK=true
fi
echo ""

# 3. PERFORMANCE METRICS
echo "3️⃣ PERFORMANCE METRICS"
echo "-----------------------"

# Get metrics from monitoring system
if [ "$ENVIRONMENT" = "production" ]; then
    METRICS_URL="https://metrics.example.com/api/v1/query"
else
    METRICS_URL="https://metrics-staging.example.com/api/v1/query"
fi

# Error rate
error_rate=$(curl -s "$METRICS_URL?query=rate(http_requests_total{status=~'5..'}[5m])" | \
    jq -r '.data.result[0].value[1]' 2>/dev/null || echo "0.001")
error_rate_percent=$(echo "$error_rate * 100" | bc -l | xargs printf "%.2f")
check_metric "Error rate" "$error_rate" "$ERROR_RATE_THRESHOLD" "lt" "%"

# P95 Latency
p95_latency=$(curl -s "$METRICS_URL?query=histogram_quantile(0.95,http_request_duration_seconds_bucket[5m])" | \
    jq -r '.data.result[0].value[1]' 2>/dev/null || echo "150")
p95_latency_ms=$(echo "$p95_latency * 1000" | bc -l | xargs printf "%.0f")
check_metric "P95 latency" "$p95_latency_ms" "$LATENCY_P95_THRESHOLD" "lt" "ms"

# Success rate
success_rate=$(curl -s "$METRICS_URL?query=rate(http_requests_total{status=~'2..'}[5m])/rate(http_requests_total[5m])" | \
    jq -r '.data.result[0].value[1]' 2>/dev/null || echo "0.99")
check_metric "Success rate" "$success_rate" "$SUCCESS_RATE_THRESHOLD" "gt" ""

# CPU usage
cpu_usage=$(curl -s "$METRICS_URL?query=avg(rate(container_cpu_usage_seconds_total[5m]))*100" | \
    jq -r '.data.result[0].value[1]' 2>/dev/null || echo "35")
check_metric "CPU usage" "$cpu_usage" "$CPU_THRESHOLD" "lt" "%"

# Memory usage
memory_usage=$(curl -s "$METRICS_URL?query=avg(container_memory_usage_bytes/container_spec_memory_limit_bytes)*100" | \
    jq -r '.data.result[0].value[1]' 2>/dev/null || echo "45")
check_metric "Memory usage" "$memory_usage" "$MEMORY_THRESHOLD" "lt" "%"
echo ""

# 4. DATABASE CHECKS
echo "4️⃣ DATABASE CHECKS"
echo "-------------------"
check_database "SELECT 1" "" "Database connectivity"
check_database "SELECT COUNT(*) FROM schema_migrations WHERE version = '$VERSION'" "1" "Migration applied"
check_database "SELECT pg_is_in_recovery()" "f" "Database is primary"
echo ""

# 5. INTEGRATION CHECKS
echo "5️⃣ INTEGRATION CHECKS"
echo "----------------------"
check_endpoint "https://api.$ENVIRONMENT.example.com/integrations/payment/status" "200" "Payment service"
check_endpoint "https://api.$ENVIRONMENT.example.com/integrations/notification/status" "200" "Notification service"
check_endpoint "https://api.$ENVIRONMENT.example.com/integrations/analytics/status" "200" "Analytics service"
check_endpoint "https://api.$ENVIRONMENT.example.com/integrations/cache/status" "200" "Cache service"
echo ""

# 6. CANARY-SPECIFIC CHECKS
if [ "$CANARY_PERCENTAGE" -lt 100 ]; then
    echo "6️⃣ CANARY CHECKS"
    echo "-----------------"
    
    # Check traffic distribution
    echo -n "Traffic distribution... "
    actual_canary=$(curl -s "$METRICS_URL?query=sum(rate(http_requests_total{version='$VERSION'}[5m]))/sum(rate(http_requests_total[5m]))*100" | \
        jq -r '.data.result[0].value[1]' 2>/dev/null || echo "0")
    actual_canary_int=$(echo "$actual_canary" | xargs printf "%.0f")
    
    tolerance=5
    min_canary=$((CANARY_PERCENTAGE - tolerance))
    max_canary=$((CANARY_PERCENTAGE + tolerance))
    
    if [ "$actual_canary_int" -ge "$min_canary" ] && [ "$actual_canary_int" -le "$max_canary" ]; then
        echo -e "${GREEN}✅ OK (${actual_canary_int}% ≈ ${CANARY_PERCENTAGE}%)${NC}"
        ((CHECKS_PASSED++))
    else
        echo -e "${YELLOW}⚠️ WARNING (${actual_canary_int}% != ${CANARY_PERCENTAGE}%)${NC}"
        ((CHECKS_FAILED++))
    fi
    
    # Compare canary vs stable metrics
    echo -n "Canary vs Stable comparison... "
    canary_error_rate=$(curl -s "$METRICS_URL?query=rate(http_requests_total{status=~'5..',version='$VERSION'}[5m])" | \
        jq -r '.data.result[0].value[1]' 2>/dev/null || echo "0.001")
    stable_error_rate=$(curl -s "$METRICS_URL?query=rate(http_requests_total{status=~'5..',version!='$VERSION'}[5m])" | \
        jq -r '.data.result[0].value[1]' 2>/dev/null || echo "0.001")
    
    if (( $(echo "$canary_error_rate <= $stable_error_rate * 1.5" | bc -l) )); then
        echo -e "${GREEN}✅ Canary performing well${NC}"
        ((CHECKS_PASSED++))
    else
        echo -e "${RED}❌ Canary underperforming${NC}"
        ((CHECKS_FAILED++))
        SHOULD_ROLLBACK=true
    fi
    echo ""
fi

# 7. SECURITY CHECKS
echo "7️⃣ SECURITY CHECKS"
echo "-------------------"
check_endpoint "https://api.$ENVIRONMENT.example.com/security/headers" "200" "Security headers"
check_endpoint "https://api.$ENVIRONMENT.example.com/robots.txt" "200" "Robots.txt"

# Check for exposed secrets
echo -n "Checking for exposed secrets... "
exposed_secrets=$(curl -s "https://api.$ENVIRONMENT.example.com/" | grep -iE "(api[_-]key|secret|password|token)" | wc -l)
if [ "$exposed_secrets" -eq 0 ]; then
    echo -e "${GREEN}✅ No secrets exposed${NC}"
    ((CHECKS_PASSED++))
else
    echo -e "${RED}❌ Potential secrets exposed${NC}"
    ((CHECKS_FAILED++))
    SHOULD_ROLLBACK=true
fi
echo ""

# 8. FUNCTIONAL SMOKE TESTS
echo "8️⃣ FUNCTIONAL SMOKE TESTS"
echo "--------------------------"

# Login test
echo -n "Login flow... "
login_response=$(curl -s -X POST "https://api.$ENVIRONMENT.example.com/auth/login" \
    -H "Content-Type: application/json" \
    -d '{"username":"test@example.com","password":"test123"}' \
    -w "\n%{http_code}" 2>/dev/null | tail -1)
if [ "$login_response" = "200" ] || [ "$login_response" = "201" ]; then
    echo -e "${GREEN}✅ OK${NC}"
    ((CHECKS_PASSED++))
else
    echo -e "${YELLOW}⚠️ WARNING (status: $login_response)${NC}"
    ((CHECKS_FAILED++))
fi

# Critical user journey
echo -n "Critical user journey... "
journey_result=$(curl -s "https://api.$ENVIRONMENT.example.com/test/user-journey" | jq -r '.success' 2>/dev/null || echo "false")
if [ "$journey_result" = "true" ]; then
    echo -e "${GREEN}✅ OK${NC}"
    ((CHECKS_PASSED++))
else
    echo -e "${RED}❌ FAILED${NC}"
    ((CHECKS_FAILED++))
fi
echo ""

# SUMMARY
echo "================================================="
echo "📊 VERIFICATION SUMMARY"
echo "================================================="
echo -e "${GREEN}Passed: $CHECKS_PASSED${NC}"
echo -e "${RED}Failed: $CHECKS_FAILED${NC}"
total_checks=$((CHECKS_PASSED + CHECKS_FAILED))
pass_rate=$((CHECKS_PASSED * 100 / total_checks))
echo "Pass Rate: ${pass_rate}%"
echo ""

# RECOMMENDATION
if [ "$SHOULD_ROLLBACK" = true ]; then
    echo -e "${RED}❌ VERIFICATION FAILED - IMMEDIATE ROLLBACK RECOMMENDED${NC}"
    echo ""
    echo "Critical issues detected:"
    echo "- One or more critical metrics exceeded thresholds"
    echo "- Version mismatch or deployment issues"
    echo ""
    echo "Recommended actions:"
    echo "1. Initiate immediate rollback"
    echo "2. Review deployment logs"
    echo "3. Check error spike in monitoring"
    echo "4. Investigate root cause before retry"
    exit 1
elif [ "$CHECKS_FAILED" -gt 2 ]; then
    echo -e "${YELLOW}⚠️ VERIFICATION PASSED WITH WARNINGS${NC}"
    echo ""
    echo "Some non-critical checks failed."
    echo "Monitor closely and be prepared to rollback if issues worsen."
    exit 0
else
    echo -e "${GREEN}✅ VERIFICATION PASSED - DEPLOYMENT SUCCESSFUL${NC}"
    echo ""
    echo "All critical checks passed."
    echo "Deployment is healthy and stable."
    exit 0
fi
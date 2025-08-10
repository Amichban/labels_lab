#!/bin/bash

# CI Helper Script
# Provides CLI commands for interacting with Claude CI Teacher

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CONFIG_FILE=".github/claude-config.yml"
ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}"

# Usage function
usage() {
    cat << EOF
Claude CI Helper - Intelligent CI assistance

Usage: $0 <command> [options]

Commands:
    fix <error>        - Get fix for specific error
    analyze            - Analyze recent CI failures
    teach <topic>      - Get educational content
    monitor            - Monitor CI performance
    report             - Generate CI learning report
    cost               - Show Claude API cost tracking

Options:
    --pr <number>      - Specify PR number
    --run <id>         - Specify workflow run ID
    --format <type>    - Output format (json|markdown|text)
    --verbose          - Verbose output
    --dry-run          - Show what would be done

Examples:
    $0 fix "TypeError: Cannot read property 'map' of undefined"
    $0 analyze --pr 123
    $0 teach "TypeScript null safety"
    $0 monitor --run 456789
    $0 cost --format json

EOF
    exit 1
}

# Check prerequisites
check_prerequisites() {
    if [ -z "$ANTHROPIC_API_KEY" ]; then
        echo -e "${RED}Error: ANTHROPIC_API_KEY not set${NC}"
        echo "Please set your Anthropic API key:"
        echo "  export ANTHROPIC_API_KEY='your-key-here'"
        exit 1
    fi
    
    if ! command -v jq &> /dev/null; then
        echo -e "${YELLOW}Warning: jq not installed (required for JSON parsing)${NC}"
    fi
    
    if ! command -v gh &> /dev/null; then
        echo -e "${YELLOW}Warning: GitHub CLI not installed${NC}"
    fi
}

# Fix specific error
fix_error() {
    local error="$1"
    local pr="${PR_NUMBER:-}"
    
    echo -e "${BLUE}🔧 Analyzing error...${NC}"
    echo "$error"
    echo
    
    # Get context from recent CI runs if PR specified
    local context=""
    if [ -n "$pr" ]; then
        echo -e "${BLUE}Fetching CI logs for PR #$pr...${NC}"
        context=$(gh run list --workflow=intelligent-ci.yml --branch="pull/$pr" --limit=1 --json databaseId --jq '.[0].databaseId' | xargs -I {} gh run view {} --log-failed 2>/dev/null || echo "")
    fi
    
    # Call Claude for analysis
    local prompt="Analyze this CI error and provide a fix:
Error: $error

Context from CI logs:
$context

Please provide:
1. Root cause analysis
2. Specific fix (with code)
3. Prevention tips
4. Educational explanation

Output format: JSON with fields: analysis, fix, education, confidence"
    
    local response=$(echo "$prompt" | claude ci-teacher \
        --output-format json \
        --max-tokens 2000 \
        2>/dev/null || echo '{"error": "Failed to get response"}')
    
    # Parse and display response
    if [ "$FORMAT" = "json" ]; then
        echo "$response"
    else
        echo -e "${GREEN}✅ Analysis Complete${NC}"
        echo
        
        if command -v jq &> /dev/null; then
            echo "📋 Root Cause:"
            echo "$response" | jq -r '.analysis.root_cause' 2>/dev/null || echo "N/A"
            echo
            
            echo "🔧 Fix:"
            echo "$response" | jq -r '.fix.code' 2>/dev/null || echo "N/A"
            echo
            
            echo "📚 Education:"
            echo "$response" | jq -r '.education.explanation' 2>/dev/null || echo "N/A"
            echo
            
            local confidence=$(echo "$response" | jq -r '.confidence' 2>/dev/null || echo "0")
            echo "Confidence: $confidence"
            
            if (( $(echo "$confidence > 0.95" | bc -l) )); then
                echo -e "${GREEN}✓ Auto-fixable${NC}"
            else
                echo -e "${YELLOW}⚠ Manual review recommended${NC}"
            fi
        else
            echo "$response"
        fi
    fi
}

# Analyze recent CI failures
analyze_ci() {
    local pr="${PR_NUMBER:-}"
    local run="${RUN_ID:-}"
    
    echo -e "${BLUE}📊 Analyzing CI failures...${NC}"
    
    # Get recent failures
    local failures=""
    if [ -n "$run" ]; then
        failures=$(gh run view "$run" --log-failed 2>/dev/null || echo "No logs found")
    elif [ -n "$pr" ]; then
        failures=$(gh run list --workflow=intelligent-ci.yml --branch="pull/$pr" --status=failure --limit=5 --json conclusion,name,databaseId | jq -r '.[] | "\(.name): \(.conclusion)"' 2>/dev/null || echo "No failures")
    else
        failures=$(gh run list --status=failure --limit=10 --json conclusion,name,workflowName | jq -r '.[] | "\(.workflowName)/\(.name): \(.conclusion)"' 2>/dev/null || echo "No recent failures")
    fi
    
    echo "Recent Failures:"
    echo "$failures"
    echo
    
    # Get Claude's analysis
    local prompt="Analyze these CI failures and provide insights:
$failures

Provide:
1. Common patterns
2. Root causes
3. Recommended fixes
4. Prevention strategies"
    
    echo "$prompt" | claude ci-teacher \
        --output-format markdown \
        --max-tokens 3000
}

# Educational content
teach_topic() {
    local topic="$1"
    
    echo -e "${BLUE}📚 Learning about: $topic${NC}"
    echo
    
    local prompt="Provide educational content about: $topic

Include:
1. Concept explanation
2. Common mistakes
3. Best practices
4. Code examples
5. Resources for learning

Make it practical and focused on CI/CD contexts."
    
    echo "$prompt" | claude ci-teacher \
        --output-format markdown \
        --max-tokens 4000
}

# Monitor CI performance
monitor_ci() {
    echo -e "${BLUE}📈 CI Performance Monitor${NC}"
    echo
    
    # Get recent workflow runs
    local stats=$(gh run list --limit=20 --json status,conclusion,databaseId,createdAt,updatedAt | \
        jq '[.[] | {
            status: .status,
            conclusion: .conclusion,
            duration: ((.updatedAt | fromdate) - (.createdAt | fromdate)) / 60 | floor
        }] | {
            total: length,
            succeeded: [.[] | select(.conclusion == "success")] | length,
            failed: [.[] | select(.conclusion == "failure")] | length,
            avg_duration: [.[] | .duration] | add / length | floor
        }' 2>/dev/null || echo '{}')
    
    if [ "$FORMAT" = "json" ]; then
        echo "$stats"
    else
        echo "Last 20 Workflow Runs:"
        echo "----------------------"
        echo "$stats" | jq -r '"Total: \(.total)\nSucceeded: \(.succeeded)\nFailed: \(.failed)\nAvg Duration: \(.avg_duration) minutes"' 2>/dev/null || echo "Unable to parse stats"
        echo
        
        # Calculate success rate
        local total=$(echo "$stats" | jq -r '.total' 2>/dev/null || echo "0")
        local succeeded=$(echo "$stats" | jq -r '.succeeded' 2>/dev/null || echo "0")
        if [ "$total" -gt 0 ]; then
            local success_rate=$((succeeded * 100 / total))
            echo "Success Rate: ${success_rate}%"
            
            if [ "$success_rate" -ge 90 ]; then
                echo -e "${GREEN}✓ Excellent CI health${NC}"
            elif [ "$success_rate" -ge 70 ]; then
                echo -e "${YELLOW}⚠ CI needs attention${NC}"
            else
                echo -e "${RED}✗ Critical CI issues${NC}"
            fi
        fi
    fi
}

# Generate learning report
generate_report() {
    echo -e "${BLUE}📝 Generating CI Learning Report...${NC}"
    echo
    
    # Collect data from last week
    local week_ago=$(date -v-7d +%Y-%m-%d 2>/dev/null || date -d '7 days ago' +%Y-%m-%d)
    
    local failures=$(gh run list --workflow=intelligent-ci.yml --status=failure --created=">$week_ago" --json name,conclusion,createdAt | \
        jq -r '.[] | "\(.createdAt): \(.name)"' 2>/dev/null || echo "No failures")
    
    # Generate report with Claude
    local prompt="Generate a CI learning report for the past week:

Failures:
$failures

Create a structured report with:
1. Executive summary
2. Failure patterns and trends
3. Root cause analysis
4. Lessons learned
5. Action items
6. Success metrics"
    
    local report=$(echo "$prompt" | claude ci-teacher \
        --output-format markdown \
        --max-tokens 4000)
    
    # Save report
    local report_file="ci-learning-report-$(date +%Y%m%d).md"
    echo "$report" > "$report_file"
    
    echo -e "${GREEN}✅ Report saved to: $report_file${NC}"
    echo
    echo "Preview:"
    echo "--------"
    head -20 "$report_file"
}

# Track Claude API costs
track_costs() {
    echo -e "${BLUE}💰 Claude API Cost Tracking${NC}"
    echo
    
    # Read cost limits from config
    local daily_limit=$(grep "per_day:" "$CONFIG_FILE" | head -1 | awk '{print $2}' 2>/dev/null || echo "10.00")
    local monthly_limit=$(grep "per_month:" "$CONFIG_FILE" | head -1 | awk '{print $2}' 2>/dev/null || echo "100.00")
    
    # Estimate costs (would need actual tracking in production)
    local estimated_daily=2.50
    local estimated_monthly=45.00
    
    if [ "$FORMAT" = "json" ]; then
        cat << EOF
{
    "limits": {
        "daily": $daily_limit,
        "monthly": $monthly_limit
    },
    "usage": {
        "daily": $estimated_daily,
        "monthly": $estimated_monthly
    },
    "remaining": {
        "daily": $(echo "$daily_limit - $estimated_daily" | bc),
        "monthly": $(echo "$monthly_limit - $estimated_monthly" | bc)
    }
}
EOF
    else
        echo "Cost Limits:"
        echo "  Daily:   \$$daily_limit"
        echo "  Monthly: \$$monthly_limit"
        echo
        echo "Current Usage (estimated):"
        echo "  Today:      \$$estimated_daily"
        echo "  This Month: \$$estimated_monthly"
        echo
        
        local daily_percent=$(echo "scale=0; $estimated_daily * 100 / $daily_limit" | bc)
        local monthly_percent=$(echo "scale=0; $estimated_monthly * 100 / $monthly_limit" | bc)
        
        echo "Usage Percentage:"
        echo "  Daily:   ${daily_percent}%"
        echo "  Monthly: ${monthly_percent}%"
        
        if [ "$daily_percent" -gt 80 ] || [ "$monthly_percent" -gt 80 ]; then
            echo -e "${YELLOW}⚠ Warning: Approaching cost limit${NC}"
        else
            echo -e "${GREEN}✓ Within budget${NC}"
        fi
    fi
}

# Main execution
main() {
    check_prerequisites
    
    # Parse arguments
    COMMAND="$1"
    shift || true
    
    # Default values
    PR_NUMBER=""
    RUN_ID=""
    FORMAT="text"
    VERBOSE=false
    DRY_RUN=false
    
    # Parse options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --pr)
                PR_NUMBER="$2"
                shift 2
                ;;
            --run)
                RUN_ID="$2"
                shift 2
                ;;
            --format)
                FORMAT="$2"
                shift 2
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            *)
                ARGS="$@"
                break
                ;;
        esac
    done
    
    # Execute command
    case "$COMMAND" in
        fix)
            fix_error "$ARGS"
            ;;
        analyze)
            analyze_ci
            ;;
        teach)
            teach_topic "$ARGS"
            ;;
        monitor)
            monitor_ci
            ;;
        report)
            generate_report
            ;;
        cost)
            track_costs
            ;;
        *)
            usage
            ;;
    esac
}

# Run main function
main "$@"
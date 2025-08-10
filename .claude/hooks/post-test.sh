#!/bin/bash
# Post-test hook: Actions after running tests

# Get test results from environment or previous command
TEST_EXIT_CODE=$?

# Log test results
if [ $TEST_EXIT_CODE -eq 0 ]; then
  echo "✅ Tests passed successfully"
  
  # Generate coverage report if available
  if [ -f "coverage.xml" ]; then
    echo "📊 Coverage report:"
    if command -v coverage &> /dev/null; then
      coverage report
    fi
  fi
else
  echo "❌ Tests failed with exit code: $TEST_EXIT_CODE"
  
  # Save failed test output for analysis
  if [ -f "test-results.xml" ]; then
    echo "Failed tests saved to test-results.xml"
  fi
fi

# Send notifications if configured
if [ -n "$SLACK_WEBHOOK_URL" ] && [ $TEST_EXIT_CODE -ne 0 ]; then
  curl -X POST $SLACK_WEBHOOK_URL \
    -H 'Content-Type: application/json' \
    -d "{\"text\":\"Tests failed in $(pwd)\"}" \
    2>/dev/null || true
fi

exit 0
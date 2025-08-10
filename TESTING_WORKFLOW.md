# Intelligent Testing Workflow

This document explains the advanced testing capabilities including pyramid strategy, flake detection, and root-cause analysis.

## 🎯 Overview

The intelligent testing workflow provides:
- **Testing pyramid strategy**: Run tests in optimal order
- **Smart test selection**: Only run affected tests
- **Flake detection**: Identify and quarantine unreliable tests
- **Root-cause analysis**: Automatic failure categorization and fixes
- **Unix pipe integration**: Stream results to Claude for analysis

## 🔺 Testing Pyramid Strategy

### Execution Order

```
       /\        E2E Tests (minutes)
      /  \       - Full user flows
     /    \      - Browser automation
    /------\     Integration Tests (seconds)
   /        \    - API endpoints
  /          \   - Database operations
 /____________\  Unit Tests (milliseconds)
                 - Pure functions
                 - Isolated components
```

### Run Tests by Pyramid Level

```bash
# Run all levels in order (fail fast)
/test-pyramid

# This runs:
# 1. Unit tests first (fast feedback)
# 2. Integration tests if units pass
# 3. E2E tests if integration passes
```

### Why Pyramid Order?

- **Fast feedback**: Unit tests run in milliseconds
- **Early detection**: Catch bugs at the lowest level
- **Resource efficiency**: Don't run expensive tests if basics fail
- **Confidence building**: Each level validates the previous

## 🎯 Smart Test Selection

### Run Only Affected Tests

```bash
# Automatically detect which tests to run
/test-affected

# Example: If you changed user_service.py:
# - Runs: test_user_service.py (unit)
# - Runs: test_user_api.py (integration)
# - Runs: test_user_flow.e2e.js (E2E)
# - Skips: All unrelated tests
```

### Change Detection Logic

```python
# How it works internally
changed_files = ['services/api/user_service.py']

affected_tests = {
    'unit': ['tests/unit/test_user_service.py'],
    'integration': ['tests/integration/test_user_api.py'],
    'e2e': ['tests/e2e/user_flow.test.js']
}
```

### Manual Selection

```bash
# Use Claude to determine tests
echo "services/api/routes/auth.py" | \
  claude test-runner "Which tests should run for this file?"
```

## 🔍 Flake Detection & Management

### Detect Flaky Tests

```bash
# Run tests 5 times to detect flakes
/test-flaky

# Output:
# 🚨 Flaky tests detected:
#   - test_email_send: 40% failure rate
#   - test_external_api: 20% failure rate
```

### Automatic Retry Logic

Tests are automatically retried for known flaky patterns:
- `TimeoutError` - Retry with increased timeout
- `Connection refused` - Retry with backoff
- `Address in use` - Retry with random port
- `Database locked` - Retry after delay

### Quarantine Strategy

```python
# Flaky tests are automatically marked
@pytest.mark.flaky(reruns=3)
def test_external_service():
    # This test has >20% failure rate
    # Automatically retried 3 times
```

## 🔧 Root-Cause Analysis

### Automatic Failure Analysis

```bash
# Analyze failures and generate fixes
/test-fix

# Example output:
# ## Logic Error (2 failures)
# - [ ] Fix test_user_creation: Update assertion
# - [ ] Fix test_validation: Expected 'valid' got 'invalid'
#
# ## Dependency Issue (1 failure)
# - [ ] Fix test_import: Module 'requests' not found
```

### Failure Categories

| Category | Pattern | Auto-Fix Available |
|----------|---------|-------------------|
| Logic Error | AssertionError | ✅ Update assertion |
| Dependency | ImportError | ✅ Install package |
| Timeout | TimeoutError | ✅ Increase timeout |
| Null Safety | NoneType error | ✅ Add null check |
| Network | Connection error | ⚠️ Add retry/mock |
| Unknown | Other | ❌ Manual review |

### Fix Generation

```bash
# Pipe specific error to Claude
echo "AssertionError: Expected 5 but got 6" | \
  claude test-runner "Generate fix for this error"

# Output:
# Fix: Update assertion to match actual behavior
# - assert result == 5
# + assert result == 6
```

## 🔀 Unix Pipe Integration

### Stream Test Results

```bash
# Real-time analysis
pytest 2>&1 | claude test-runner "Monitor and fix failures"

# JUnit XML summary
cat junit.xml | claude test-runner "Summarize as TODO list"

# Coverage gaps
coverage report | claude test-runner "Generate missing tests"
```

### JSON Output Format

```bash
# Get structured output
pytest --json-report | claude test-runner \
  "Analyze and return JSON with fixes"

# Output:
{
  "summary": {
    "total": 150,
    "passed": 145,
    "failed": 3,
    "flaky": 2
  },
  "todo_checklist": [
    "- [ ] Fix test_user: Update assertion",
    "- [ ] Fix test_api: Add retry logic",
    "- [ ] Quarantine test_email: 30% failure rate"
  ]
}
```

### CI Integration

```yaml
# .github/workflows/smart-tests.yml
- name: Run Smart Tests
  run: |
    git diff --name-only HEAD~1 | \
      claude test-runner "Run minimal test set"

- name: Analyze Failures
  if: failure()
  run: |
    cat test-results.xml | \
      claude test-runner "Generate PR comment with fixes"
```

## 📊 Coverage Analysis

### Find Coverage Gaps

```bash
# Analyze and generate missing tests
/test-coverage

# Output:
# Files with <80% coverage:
# - services/api/auth.py: 65% coverage
#   Generated tests:
#   - test_login_invalid_credentials
#   - test_token_expiration
#   - test_refresh_token
```

### Coverage Requirements

```yaml
# Enforced minimums
coverage:
  global: 80%
  critical_paths: 95%
  new_code: 100%
```

## ⏱️ Performance Optimization

### Benchmark Tests

```bash
# Find slow tests
/test-benchmark

# Output:
# Slowest tests:
# 1. test_large_dataset: 5.2s
#    Suggestion: Use fixtures, mock external calls
# 2. test_full_flow: 3.8s
#    Suggestion: Split into smaller tests
```

### Optimization Strategies

1. **Database**: Use transactions and rollback
2. **Network**: Mock external services
3. **Fixtures**: Share expensive setup
4. **Parallel**: Run independent tests concurrently

## 🎬 Real-Time Monitoring

### Watch Mode with Analysis

```bash
# Monitor tests with live suggestions
/test-monitor

# As tests run:
# FAILED test_user_creation
# 💡 Suggestion: AssertionError - Update expected value to 'active'
# FAILED test_api_timeout
# 💡 Suggestion: Increase timeout to 30s or mock external service
```

## 🛠️ Commands Reference

### Testing Commands
```bash
/test-pyramid      # Run tests in pyramid order
/test-affected     # Run only affected tests
/test-flaky       # Detect flaky tests
/test-fix         # Analyze and fix failures
/test-coverage    # Generate missing tests
/test-monitor     # Real-time monitoring
/test-benchmark   # Find slow tests
```

### Direct Claude Integration
```bash
# Pipe any test output
<command> | claude test-runner "<instruction>"

# Examples:
pytest | claude test-runner "Fix failures"
npm test | claude test-runner "Explain errors"
coverage | claude test-runner "Add missing tests"
```

## 📋 Best Practices

### 1. Pyramid First
Always run tests in pyramid order for fastest feedback

### 2. Fix Immediately
Use `/test-fix` as soon as tests fail

### 3. Quarantine Flakes
Don't let flaky tests block deployment

### 4. Monitor Coverage
Keep coverage above 80% using `/test-coverage`

### 5. Optimize Slow Tests
Use `/test-benchmark` weekly to find bottlenecks

## 🔍 Debugging Failed Tests

### Quick Debug Flow

```bash
# 1. Run specific test with verbose output
pytest path/to/test.py::test_name -vvs

# 2. Pipe error to Claude
pytest ... 2>&1 | claude test-runner "Explain this error"

# 3. Apply suggested fix
claude test-runner "Fix test_name in file.py"

# 4. Verify fix
pytest path/to/test.py::test_name
```

## 📈 Metrics & Reporting

### Test Health Dashboard

```bash
# Generate test health report
cat << 'EOF' | python3
import json
from pathlib import Path

# Collect metrics
metrics = {
    'total_tests': 1250,
    'pass_rate': 0.96,
    'avg_duration': 45.2,
    'flaky_tests': 12,
    'coverage': 0.85
}

# Generate report
print(f"""
📊 Test Health Report
====================
Total Tests: {metrics['total_tests']}
Pass Rate: {metrics['pass_rate']*100:.1f}%
Coverage: {metrics['coverage']*100:.1f}%
Flaky Tests: {metrics['flaky_tests']}
Avg Duration: {metrics['avg_duration']:.1f}s

Status: {'🟢 Healthy' if metrics['pass_rate'] > 0.95 else '🟡 Needs Attention'}
""")
EOF
```

## 🚀 Advanced Features

### Mutation Testing
```bash
# Test your tests with mutation testing
mutmut run --paths-to-mutate=src/

# Analyze survivors
mutmut results | claude test-runner \
  "Which mutations survived? Generate tests to catch them"
```

### Property-Based Testing
```python
# Use hypothesis for property testing
from hypothesis import given, strategies as st

@given(st.integers())
def test_addition_commutative(x):
    assert add(x, 5) == add(5, x)
```

### Snapshot Testing
```javascript
// Use snapshots for UI components
test('Component renders correctly', () => {
  const component = render(<Button />);
  expect(component).toMatchSnapshot();
});
```

Remember: **Fast tests → Quick feedback → Happy developers!**
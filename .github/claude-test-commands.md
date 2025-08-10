# GitHub Test Commands via PR Comments

This document shows how to trigger Claude test commands directly from GitHub PR comments.

## 🤖 Available Commands

### In PR Comments

Type these commands in a PR comment to trigger actions:

#### Test Execution
```
@claude run tests
```
Runs only affected tests based on PR changes

```
@claude run all tests
```
Runs complete test suite following pyramid strategy

```
@claude test pyramid
```
Runs tests in pyramid order (unit → integration → E2E)

#### Failure Analysis
```
@claude analyze failures
```
Analyzes test failures and posts root cause analysis

```
@claude fix tests
```
Automatically fixes simple test failures and creates a PR

```
@claude explain error <error message>
```
Explains specific error and suggests fix

#### Flake Detection
```
@claude detect flakes
```
Runs tests multiple times to identify flaky tests

```
@claude quarantine flaky tests
```
Marks flaky tests to skip in CI

#### Coverage
```
@claude check coverage
```
Analyzes coverage and suggests missing tests

```
@claude generate tests for <file>
```
Generates test cases for specific file

#### Performance
```
@claude benchmark tests
```
Runs performance benchmarks and identifies slow tests

```
@claude optimize slow tests
```
Suggests optimizations for slowest tests

## 🔄 Automatic Triggers

### On Every PR

1. **Smart Test Selection**: Only runs affected tests
2. **Pyramid Execution**: Fails fast at lowest level
3. **Failure Analysis**: Posts root cause as comment
4. **Coverage Check**: Reports coverage changes

### On Test Failure

1. **Flake Detection**: Retries suspected flakes
2. **Root Cause Analysis**: Categorizes failures
3. **Fix Generation**: Creates TODO checklist
4. **Auto-Fix Option**: Can create fix PR

## 📊 PR Status Checks

### Required Checks
- ✅ Unit Tests
- ✅ Integration Tests (if needed)
- ✅ E2E Tests (if needed)
- ✅ Coverage > 80%

### Optional Checks
- 🔍 Flake Detection
- 🔧 Auto-Fix Available
- ⚡ Performance Benchmark

## 💬 Comment Examples

### Request Specific Analysis
```markdown
@claude analyze failures

The user creation test is failing. Can you check if it's related to the schema change?
```

### Fix Specific Test
```markdown
@claude fix test_user_validation

This test has been failing since we updated the validation rules.
```

### Generate Missing Tests
```markdown
@claude generate tests for services/api/auth.py

Coverage dropped below 80% for this file.
```

## 🎯 Smart Features

### 1. Affected Test Detection
- Analyzes git diff
- Maps changes to tests
- Runs minimal set first

### 2. Failure Categorization
- **Logic Error**: Assertion failures
- **Dependency**: Import/package issues
- **Timeout**: Performance problems
- **Null Safety**: None/null errors
- **Network**: External service issues

### 3. Auto-Fix Capabilities
- Update assertions
- Fix imports
- Increase timeouts
- Add null checks
- Mock external services

### 4. Flake Management
- Detects intermittent failures
- Retries with backoff
- Quarantines persistent flakes
- Creates tracking issues

## 🔧 Configuration

### Enable Claude Commands

Add to `.github/claude.yml`:
```yaml
test_commands:
  enabled: true
  auto_fix: true
  coverage_threshold: 80
  flake_retry: 3
  pyramid_mode: true
```

### Permissions

Add Claude app with permissions:
- Read: Code, metadata
- Write: Issues, pull requests, checks
- Admin: Actions

## 📈 Test Reports

### In PR Comments
```markdown
## 🧪 Test Results

### Summary
- Total: 245 tests
- Passed: 240 ✅
- Failed: 3 ❌
- Flaky: 2 ⚠️
- Duration: 2m 34s

### Failures
1. `test_user_creation` - Logic Error
   - Expected 'active' but got 'pending'
   - Fix: Update assertion to match new default

2. `test_email_send` - Network Error
   - Connection timeout to SMTP
   - Fix: Mock email service in tests

3. `test_data_migration` - Timeout
   - Exceeded 5s timeout
   - Fix: Increase timeout to 30s

### TODO Checklist
- [ ] Fix test_user_creation: Update status assertion
- [ ] Fix test_email_send: Add email mock
- [ ] Fix test_data_migration: Increase timeout
- [ ] Quarantine test_external_api (40% failure rate)

### Coverage
- Current: 84.5% ✅
- Change: +2.3%
- Files below 80%:
  - `auth.py`: 72% (needs 8% more)
  - `payments.py`: 65% (needs 15% more)
```

## 🚀 Advanced Usage

### Chain Commands
```
@claude run tests && analyze failures && fix tests
```

### Conditional Execution
```
@claude if tests fail then generate fixes
```

### Custom Prompts
```
@claude test-runner "Focus on authentication tests and check for race conditions"
```

## 🔒 Security

- Commands only work from authorized users
- Auto-fixes require approval
- Sensitive tests never auto-fixed
- All actions are logged

## 📚 Examples

### Complete Workflow
```markdown
# Developer creates PR
1. CI runs affected tests automatically
2. Test fails with timeout error

# Developer comments:
@claude analyze failures

# Claude responds:
The test is failing due to a network timeout when calling the payment API.
This appears to be a flaky test that should mock the external service.

# Developer comments:
@claude fix tests

# Claude creates fix PR:
- Adds mock for payment API
- Updates test to use mock
- Increases timeout as fallback

# Original PR is now green ✅
```

Remember: Claude is here to keep your tests green and your velocity high!
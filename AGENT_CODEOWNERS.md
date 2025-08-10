# Agent-Based Code Ownership

## 🤖 How It Works

Instead of human teams, our CODEOWNERS file maps code paths to specialized Claude agents. This enables automatic, intelligent code review and maintenance.

## The Agent Owners

### @claude-reviewer
- **Owns**: Everything (default fallback)
- **Responsibilities**: General code review, best practices, style consistency
- **Auto-triggered**: On all PRs

### @claude-frontend
- **Owns**: `/apps/`, `*.tsx`, `*.jsx`, `*.css`
- **Responsibilities**: React components, styling, UX implementation
- **Auto-triggered**: When frontend files change

### @claude-backend
- **Owns**: `/services/`, `*.py`, `requirements.txt`
- **Responsibilities**: API endpoints, business logic, integrations
- **Auto-triggered**: When backend files change

### @claude-migrator
- **Owns**: `/migrations/`, `*.sql`, `models.py`
- **Responsibilities**: Database schema changes, data migrations
- **Auto-triggered**: When database structure changes

### @claude-test-runner
- **Owns**: `/tests/`, `*.test.*`, `*.spec.*`
- **Responsibilities**: Test coverage, test fixes, test generation
- **Auto-triggered**: When tests change or code without tests is added

### @claude-api-contractor
- **Owns**: `/docs/api/`, `openapi.yaml`, `*.graphql`
- **Responsibilities**: API documentation, contract validation
- **Auto-triggered**: When API contracts change

### @claude-discovery-writer
- **Owns**: `PRD.md`, `README.md`, `/docs/`
- **Responsibilities**: Documentation, requirements, changelogs
- **Auto-triggered**: When documentation needs updating

### @claude-journey-planner
- **Owns**: `/docs/ux/`, `/wireframes/`
- **Responsibilities**: User flows, screen designs, UX documentation
- **Auto-triggered**: When UX documents change

### @claude-security-scanner
- **Owns**: `.env.example`, `/auth/`, `*.pem`, `*.key`
- **Responsibilities**: Security review, vulnerability scanning
- **Auto-triggered**: When security-sensitive files change

### @claude-scaffolder
- **Owns**: `/.github/`, `/.claude/`, `Dockerfile`, CI/CD files
- **Responsibilities**: Infrastructure, DevOps, project configuration
- **Auto-triggered**: When infrastructure files change

## 📋 Automatic Workflows

### On Pull Request

```mermaid
graph LR
    A[PR Created] --> B[CODEOWNERS Check]
    B --> C{Which Files Changed?}
    C -->|Frontend| D[@claude-frontend reviews]
    C -->|Backend| E[@claude-backend reviews]
    C -->|Database| F[@claude-migrator reviews]
    C -->|Security| G[@claude-security-scanner reviews]
    D --> H[Post Review Comments]
    E --> H
    F --> H
    G --> H
```

### Example PR Review Flow

1. **Developer creates PR** changing `/services/api/users.py`

2. **GitHub checks CODEOWNERS**:
   - `*.py` → @claude-backend
   - `/services/` → @claude-backend

3. **GitHub Action triggers**:
   ```bash
   claude backend "Review PR #123 changes to users.py"
   ```

4. **Claude Backend Agent**:
   - Reviews code for FastAPI best practices
   - Checks async/await usage
   - Validates Pydantic models
   - Ensures proper error handling

5. **Posts review comment**:
   ```markdown
   ## Backend Agent Review
   
   ✅ Approved with suggestions
   
   ### Suggestions:
   - Line 45: Consider using async for database call
   - Line 78: Add input validation for email field
   - Line 92: Missing error handling for 404 case
   ```

## 🔧 GitHub Integration

### Setup GitHub Action

```yaml
# .github/workflows/agent-review.yml
name: Agent Code Review

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  agent-review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Get changed files
        id: files
        uses: tj-actions/changed-files@v40
        
      - name: Determine code owners
        id: owners
        run: |
          # Parse CODEOWNERS to find which agents to invoke
          python .github/scripts/get-owners.py "${{ steps.files.outputs.all_changed_files }}"
          
      - name: Invoke agent reviews
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          # For each owner agent, run review
          for agent in ${{ steps.owners.outputs.agents }}; do
            claude $agent "Review PR #${{ github.event.pull_request.number }}"
          done
```

### Manual Agent Invocation

You can also manually request specific agent reviews:

```markdown
# In PR comment:
@claude-security-scanner please review for security issues
@claude-test-runner add missing tests
@claude-backend optimize database queries
```

## 🎯 Benefits

### 1. Consistent Reviews
Every PR gets reviewed by the appropriate specialist agent, ensuring consistent quality standards.

### 2. 24/7 Availability
Agents are always available - no waiting for human reviewers in different time zones.

### 3. Domain Expertise
Each agent is specialized in their domain with deep knowledge of best practices.

### 4. Automatic Fixes
Agents can not only review but also suggest or implement fixes:

```bash
# Agent sees missing tests
@claude-test-runner: "I notice auth.py lacks tests. Generating them now..."
# Creates PR with tests
```

### 5. Learning & Improvement
Agents learn from your codebase patterns and improve their reviews over time.

## 📊 Metrics & Tracking

Track agent effectiveness:

```bash
# See agent review stats
claude "Show me stats on agent reviews this month"

# Output:
Agent Review Statistics - November 2024
- Total PRs reviewed: 156
- Reviews by agent:
  - @claude-backend: 89
  - @claude-frontend: 67
  - @claude-security-scanner: 45
  - @claude-test-runner: 34
- Issues found: 234
- Auto-fixed: 145
- Average review time: 2.3 minutes
```

## 🔄 Customizing Agent Ownership

### Add New Ownership Rules

Edit `.github/CODEOWNERS`:

```bash
# Add new pattern
/new-feature/ @claude-backend @claude-test-runner
```

### Create Custom Agent

1. Create agent definition:
```bash
vim .claude/subagents/performance.md
```

2. Add to CODEOWNERS:
```bash
/benchmarks/ @claude-performance
*.bench.js @claude-performance
```

3. Agent automatically starts reviewing performance-related changes

## 🚀 Advanced Patterns

### Multi-Agent Reviews

Some changes need multiple agents:

```bash
# Changes to payment processing
/payments/ @claude-backend @claude-security-scanner @claude-test-runner
```

All three agents review payment-related changes.

### Conditional Ownership

```python
# .github/scripts/dynamic-owners.py
def get_owners(file_path):
    if "critical" in file_path:
        return ["@claude-security-scanner", "@claude-reviewer"]
    elif file_path.endswith(".sql"):
        return ["@claude-migrator", "@claude-backend"]
    # ... more conditions
```

### Agent Escalation

```yaml
# High-risk changes escalate to multiple agents
/auth/ @claude-security-scanner @claude-backend @claude-reviewer
/payments/ @claude-security-scanner @claude-backend @claude-reviewer
```

## 💡 Best Practices

1. **Start with defaults**: Let @claude-reviewer handle most files initially
2. **Add specific owners gradually**: As patterns emerge, add specific ownership
3. **Use multiple agents for critical paths**: Security, payments, auth
4. **Keep agents focused**: Each agent should have clear responsibilities
5. **Review agent feedback**: Continuously improve agent prompts based on results

This agent-based CODEOWNERS system ensures every line of code gets expert review, automatically!
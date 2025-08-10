# Context & Memory Management Guide

This guide explains how Claude maintains context and memory across sessions in this native template.

## 🧠 Overview

The template uses multiple layers of context management:
1. **CLAUDE.md** - Persistent project instructions
2. **PROJECT_STATE.md** - Current tasks and progress
3. **Subagent Context** - Specialized knowledge per agent
4. **Volumetrics & Decisions** - Data-driven context
5. **Git History** - Change tracking
6. **Automated Updates** - Self-maintaining documentation

## 📁 Core Memory Files

### 1. CLAUDE.md (Project Instructions)
**Location**: `/CLAUDE.md`
**Purpose**: Persistent project context that Claude reads EVERY session
**Updates**: Manually or via Claude when architecture changes

```markdown
# Project Instructions

## Stack
- FastAPI with async/await
- PostgreSQL for data, Redis for cache
- Next.js 14 with TypeScript

## Code Standards
- Test coverage minimum 80%
- Use Pydantic for validation

## Patterns
- Repository pattern for data
- Service layer for logic

## Recent Decisions
- 2024-01-08: Switched to PostgreSQL from MySQL
- 2024-01-07: Added feature flags system

## Known Issues
- Performance bottleneck in /api/users endpoint
- Migration needed for user preferences
```

### 2. PROJECT_STATE.md (Working Memory)
**Location**: `/.claude/PROJECT_STATE.md`
**Purpose**: Track current work, TODOs, and blockers
**Updates**: After each significant change

```markdown
# Project State

## Current Sprint (2024-01-08)
- Implementing payment system
- Feature flag: payment-v2

## In Progress
- [ ] Payment webhook handler (@john)
- [ ] Stripe integration tests (@jane)
- [x] Database schema for payments

## Blocked
- Waiting for Stripe API keys from DevOps

## Completed Today
- ✅ Payment model created
- ✅ API endpoints scaffolded

## Context for Next Session
- Remember: Using Stripe Connect, not standard Stripe
- Test webhook with: stripe listen --forward-to localhost:8000/webhook
- Migration pending: alembic upgrade head
```

## 🤖 Subagent Memory

Each subagent maintains its own context via frontmatter:

### Example: API Contractor Agent
```yaml
---
name: api-contractor
paths:
  - 'docs/api/**'
  - 'services/api/schemas/**'
memory:
  last_openapi_version: "3.1.0"
  validation_library: "pydantic"
  sdk_languages: ["typescript", "python"]
  recent_changes:
    - "2024-01-08: Added pagination to all list endpoints"
    - "2024-01-07: Standardized error responses"
---
```

## 📊 Data-Driven Context

### VOLUMETRICS.md
Stores critical data metrics that affect all decisions:

```markdown
# System Volumetrics

## Database
| Table | Row Count | Growth Rate | Size |
|-------|-----------|-------------|------|
| users | 1.2M | 5K/day | 2.3GB |
| orders | 8.5M | 50K/day | 18GB |

## Performance Targets
- API p95 latency: <200ms
- Database queries: <50ms
- Concurrent users: 10K

## These metrics inform:
- Migration strategies (chunked for large tables)
- Caching decisions (Redis for hot paths)
- Scaling plans (read replicas needed)
```

## 🔄 Automated Context Updates

### 1. Git Hooks for State Tracking
```bash
# .git/hooks/post-commit
#!/bin/bash
# Auto-update PROJECT_STATE.md after commits

TASK=$(git log -1 --pretty=%B | grep -oP 'feat:|fix:' | head -1)
if [ -n "$TASK" ]; then
  echo "- ✅ $TASK" >> .claude/PROJECT_STATE.md
fi
```

### 2. Session Persistence
```yaml
# .claude/config.yaml
session:
  persist: true
  max_context: 100000
  summarize_after: 50000
  
memory:
  auto_save: true
  save_on:
    - file_write
    - test_run
    - deployment
```

### 3. Context Filtering Rules
```yaml
# .claude/subagents/filtering.yaml
rules:
  - name: "Exclude test files from context"
    pattern: "**/*.test.{js,ts,py}"
    action: exclude
    unless: "working_on_tests"
    
  - name: "Include recent changes"
    pattern: "git log --oneline -20"
    action: include
    when: "starting_session"
    
  - name: "Include error logs"
    pattern: "logs/error.log"
    action: include
    max_lines: 100
    when: "debugging"
```

## 📝 ADRs as Long-term Memory

Architecture Decision Records serve as long-term memory:

```markdown
# ADR-001: Use PostgreSQL

Date: 2024-01-08
Status: Accepted

## Context
We need ACID compliance for financial transactions...

## Decision
Use PostgreSQL with connection pooling...

## Claude Memory Note
When working with database:
- Always use connection pooling
- Index foreign keys
- Use JSONB for flexible schemas
```

## 🔍 Context Retrieval Patterns

### 1. Semantic Search
Claude can search for relevant context:

```typescript
// .claude/memory/search.ts
export async function findRelevantContext(query: string) {
  const contexts = [
    await searchADRs(query),
    await searchProjectState(query),
    await searchGitHistory(query),
    await searchDocumentation(query)
  ];
  
  return rankByRelevance(contexts, query);
}
```

### 2. Time-based Context
Recent changes get priority:

```bash
# Get context for current work
claude context --recent 7d --include-decisions --include-blockers
```

### 3. Task-specific Context
Each slash command loads relevant context:

```yaml
# /api-design loads:
- docs/api/openapi.yaml
- Recent API changes
- Validation patterns
- SDK requirements

# /test loads:
- Test coverage reports
- Flaky test history
- Recent test failures
- Testing patterns
```

## 🚀 Best Practices for Context Management

### 1. Update CLAUDE.md When:
- Technology stack changes
- Major architectural decisions made
- New patterns adopted
- Team conventions established

### 2. Update PROJECT_STATE.md:
```bash
# After each work session
/session-summary

# When switching tasks
/context-switch "Working on payments"

# When blocked
/blocked "Waiting for API keys"
```

### 3. Use Semantic Markers
Help Claude understand importance:

```markdown
## 🚨 CRITICAL: Always use this pattern
## ⚠️ WARNING: Known issue here
## 💡 TIP: Performance optimization
## 📌 REMEMBER: For next session
## 🔒 SECURITY: Never expose this
```

### 4. Leverage Git for Memory
```bash
# Tag important decisions
git tag -a "decision-use-postgresql" -m "ADR-001: PostgreSQL chosen"

# Use semantic commits for context
git commit -m "feat(payments): Add Stripe webhook handler

Context: Using Stripe Connect for marketplace
Remember: Test with stripe listen command
Related: ADR-015-payment-architecture"
```

## 🔄 Context Lifecycle

### Session Start
1. Claude reads CLAUDE.md
2. Loads PROJECT_STATE.md
3. Gets recent git history
4. Checks for blocking issues
5. Loads relevant subagent contexts

### During Work
1. Updates PROJECT_STATE.md incrementally
2. Creates breadcrumbs in comments
3. Maintains decision log
4. Tracks dependencies

### Session End
1. Summarizes changes
2. Updates PROJECT_STATE.md
3. Creates TODOs for next session
4. Archives decisions to ADRs

## 📊 Memory Optimization

### 1. Context Pruning
```python
# .claude/scripts/prune_context.py
def prune_old_context():
    """Remove outdated context to stay within limits."""
    
    # Archive completed tasks older than 30 days
    archive_completed_tasks()
    
    # Summarize old decisions
    summarize_old_adrs()
    
    # Compress git history
    compress_git_context()
    
    # Remove stale TODOs
    clean_stale_todos()
```

### 2. Smart Summarization
```yaml
# When context exceeds 50K tokens
summarization:
  strategy: "hierarchical"
  preserve:
    - current_task
    - blockers
    - recent_decisions
    - critical_patterns
  summarize:
    - old_tasks
    - resolved_issues
    - implemented_features
```

### 3. Contextual Caching
```typescript
// Cache frequently needed context
const contextCache = new Map<string, Context>();

// Pre-load common contexts
contextCache.set('api-patterns', await loadAPIPatterns());
contextCache.set('test-patterns', await loadTestPatterns());
contextCache.set('security-rules', await loadSecurityRules());
```

## 🎯 Commands for Context Management

### View Current Context
```bash
# Show what Claude knows
/context

# Show specific context
/context api
/context testing
/context security
```

### Update Context
```bash
# Add to working memory
/remember "Use Stripe Connect, not standard Stripe"

# Update project state
/update-state "Payment system 80% complete"

# Add blocker
/blocked "Need production API keys"
```

### Clean Context
```bash
# Archive old tasks
/archive-completed

# Summarize verbose context
/summarize-context

# Reset working memory
/clear-state
```

## 🔐 Security Considerations

### Never Store in Context:
- API keys or secrets
- User PII data
- Production database credentials
- Security vulnerabilities details

### Safe Context Storage:
```markdown
## Database
Connection: Use env var DATABASE_URL
Schema: See migrations/

## API Keys
Stripe: env var STRIPE_SECRET_KEY
AWS: IAM role authentication

## Security Notes
- SQL injection prevented via Pydantic
- XSS handled by React
- CSRF tokens required
```

## 📈 Metrics for Context Effectiveness

Track how well context is working:

```typescript
interface ContextMetrics {
  contextRelevanceScore: number;  // 0-1, how often context is used
  contextMissRate: number;        // Times needed info wasn't in context
  contextOverhead: number;         // Tokens used for context vs work
  decisionAccuracy: number;        // Decisions aligned with context
}

// Log when context helps or hinders
/context-feedback "helpful: found migration pattern in ADR-003"
/context-feedback "missing: no context about websocket implementation"
```

## 🚦 Context Health Indicators

### Green (Healthy)
- PROJECT_STATE.md updated today ✅
- Recent decisions documented ✅
- No unresolved blockers > 3 days ✅
- Context size < 50K tokens ✅

### Yellow (Needs Attention)
- PROJECT_STATE.md > 3 days old ⚠️
- Multiple unresolved blockers ⚠️
- Context size 50-80K tokens ⚠️
- Missing ADRs for decisions ⚠️

### Red (Critical)
- No PROJECT_STATE.md ❌
- Context > 80K tokens ❌
- Conflicting information ❌
- Stale beyond 1 week ❌

---

Remember: **Good context management = Better Claude assistance**

The key is to maintain a living, breathing context that evolves with your project while staying focused and relevant.
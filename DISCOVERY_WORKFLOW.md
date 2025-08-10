# Discovery & PRD Workflow with Claude

## 🚀 From Idea to Implementation

This workflow shows how to go from rough notes to a fully-planned project with GitHub issues, using Claude's specialized agents.

## The Three Discovery Agents

### 1. `discovery-writer`
- Creates comprehensive PRDs from notes
- Formats acceptance criteria for automation
- Defines success metrics and timelines

### 2. `journey-planner`
- Designs user flows and screen layouts
- Maps user journeys and interactions
- Creates route structures and navigation

### 3. `api-contractor`
- Generates OpenAPI/GraphQL specifications
- Creates example payloads
- Defines API contracts and schemas

## 📋 Complete Workflow Example

### Step 1: Start with Notes

```bash
# Create rough notes
cat > notes.md << 'EOF'
We need a portfolio tracking app where users can:
- Track their stock investments
- See real-time prices
- Calculate gains/losses
- View performance charts
- Set price alerts

Target: 100 users in 3 months
Must work on mobile
Need OAuth login
EOF
```

### Step 2: Generate PRD

```bash
# Interactive mode - Claude asks clarifying questions
claude discovery-writer "Create a PRD from notes.md"

# Or use the slash command
claude /prd notes.md

# Result: Creates /docs/PRD.md with:
# - Problem statement
# - User stories
# - Acceptance criteria (AC-001, AC-002, etc.)
# - Success metrics
# - Timeline
```

### Step 3: Design UX Flows

```bash
# Claude reads PRD and creates screens
claude journey-planner "Design UX flows from docs/PRD.md"

# Or use slash command
claude /plan-ux

# Result: Creates /docs/ux/ with:
# - user-journeys.md
# - screen-flows.md
# - Routes and navigation
# - Component inventory
```

### Step 4: Design API

```bash
# Claude creates API specification
claude api-contractor "Generate OpenAPI spec from docs/PRD.md"

# Or use slash command
claude /design-api

# Result: Creates /docs/api/ with:
# - openapi.yaml
# - Example requests/responses
# - GraphQL schema (if needed)
```

### Step 5: Create GitHub Issues

```bash
# Parse PRD and create issues automatically
claude /prd-to-issues "Project Alpha"

# This will:
# 1. Parse all AC-XXX items from PRD
# 2. Create individual GitHub issues
# 3. Add appropriate labels (backend, frontend, size-s/m/l)
# 4. Add to Project board
# 5. Create tracking epic
```

### Step 6: Or Do Everything at Once!

```bash
# Run complete discovery process
claude /discovery

# This runs all agents in sequence:
# 1. discovery-writer → PRD
# 2. journey-planner → UX flows
# 3. api-contractor → API spec
# 4. prd-to-issues → GitHub issues
```

## 📝 PRD Format for Automation

The PRD must format acceptance criteria like this for automation:

```markdown
## Acceptance Criteria
- [ ] AC-001: User can register with email and password
- [ ] AC-002: User can log in with OAuth (Google/GitHub)
- [ ] AC-003: User can add stocks to portfolio
- [ ] AC-004: System displays real-time stock prices
- [ ] AC-005: User can view performance charts
```

Each AC becomes a GitHub issue with:
- Title: `[AC-001] User can register with email and password`
- Labels: Auto-detected (backend, frontend, size-s/m/l)
- Milestone: From PRD
- Linked to tracking epic

## 🎯 Using Sub-agents Directly

```bash
# Call specific agents
claude discovery-writer "Draft PRD for e-commerce platform"
claude journey-planner "Design checkout flow with 3 payment options"
claude api-contractor "Create REST API for user management"

# Claude auto-delegates when appropriate
claude "Plan a social media app" 
# Claude sees "plan" → delegates to discovery-writer

claude "Design the onboarding screens"
# Claude sees "screens" → delegates to journey-planner

claude "Create API for real-time chat"
# Claude sees "API" → delegates to api-contractor
```

## 📊 GitHub Integration

### Issue Templates Created

Each AC becomes an issue with this structure:

```markdown
## Acceptance Criteria
[Description from PRD]

## Definition of Done
- [ ] Implementation complete
- [ ] Tests written and passing  
- [ ] Documentation updated
- [ ] Code reviewed
- [ ] Deployed to staging

## Technical Notes
[Empty for team to fill]

## Source
- PRD: docs/PRD.md
- AC ID: AC-001
```

### Tracking Epic

A master tracking issue is created:

```markdown
## PRD Implementation Tracking

### Acceptance Criteria Issues
- [ ] AC-001: #1
- [ ] AC-002: #2
- [ ] AC-003: #3

### Progress
- Total: 15 issues
- Completed: 0
- In Progress: 0

### Milestones
- [ ] MVP: Issues #1-8
- [ ] V1.1: Issues #9-12
- [ ] V2.0: Issues #13-15
```

## 🔄 Iterative Refinement

```bash
# After team review, update PRD
vim docs/PRD.md

# Regenerate issues for new ACs only
claude /prd-to-issues

# Update UX based on feedback
claude journey-planner "Add mobile-specific flows to existing UX docs"

# Extend API
claude api-contractor "Add websocket events to existing API spec"
```

## 💡 Pro Tips

### 1. Start Small
```bash
# Begin with core features only
echo "MVP: User auth and basic portfolio view" > notes.md
claude /prd notes.md
```

### 2. Use Templates
```bash
# Create template for your domain
claude discovery-writer "Create a PRD template for SaaS products"
cp docs/PRD.md templates/saas-prd-template.md
```

### 3. Batch Operations
```bash
# Create multiple PRDs for different features
for feature in auth payments notifications; do
  claude discovery-writer "PRD for $feature feature" > docs/features/$feature-prd.md
done
```

### 4. Link Everything
```bash
# After creating issues, link them
gh issue edit 1 --add-link 2,3,4  # Dependencies
gh issue edit 1 --milestone "Sprint 1"
gh issue edit 1 --add-project "Q1 Roadmap"
```

## 📈 Metrics and Tracking

Claude can help track progress:

```bash
# Get implementation status
claude "Analyze GitHub issues labeled 'acceptance-criteria' and summarize progress"

# Update PRD with learnings
claude "Update docs/PRD.md with lessons learned from completed issues"

# Generate sprint report
claude "Create sprint report from closed issues in last 2 weeks"
```

## 🚦 Complete Example: Portfolio App

```bash
# 1. Initial notes
echo "Portfolio tracker with real-time prices" > notes.md

# 2. Full discovery
claude /discovery

# 3. Review generated docs
ls -la docs/
# PRD.md
# ux/user-journeys.md
# ux/screen-flows.md  
# api/openapi.yaml

# 4. Check GitHub issues
gh issue list --label acceptance-criteria

# 5. Start implementation
claude /feature portfolio-dashboard

# 6. Implement first AC
claude backend "Implement AC-001: User registration endpoint"
```

## 🎓 Why This Works

1. **Structured Discovery**: Forces thinking through requirements
2. **Automatic Decomposition**: Breaks work into manageable issues
3. **Parallel Development**: Frontend/backend can work from contracts
4. **Clear Success Criteria**: Everyone knows what "done" means
5. **Traceability**: From PRD → Issue → Code → Test

This workflow typically saves 10-15 hours of planning and setup time per project!
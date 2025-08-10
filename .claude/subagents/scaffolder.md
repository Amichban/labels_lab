---
name: scaffolder
description: Initialize repository structure and set up quality gates
tools:
  - bash
  - write_file
  - read_file
paths:
  - .github/**
  - .claude/**
  - scripts/**
  - "*"  # Root level config files only
---

# Scaffolder Agent

You are a specialized agent for repository initialization and quality gate setup.

## Responsibilities

### Repository Structure
```
project/
├── .github/
│   ├── workflows/         # CI/CD pipelines
│   ├── ISSUE_TEMPLATE/    # Issue templates
│   ├── pull_request_template.md
│   └── CODEOWNERS
├── .claude/
│   ├── subagents/         # Agent definitions
│   ├── hooks.yaml         # Automation hooks
│   ├── settings.json      # Claude settings
│   └── context-rules.yml  # Path restrictions
├── apps/                  # Frontend applications
├── services/              # Backend services
├── packages/              # Shared packages
├── scripts/               # Automation scripts
└── docs/                  # Documentation
```

### GitHub Templates

#### Issue Template (.github/ISSUE_TEMPLATE/feature.yml)
```yaml
name: Feature Request
description: Propose a new feature
labels: ["feature", "needs-triage"]
body:
  - type: textarea
    id: problem
    attributes:
      label: Problem Statement
      description: What problem does this solve?
    validations:
      required: true
  - type: textarea
    id: solution
    attributes:
      label: Proposed Solution
      description: How should we solve it?
  - type: checkboxes
    id: checklist
    attributes:
      label: Acceptance Criteria
      options:
        - label: User story defined
        - label: Success metrics identified
        - label: API contract agreed
```

#### PR Template (.github/pull_request_template.md)
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No sensitive data exposed
```

### Branch Protection Rules
```yaml
# .github/branch-protection.yml
protection_rules:
  - name: main
    required_reviews: 1
    dismiss_stale_reviews: true
    require_code_owner_reviews: true
    required_status_checks:
      - test
      - lint
      - security-scan
    enforce_admins: false
    restrictions:
      users: []
      teams: ["maintainers"]
```

### Quality Gates

#### Pre-commit Hooks (.pre-commit-config.yaml)
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: detect-private-key
      
  - repo: https://github.com/psf/black
    rev: 24.1.0
    hooks:
      - id: black
        
  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
        
  - repo: https://github.com/pre-commit/mirrors-prettier
    rev: v3.1.0
    hooks:
      - id: prettier
        files: \.(js|ts|jsx|tsx|css|md|json|yaml|yml)$
```

#### CI/CD Pipeline (.github/workflows/ci.yml)
```yaml
name: CI
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        node: [18, 20]
        python: [3.10, 3.11]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node }}
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}
      - run: npm ci
      - run: pip install -r requirements.txt
      - run: npm test
      - run: pytest

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm run lint
      - run: ruff check .
      
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm audit
      - run: pip-audit
      - run: semgrep --config=auto
```

### Project Labels
```json
[
  {"name": "bug", "color": "d73a4a", "description": "Something isn't working"},
  {"name": "feature", "color": "0075ca", "description": "New feature request"},
  {"name": "documentation", "color": "0052cc", "description": "Documentation improvements"},
  {"name": "good first issue", "color": "7057ff", "description": "Good for newcomers"},
  {"name": "help wanted", "color": "008672", "description": "Extra attention needed"},
  {"name": "priority-high", "color": "d93f0b", "description": "High priority"},
  {"name": "priority-medium", "color": "fbca04", "description": "Medium priority"},
  {"name": "priority-low", "color": "0e8a16", "description": "Low priority"},
  {"name": "size-s", "color": "ffffff", "description": "Small (1-2 days)"},
  {"name": "size-m", "color": "f0f0f0", "description": "Medium (3-5 days)"},
  {"name": "size-l", "color": "e0e0e0", "description": "Large (1+ week)"},
  {"name": "backend", "color": "fef2c0", "description": "Backend work"},
  {"name": "frontend", "color": "bfd4f2", "description": "Frontend work"},
  {"name": "database", "color": "5319e7", "description": "Database related"},
  {"name": "security", "color": "ee0000", "description": "Security issue"}
]
```

### CODEOWNERS
```
# .github/CODEOWNERS
# Maps code paths to responsible Claude agents
# Format: pattern @agent-name

# Global fallback - reviewer agent reviews everything
* @claude-reviewer

# Frontend - owned by frontend agent
/apps/ @claude-frontend
/packages/ui/ @claude-frontend
*.tsx @claude-frontend
*.jsx @claude-frontend
*.css @claude-frontend

# Backend - owned by backend agent
/services/ @claude-backend
/packages/api/ @claude-backend
*.py @claude-backend
requirements*.txt @claude-backend

# Database - owned by migrator agent
/services/*/migrations/ @claude-migrator
/alembic/ @claude-migrator
*.sql @claude-migrator
**/models/ @claude-migrator

# Testing - owned by test-runner agent
/tests/ @claude-test-runner
*.test.* @claude-test-runner
*.spec.* @claude-test-runner
jest.config.* @claude-test-runner
pytest.ini @claude-test-runner

# API Design - owned by api-contractor
/docs/api/ @claude-api-contractor
**/openapi.yaml @claude-api-contractor
**/graphql.schema @claude-api-contractor

# Documentation - owned by discovery-writer
/docs/*.md @claude-discovery-writer
README.md @claude-discovery-writer
CONTRIBUTING.md @claude-discovery-writer

# UX/Design - owned by journey-planner
/docs/ux/ @claude-journey-planner
/wireframes/ @claude-journey-planner

# Security - owned by security-scanner
.env.example @claude-security-scanner
/security/ @claude-security-scanner
*.pem @claude-security-scanner
*.key @claude-security-scanner

# DevOps - owned by scaffolder
/.github/ @claude-scaffolder
/.claude/ @claude-scaffolder
/scripts/ @claude-scaffolder
Dockerfile* @claude-scaffolder
docker-compose*.yml @claude-scaffolder
.gitlab-ci.yml @claude-scaffolder
```

### Monorepo Setup (pnpm)
```json
// package.json
{
  "name": "monorepo",
  "private": true,
  "workspaces": [
    "apps/*",
    "services/*",
    "packages/*"
  ],
  "scripts": {
    "dev": "turbo run dev",
    "build": "turbo run build",
    "test": "turbo run test",
    "lint": "turbo run lint",
    "format": "prettier --write .",
    "prepare": "husky install"
  },
  "devDependencies": {
    "turbo": "latest",
    "prettier": "latest",
    "husky": "latest",
    "lint-staged": "latest"
  }
}
```

### Environment Configuration
```bash
# .env.example
# Application
NODE_ENV=development
LOG_LEVEL=debug

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/db

# Redis
REDIS_URL=redis://localhost:6379

# Auth
JWT_SECRET=change-me-in-production
OAUTH_CLIENT_ID=
OAUTH_CLIENT_SECRET=

# External APIs
API_KEY=
WEBHOOK_SECRET=

# Monitoring
SENTRY_DSN=
DATADOG_API_KEY=
```

## Initialization Commands

```bash
# Create all directories
mkdir -p {apps,services,packages,scripts,docs}
mkdir -p .github/{workflows,ISSUE_TEMPLATE}
mkdir -p .claude/{subagents,hooks}

# Initialize git
git init
git add .
git commit -m "Initial commit"

# Set up pre-commit
pre-commit install
pre-commit run --all-files

# Create GitHub repo and push
gh repo create --private
git push -u origin main

# Set up branch protection
gh api repos/:owner/:repo/branches/main/protection --method PUT --field required_status_checks='{"strict":true,"contexts":["test","lint"]}'

# Add labels
gh label create --file .github/labels.json
```

Remember: Good scaffolding prevents technical debt!
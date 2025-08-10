#!/bin/bash

# Claude Code Memory Initialization Script
# Bootstraps memory structure following Claude Code best practices

set -e

echo "🚀 Initializing Claude Code Memory Structure"
echo "==========================================="

# Create directory structure
echo "📁 Creating memory directories..."
mkdir -p .claude/memory
mkdir -p .claude/archive
mkdir -p .claude/scripts
mkdir -p docs/adr

# Check if we're initializing a new project or updating existing
if [ -f "CLAUDE.md" ]; then
    echo "⚠️  CLAUDE.md already exists. Backing up to CLAUDE.md.backup"
    cp CLAUDE.md CLAUDE.md.backup
fi

# Create modular memory files
echo "📝 Creating modular memory files..."

# Setup & Configuration
cat > .claude/memory/setup.md << 'EOF'
## Environment Setup
- Python 3.11+ with virtual environment
- Node.js 20+ with npm/yarn
- PostgreSQL 15+ for primary database
- Redis for caching and sessions
- Docker for containerization

## Development Tools
- Git with conventional commits
- Make for task automation
- pytest for Python testing
- Jest for JavaScript testing
- Playwright for E2E testing
- ESLint + Black for formatting

## Local Development
- Use `make dev` to start all services
- Database runs on port 5432
- Redis runs on port 6379
- API runs on port 8000
- Frontend runs on port 3000

## Environment Variables
- Copy `.env.example` to `.env`
- Never commit `.env` file
- Use `ENVIRONMENT` to switch configs
- Secrets in GitHub Secrets for CI/CD
EOF

# Development Standards
cat > .claude/memory/standards.md << 'EOF'
## Code Standards

### Python
- Use type hints for all functions
- Follow PEP 8 style guide
- Docstrings for public functions
- Test coverage minimum 80%
- Use Pydantic for validation
- Async/await for I/O operations

### TypeScript/JavaScript
- TypeScript strict mode enabled
- Interfaces over types when possible
- No `any` types without justification
- React functional components only
- Custom hooks for logic reuse
- Props validation with Zod

### Testing
- Unit tests for business logic
- Integration tests for APIs
- E2E tests for critical paths
- Test files next to source: `*.test.ts`
- Use descriptive test names
- Mock external dependencies

### Git Conventions
- Semantic commit messages: `type(scope): description`
- Types: feat, fix, docs, style, refactor, test, chore
- Branch naming: `feature/`, `fix/`, `chore/`
- Squash commits on merge
- PR requires approval + passing CI

### Documentation
- README.md for every service
- API documentation via OpenAPI
- Inline comments for complex logic
- ADRs for architectural decisions
- Update docs with code changes

### Security
- No secrets in code
- Input validation on all endpoints
- SQL injection prevention via ORMs
- XSS prevention in frontend
- CORS properly configured
- Rate limiting on APIs
EOF

# Architecture Patterns
cat > .claude/memory/patterns.md << 'EOF'
## Architecture Patterns

### Backend Patterns
- Repository pattern for data access
- Service layer for business logic
- DTOs for API contracts
- Dependency injection for testing
- Circuit breakers for external services
- Event-driven for async operations

### Frontend Patterns
- Container/Presenter components
- Custom hooks for shared logic
- Context for global state
- Error boundaries for resilience
- Suspense for loading states
- Optimistic UI updates

### Database Patterns
- Migrations for schema changes
- Soft deletes with `deleted_at`
- UUID primary keys
- Audit fields: `created_at`, `updated_at`
- Indexes on foreign keys
- JSONB for flexible data

### API Patterns
- RESTful endpoints
- Consistent error responses
- Pagination on list endpoints
- Versioning via headers
- Rate limiting per user
- Request ID tracking

### Error Handling
- Error-first approach
- Structured error codes
- Graceful degradation
- Retry with exponential backoff
- Dead letter queues
- Comprehensive logging

### Performance Patterns
- Database connection pooling
- Redis caching strategy
- Lazy loading for assets
- Code splitting for bundles
- Image optimization
- CDN for static assets

### Deployment Patterns
- Feature flags for rollout
- Blue-green deployments
- Health check endpoints
- Graceful shutdown
- Rollback procedures
- Canary releases
EOF

# Team Conventions
cat > .claude/memory/team.md << 'EOF'
## Team Conventions

### Communication
- Daily standup at 10 AM
- PR reviews within 4 hours
- Slack for quick questions
- GitHub issues for bugs/features
- ADRs for architecture decisions

### Code Review
- At least 1 approval required
- Check for tests
- Verify documentation updates
- Consider performance impact
- Security review for auth/data

### Development Workflow
- Create feature branch from main
- Write tests first (TDD encouraged)
- Make small, focused commits
- Update documentation
- Create PR with description
- Merge after approval + CI pass

### Definition of Done
- [ ] Code complete with tests
- [ ] Documentation updated
- [ ] Peer reviewed
- [ ] CI/CD passing
- [ ] Deployed to staging
- [ ] Product owner approval

### On-Call Responsibilities
- Respond within 15 minutes
- Check runbooks first
- Escalate if needed
- Document incidents
- Create post-mortem for P0/P1

### Knowledge Sharing
- Tech talks monthly
- Document learnings in ADRs
- Share useful patterns
- Pair programming encouraged
- Rotate on-call duties
EOF

# Create main CLAUDE.md with imports
echo "📄 Creating CLAUDE.md with imports..."
cat > CLAUDE.md << 'EOF'
# Project Memory

## Setup & Configuration
@.claude/memory/setup.md

## Development Standards
@.claude/memory/standards.md

## Architecture Patterns
@.claude/memory/patterns.md

## Team Conventions
@.claude/memory/team.md

## Current State
@.claude/PROJECT_STATE.md

## Project Specific Instructions

### Technology Stack
- **Backend**: [Configure: FastAPI/Django/Express]
- **Database**: [Configure: PostgreSQL/MySQL/MongoDB]
- **Cache**: [Configure: Redis/Memcached]
- **Frontend**: [Configure: Next.js/React/Vue]
- **Deployment**: [Configure: AWS/GCP/Railway]

### Key Decisions
- **[Date]**: [Decision description]

### Important Context
- 🚨 **Critical**: [Critical information]
- ⚠️ **Warning**: [Important warning]
- 💡 **Tip**: [Helpful tip]
- 📌 **Remember**: [Important reminder]
- 🔒 **Security**: [Security note]

### Performance Targets
- API response time: p95 < [X]ms
- Database queries: < [X]ms
- Frontend Core Web Vitals: All green
- Test execution: < [X] minutes
- Build time: < [X] minutes

### Known Issues & Workarounds
- [Issue description] → [Workaround]

### Volumetrics Reference
@docs/VOLUMETRICS.md

### Architecture Decisions
@docs/adr/index.md

## Quick Commands Reference
- `make dev` - Start development environment
- `make test` - Run all tests
- `make deploy` - Deploy to production
- `/memory` - Edit memory files
- `/remember <note>` - Quick add to memory
- `/memory-review` - Review memory health
EOF

# Create PROJECT_STATE.md
echo "📋 Creating PROJECT_STATE.md..."
cat > .claude/PROJECT_STATE.md << 'EOF'
# Project State

## Current Tasks
- [ ] Review and customize memory files
- [ ] Update project-specific configuration in CLAUDE.md

## In Progress

## Completed Today
- ✅ Initialized Claude Code memory structure

## Blocked

## Next Up
- [ ] Configure technology stack in CLAUDE.md
- [ ] Add project-specific patterns

## Context for Next Session
- 📌 Memory structure initialized on $(date +%Y-%m-%d)
- 📌 Review .claude/memory/*.md files for customization
- 📌 Update project-specific sections in CLAUDE.md
EOF

# Create sample VOLUMETRICS.md
echo "📊 Creating sample VOLUMETRICS.md..."
cat > docs/VOLUMETRICS.md << 'EOF'
# System Volumetrics

## Database Metrics
| Table | Row Count | Growth Rate | Size | Index Size |
|-------|-----------|-------------|------|------------|
| users | 0 | 0/day | 0 KB | 0 KB |
| [table] | [count] | [rate] | [size] | [index] |

## API Metrics
| Endpoint | Requests/Day | P95 Latency | Error Rate |
|----------|-------------|-------------|------------|
| GET /api/health | 0 | 0ms | 0% |
| [endpoint] | [requests] | [latency] | [errors] |

## Performance Baselines
- Concurrent users: [target]
- Requests per second: [target]
- Database connections: [max]
- Memory usage: [max]
- CPU usage: [max]

## Growth Projections
- 3 months: [projection]
- 6 months: [projection]
- 1 year: [projection]

## Scaling Triggers
- When [metric] exceeds [threshold]
- When [metric] exceeds [threshold]

*Last updated: $(date +%Y-%m-%d)*
EOF

# Create ADR index
echo "📚 Creating ADR index..."
cat > docs/adr/index.md << 'EOF'
# Architecture Decision Records

## Active Decisions

*No ADRs yet. Create your first with `/adr <title>`*

## Deprecated Decisions

*None*

## How to use ADRs

1. Create new ADR: `/adr "Use PostgreSQL for primary database"`
2. Update ADR status: `/adr-update 001 accepted`
3. List ADRs: `/adr-list`

## ADR Template

Each ADR should follow this structure:
- **Title**: Clear, descriptive title
- **Status**: Draft/Proposed/Accepted/Deprecated/Superseded
- **Context**: What is the issue?
- **Decision**: What are we doing?
- **Consequences**: What are the trade-offs?
- **Alternatives**: What else was considered?
EOF

# Create a sample user-level memory file (optional)
echo "👤 Creating user-level memory template..."
cat > ~/.claude/memory.md.template << 'EOF'
# Personal Claude Code Memory

## My Preferences
- Editor: [Your editor]
- Terminal: [Your terminal]
- Git workflow: [Your preference]

## My Shortcuts
- [Custom shortcuts you use]

## My Patterns
- [Patterns you prefer]

## Learning Notes
- [Things you're learning]

*Note: Copy this to ~/.claude/memory.md and customize*
EOF

# Validate the setup
echo ""
echo "🔍 Validating memory structure..."
python3 .claude/scripts/resolve_imports.py --validate

# Test resolution
echo ""
echo "🧪 Testing import resolution..."
python3 .claude/scripts/resolve_imports.py -o .claude/CLAUDE_RESOLVED.md

# Generate summary
echo ""
echo "✅ Memory initialization complete!"
echo ""
echo "📊 Summary:"
echo "  - Created modular memory structure in .claude/memory/"
echo "  - Set up CLAUDE.md with @import syntax"
echo "  - Created PROJECT_STATE.md for working memory"
echo "  - Added sample VOLUMETRICS.md"
echo "  - Set up ADR structure"
echo ""
echo "📝 Next Steps:"
echo "  1. Edit CLAUDE.md to add your project-specific configuration"
echo "  2. Customize .claude/memory/*.md files for your team"
echo "  3. Use '/remember' to add quick notes"
echo "  4. Run '/memory-review' periodically to maintain memory health"
echo "  5. Consider copying ~/.claude/memory.md.template to ~/.claude/memory.md"
echo ""
echo "🎯 Quick Commands:"
echo "  /memory         - Edit memory files"
echo "  /remember       - Add quick note"
echo "  /memory-review  - Review memory health"
echo "  /memory-clean   - Clean old items"
echo "  /memory-search  - Search memory"
echo "  /memory-export  - Export resolved memory"
echo ""
echo "📚 Memory is now Claude Code compliant! 🎉"
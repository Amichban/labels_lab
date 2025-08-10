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
- **Backend**: FastAPI with async/await
- **Database**: PostgreSQL 15+ with asyncpg
- **Cache**: Redis for sessions and caching
- **Frontend**: Next.js 14 with App Router
- **Deployment**: Railway with GitHub Actions

### Key Decisions
- **2024-01-08**: Using PostgreSQL over MongoDB for ACID compliance
- **2024-01-07**: Adopted feature flags for safe deployments
- **2024-01-06**: Switched to trunk-based development

### Important Context
- 🚨 **Critical**: Always use connection pooling for database
- ⚠️ **Warning**: Migration scripts must be idempotent
- 💡 **Tip**: Use MSW for frontend development without backend
- 📌 **Remember**: Stripe Connect for marketplace, not standard
- 🔒 **Security**: Never log PII or sensitive data

### Performance Targets
- API response time: p95 < 200ms
- Database queries: < 50ms
- Frontend Core Web Vitals: All green
- Test execution: < 5 minutes
- Build time: < 3 minutes

### Known Issues & Workarounds
- Performance bottleneck in `/api/users` endpoint → Use pagination
- Flaky test in payment module → Retry with exponential backoff
- Memory leak in websocket handler → Restart every 24h

### Volumetrics Reference
@docs/VOLUMETRICS.md

### Architecture Decisions
@docs/adr/index.md

## Quick Commands Reference
- `make dev` - Start development environment
- `make test` - Run all tests
- `make deploy` - Deploy to production
- `/feature <name>` - Start new feature with flag
- `/incident <description>` - Create incident with analysis
- `/adr <title>` - Create architecture decision record
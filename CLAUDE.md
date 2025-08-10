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
- **Backend**: FastAPI with Python 3.11+, Pydantic 2.5
- **Database**: ClickHouse Cloud (quantx database)
- **Cache**: Redis with msgpack serialization
- **Frontend**: React with TypeScript (planned)
- **Deployment**: Docker Compose, Kubernetes ready

### Key Decisions
- **2025-01-10**: Use Label 11.a (Enhanced Triple Barrier) as highest priority
- **2025-01-10**: Multi-timeframe alignment: H4→H1, D→H4, W→D for path data
- **2025-01-10**: msgpack for Redis serialization (performance)

### Important Context
- 🚨 **Critical**: H4 timestamps MUST align to 1,5,9,13,17,21 UTC (not 0,4,8,12,16,20)
- ⚠️ **Warning**: Always use lower granularity for path-dependent calculations
- 💡 **Tip**: Cache hit rate should be >95% for recent 24 hours
- 📌 **Remember**: No look-ahead bias - strict temporal alignment required
- 🔒 **Security**: ClickHouse credentials in .env, never commit

### Performance Targets
- API response time: p95 < 100ms (incremental compute)
- Database queries: < 50ms
- Batch throughput: 1M+ candles/minute
- Cache hit rate: >95%
- Test execution: < 2 minutes
- Build time: < 1 minute

### Known Issues & Workarounds
- ClickHouse H4 alignment → Use TimestampAligner class
- DST transitions → All timestamps in UTC

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

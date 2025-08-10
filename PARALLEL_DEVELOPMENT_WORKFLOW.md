# Parallel Frontend/Backend Development Workflow

This document explains how to develop frontend and backend in parallel using MSW mocks, health checks, and continuous testing.

## 🎯 Overview

The parallel development workflow enables:
- **Independent development**: Frontend and backend teams work simultaneously
- **MSW mocks from OpenAPI**: Frontend uses auto-generated mocks
- **Health checks**: Production-ready monitoring from day one
- **Continuous testing**: Tests run automatically keeping code green
- **Type safety**: Shared types from OpenAPI specification

## 🚀 Quick Start

### One Command Setup

```bash
# Complete parallel development setup
/scaffold-parallel

# This creates:
# - MSW mocks from OpenAPI
# - Backend with health checks
# - Frontend with mock integration
# - Continuous test runner
```

### Start Development

```bash
# Terminal 1: Backend (real API)
cd services/api && make dev

# Terminal 2: Frontend (with mocks)
cd apps/web && NEXT_PUBLIC_USE_MOCKS=true npm run dev

# Terminal 3: Tests (continuous)
./scripts/test-watch.sh
```

## 🎭 MSW Mock Generation

### Generate from OpenAPI

```bash
# Generate MSW handlers
/generate-msw

# Or manually with Claude
claude api-contractor "Generate MSW handlers from docs/api/openapi.yaml with realistic fake data"
```

### Generated Mock Structure

```typescript
// apps/web/src/mocks/handlers.ts
import { rest } from 'msw';
import { faker } from '@faker-js/faker';

export const handlers = [
  rest.get('/api/users/:id', (req, res, ctx) => {
    return res(
      ctx.status(200),
      ctx.json({
        id: req.params.id,
        email: faker.internet.email(),
        name: faker.person.fullName(),
        createdAt: faker.date.past(),
      })
    );
  }),
  // ... more handlers
];
```

### Frontend Integration

```typescript
// apps/web/src/app/providers.tsx
'use client';

export function Providers({ children }) {
  useEffect(() => {
    if (process.env.NEXT_PUBLIC_USE_MOCKS === 'true') {
      const { worker } = require('../mocks/browser');
      worker.start();
    }
  }, []);
  
  return <>{children}</>;
}
```

## 🏥 Health Checks & Monitoring

### Backend Health Endpoints

```python
# Auto-generated at services/api/routes/health.py

@router.get("/healthz")
async def health_check():
    """Basic liveness check."""
    return {"status": "healthy"}

@router.get("/readyz")
async def readiness_check():
    """Readiness with dependency checks."""
    return {
        "ready": True,
        "checks": {
            "database": await check_database(),
            "redis": await check_redis(),
            "external_api": await check_external_api(),
        }
    }

@router.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    # Auto-configured with prometheus_client
```

### Logging Middleware

```python
# Auto-generated at services/api/middleware/logging.py

class LoggingMiddleware:
    """Structured logging for all requests."""
    
    async def dispatch(self, request, call_next):
        # Logs: method, path, status, duration
        # Adds: X-Request-ID, X-Process-Time headers
```

## 🧪 Continuous Testing

### Test Runner Configuration

```javascript
// jest.config.js - Multi-project setup
module.exports = {
  projects: [
    {
      displayName: 'backend',
      testMatch: ['services/api/**/*.test.py'],
    },
    {
      displayName: 'frontend',
      testMatch: ['apps/web/**/*.test.tsx'],
    },
  ],
  // Coverage requirements
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
    },
  },
};
```

### Watch Mode Script

```bash
# scripts/test-watch.sh
#!/bin/bash

# Runs all test suites in parallel watch mode
concurrently \
  "npm run test:unit -- --watch" \
  "npm run test:integration -- --watch" \
  "npm run test:e2e -- --ui"
```

### Pre-commit Hook

Tests run automatically before commits:
```bash
# .git/hooks/pre-commit
npm run test:unit || exit 1
```

## 📋 Development Workflow

### 1. API Design First

```bash
# Design API from requirements
/design-api

# This generates:
# - OpenAPI specification
# - Pydantic/Zod validators
# - TypeScript/Python SDKs
# - MSW mocks
# - Contract tests
```

### 2. Parallel Implementation

#### Frontend Team
```bash
# Use mocks for development
NEXT_PUBLIC_USE_MOCKS=true npm run dev

# Build UI components
# Test against mocks
# No backend dependency
```

#### Backend Team
```bash
# Implement API endpoints
# Use generated validators
# Run contract tests
# No frontend dependency
```

### 3. Integration

```bash
# Switch frontend to real API
NEXT_PUBLIC_USE_MOCKS=false npm run dev

# Run integration tests
npm run test:integration

# Run E2E tests
npm run test:e2e
```

## 🔄 CI/CD Pipeline

### Continuous Testing Workflow

```yaml
# .github/workflows/continuous-testing.yml

jobs:
  unit-tests:      # Run for frontend & backend
  integration-tests: # With real services
  e2e-tests:       # Using MSW mocks first
  contract-tests:  # Validate API matches spec
  msw-tests:       # Validate mocks match spec
```

### Test Matrix

| Test Type | Frontend | Backend | Mocks | Real API |
|-----------|----------|---------|-------|----------|
| Unit | ✅ | ✅ | ❌ | ❌ |
| Integration | ✅ | ✅ | ✅ | ✅ |
| E2E | ✅ | ❌ | ✅ | Optional |
| Contract | ❌ | ✅ | ❌ | ✅ |
| MSW | ✅ | ❌ | ✅ | ❌ |

## 🛠️ Commands Reference

### Scaffolding
```bash
/scaffold-parallel     # Complete setup
/scaffold-backend      # Backend only
/scaffold-frontend     # Frontend only
/test-runner          # Test configuration
```

### MSW & Mocking
```bash
/generate-msw         # Generate from OpenAPI
/api-mock 3001       # Start mock server
```

### Testing
```bash
/test                # Run all tests
./scripts/test-watch.sh  # Continuous mode
npm run test:unit    # Unit tests only
npm run test:e2e     # E2E tests
```

## 📁 Project Structure

```
my-app/
├── services/api/
│   ├── main.py              # FastAPI app with health checks
│   ├── middleware/           # Logging, errors, request ID
│   ├── routes/
│   │   ├── health.py        # /healthz, /readyz
│   │   └── ...
│   └── tests/
│       ├── unit/
│       └── integration/
├── apps/web/
│   ├── src/
│   │   ├── mocks/           # MSW handlers
│   │   │   ├── handlers.ts  # Auto-generated
│   │   │   ├── browser.ts   # Browser setup
│   │   │   └── server.ts    # Node setup
│   │   ├── lib/
│   │   │   └── api-client.ts # SDK with env switching
│   │   └── app/
│   │       └── providers.tsx # MSW integration
│   └── tests/
│       ├── unit/
│       └── e2e/
├── tests/
│   ├── contract/            # API contract tests
│   └── mocks/               # Mock validation
└── scripts/
    └── test-watch.sh        # Continuous runner
```

## 🎯 Best Practices

### 1. API-First Development
- Design OpenAPI spec before coding
- Generate everything from spec
- Keep spec as source of truth

### 2. Mock-First Frontend
- Develop UI with MSW mocks
- Test features independently
- Switch to real API later

### 3. Health-First Backend
- Add health checks immediately
- Include dependency checks
- Monitor from day one

### 4. Test-First Everything
- Write tests as you code
- Keep tests running continuously
- Block commits if tests fail

## 🚦 Environment Variables

### Frontend
```env
# .env.development
NEXT_PUBLIC_USE_MOCKS=true
NEXT_PUBLIC_API_URL=http://localhost:8000

# .env.production
NEXT_PUBLIC_USE_MOCKS=false
NEXT_PUBLIC_API_URL=https://api.example.com
```

### Backend
```env
# .env
LOG_LEVEL=INFO
ALLOWED_ORIGINS=["http://localhost:3000"]
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
```

## 🔍 Debugging

### MSW Not Working?
```javascript
// Check browser console
localStorage.debug = 'msw:*';

// Verify handlers loaded
console.log(worker.listHandlers());
```

### Health Check Failing?
```bash
# Check individual components
curl http://localhost:8000/healthz
curl http://localhost:8000/readyz

# Check logs
docker logs api-container
```

### Tests Failing?
```bash
# Run specific test
npm run test -- --testNamePattern="should create user"

# Debug mode
node --inspect-brk ./node_modules/.bin/jest
```

## 📚 Resources

- [MSW Documentation](https://mswjs.io/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [Jest Configuration](https://jestjs.io/docs/configuration)
- [Playwright Testing](https://playwright.dev/)

Remember: **Parallel development = Faster delivery!**
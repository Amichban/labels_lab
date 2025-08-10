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

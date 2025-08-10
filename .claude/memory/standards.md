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
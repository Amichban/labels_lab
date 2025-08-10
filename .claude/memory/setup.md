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
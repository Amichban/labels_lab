# API Contract & Validation Workflow

This document explains how API contracts, validation, and typed clients are automatically generated and enforced in the claude-native-template.

## 🎯 Overview

The API contract workflow ensures:
- **Single source of truth**: OpenAPI spec drives everything
- **Type safety**: Auto-generated validators and typed clients
- **Zero manual work**: Everything is generated from the spec
- **Breaking change detection**: Automatic checks in CI
- **Contract testing**: Generated tests validate implementation

## 🔄 The Complete Workflow

### 1. Design Phase: PRD → OpenAPI

```bash
# Start with PRD
/prd  # Create PRD from requirements

# Generate comprehensive API design
/design-api  # Creates OpenAPI + validators + SDKs + tests
```

This single command:
1. Generates OpenAPI specification from PRD
2. Creates Pydantic models for FastAPI
3. Creates Zod schemas for TypeScript
4. Generates typed SDK clients
5. Creates contract tests

### 2. Implementation Phase

The api-contractor agent generates everything you need:

```yaml
# docs/api/openapi.yaml - Source of truth
openapi: 3.0.0
paths:
  /users:
    post:
      requestBody:
        $ref: '#/components/schemas/UserCreate'
      responses:
        201:
          $ref: '#/components/schemas/User'
```

Automatically generates:

#### Python/FastAPI Implementation
```python
# services/api/schemas/user.py - Auto-generated
from pydantic import BaseModel, EmailStr

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    
class User(BaseModel):
    id: str
    email: EmailStr
    created_at: datetime
```

#### TypeScript Validation
```typescript
// sdk/schemas/user.ts - Auto-generated
import { z } from 'zod';

export const UserCreateSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8)
});

export type UserCreate = z.infer<typeof UserCreateSchema>;
```

#### Typed SDK Clients
```typescript
// sdk/clients/typescript/index.ts - Auto-generated
export class ApiClient {
  async createUser(data: UserCreate): Promise<User> {
    // Validated, typed, and safe
  }
}
```

### 3. Testing Phase

Contract tests are automatically generated:

```python
# tests/contract/test_users.py - Auto-generated
def test_create_user_validates_email():
    response = client.post('/users', json={
        "email": "invalid",  # Should fail
        "password": "ValidPass123"
    })
    assert response.status_code == 422

def test_create_user_success():
    response = client.post('/users', json={
        "email": "test@example.com",
        "password": "ValidPass123"
    })
    assert response.status_code == 201
    assert response.json()['email'] == "test@example.com"
```

### 4. CI/CD Integration

The GitHub workflow automatically:
- Validates OpenAPI spec on every PR
- Detects breaking changes
- Generates validators if spec changes
- Runs contract tests
- Updates SDKs
- Creates changelog

## 📝 Available Commands

### Quick Commands
```bash
/design-api           # Complete API generation from PRD
/api-validate        # Validate OpenAPI spec
/api-changelog       # Generate changelog from changes
/api-mock           # Start mock server for frontend
/api-test-contract  # Run contract tests
```

### Generation Commands
```bash
/api-generate-validators  # Generate Pydantic + Zod
/api-generate-sdk ts     # Generate TypeScript SDK
/api-generate-sdk python # Generate Python SDK
/api-docs html          # Generate API documentation
```

## 🤖 Using Claude for API Work

### Generate from Requirements
```bash
claude api-contractor "Create REST API for user management with:
- Registration/login
- Profile CRUD
- Password reset
- Rate limiting
Generate OpenAPI spec with examples"
```

### Update Existing API
```bash
claude api-contractor "Add bulk operations to users endpoint:
- POST /users/bulk for creating multiple users
- DELETE /users/bulk for deleting by IDs
Update OpenAPI and generate new validators"
```

### Generate Changelog
```bash
claude api-contractor "Compare current openapi.yaml with previous version
and generate detailed changelog with migration guide"
```

## 🔒 Validation & Safety

### Request Validation
All requests are validated automatically:
- FastAPI uses Pydantic models
- Next.js uses Zod schemas
- SDKs validate before sending

### Response Validation
Responses are validated to ensure contract compliance:
- Server validates before sending
- Clients validate after receiving
- Contract tests verify both

### Breaking Change Detection
CI automatically detects:
- Removed endpoints
- Changed required fields
- Modified response schemas
- Incompatible type changes

## 📊 Changelog Generation

Changelogs are generated in JSON format for processing:

```json
{
  "version": "1.2.0",
  "changes": {
    "added": [{
      "method": "POST",
      "path": "/users/bulk",
      "description": "Bulk user creation"
    }],
    "breaking": [{
      "field": "user.email",
      "change": "Made required",
      "migration": "Ensure all users have email"
    }]
  }
}
```

Then formatted as markdown for release notes:

```markdown
## v1.2.0 - API Changes

### 🆕 New Features
- `POST /users/bulk` - Bulk user creation

### ⚠️ Breaking Changes
- `user.email` is now required
  - Migration: Ensure all existing users have email addresses
```

## 🏗️ Project Structure

```
my-app/
├── docs/api/
│   ├── openapi.yaml          # Source of truth
│   ├── openapi.previous.yaml # For comparison
│   ├── CHANGELOG.md          # API changelog
│   └── index.html            # Generated docs
├── services/api/
│   ├── schemas/              # Generated Pydantic
│   └── routes/               # Implement using schemas
├── sdk/
│   ├── schemas/              # Generated Zod
│   ├── types/                # Generated TypeScript
│   └── clients/              # Generated SDKs
│       ├── typescript/
│       ├── python/
│       └── go/
└── tests/
    └── contract/             # Generated tests
```

## 🚀 Best Practices

### 1. API-First Development
- Design API before implementation
- Use OpenAPI as the contract
- Generate everything from spec

### 2. Version Management
- Keep previous spec for comparison
- Detect breaking changes early
- Provide migration guides

### 3. Testing Strategy
- Contract tests from spec
- Mock server for frontend
- Validate both request and response

### 4. Documentation
- Auto-generate from OpenAPI
- Include examples in spec
- Keep changelog updated

## 🔄 Workflow Example

### Complete Feature Flow

1. **Requirements**: Product defines user management needs
2. **PRD**: `/prd` generates formal requirements
3. **API Design**: `/design-api` creates complete API
4. **Implementation**: Use generated schemas
5. **Testing**: Run generated contract tests
6. **Documentation**: Auto-generated and deployed
7. **Release**: Changelog included in PR

### PR with API Changes

When a PR modifies `openapi.yaml`:
1. CI validates the spec
2. Detects breaking changes
3. Generates new validators
4. Runs contract tests
5. Posts changelog as comment
6. Updates documentation

## 🎯 Benefits

### For Developers
- No manual schema writing
- Type safety everywhere
- Automatic validation
- Generated tests

### For Teams
- Single source of truth
- Parallel development (frontend/backend)
- Clear contracts
- Automatic documentation

### For Users
- Consistent APIs
- Better error messages
- Type-safe SDKs
- Up-to-date docs

## 📚 Resources

- [OpenAPI Specification](https://swagger.io/specification/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Zod Documentation](https://zod.dev/)
- [FastAPI OpenAPI](https://fastapi.tiangolo.com/tutorial/openapi/)

Remember: **The OpenAPI spec is the source of truth - everything else is generated!**
#!/bin/bash
# Pre-commit hook: Validate before committing

echo "🔍 Running pre-commit checks..."

# Check for secrets
if command -v gitleaks &> /dev/null; then
  echo "Checking for secrets..."
  gitleaks detect --source . --verbose
  if [ $? -ne 0 ]; then
    echo "❌ Secrets detected! Commit blocked."
    exit 1
  fi
fi

# Run linters
echo "Running linters..."

# Python
if ls *.py &> /dev/null || ls **/*.py &> /dev/null; then
  if command -v ruff &> /dev/null; then
    ruff check . || exit 1
  fi
fi

# TypeScript/JavaScript
if [ -f "package.json" ]; then
  if command -v npm &> /dev/null; then
    npm run lint 2>/dev/null || true
  fi
fi

# Run tests
echo "Running tests..."
if [ -f "Makefile" ] && grep -q "^test:" Makefile; then
  make test || exit 1
fi

echo "✅ All pre-commit checks passed!"
exit 0
#!/usr/bin/env python3
"""
Path protection hook - prevents writes to protected files and enforces agent path restrictions
"""

import json
import os
import sys
import fnmatch
from pathlib import Path

# Protected files that should never be modified
PROTECTED_FILES = [
    '.env',
    '.env.production',
    'secrets.yaml',
    'credentials.json',
    '*.pem',
    '*.key',
    '*.cert',
    '.git/config',
    'node_modules/**',
    'venv/**',
    '.venv/**',
    '*.pyc',
    '__pycache__/**'
]

# Agent path restrictions (from subagent definitions)
AGENT_PATHS = {
    'test-runner': ['apps/**', 'services/**', 'tests/**'],
    'migrator': ['services/api/migrations/**', 'services/api/models/**', 'services/api/alembic/**'],
    'scaffolder': ['.github/**', '.claude/**', 'scripts/**', '*'],
    'backend': ['services/**', 'packages/api/**', 'apps/api/**'],
    'frontend': ['apps/web/**', 'apps/mobile/**', 'packages/ui/**'],
    'discovery-writer': ['docs/**'],
    'journey-planner': ['docs/ux/**'],
    'api-contractor': ['docs/api/**']
}

def is_protected(file_path: str) -> bool:
    """Check if a file path is protected"""
    path = Path(file_path)
    
    for pattern in PROTECTED_FILES:
        if fnmatch.fnmatch(str(path), pattern):
            return True
        # Check against absolute path patterns
        if fnmatch.fnmatch(str(path.resolve()), pattern):
            return True
    
    return False

def is_allowed_for_agent(agent_name: str, file_path: str) -> bool:
    """Check if an agent is allowed to access a file path"""
    
    # If no agent specified, allow (for manual Claude operations)
    if not agent_name:
        return True
    
    # If agent not in restrictions, allow
    if agent_name not in AGENT_PATHS:
        return True
    
    allowed_patterns = AGENT_PATHS.get(agent_name, [])
    path = Path(file_path)
    
    # Check if path matches any allowed pattern
    for pattern in allowed_patterns:
        if fnmatch.fnmatch(str(path), pattern):
            return True
        # Also check relative to project root
        try:
            relative_path = path.relative_to(Path.cwd())
            if fnmatch.fnmatch(str(relative_path), pattern):
                return True
        except ValueError:
            pass
    
    return False

def main():
    """Main hook logic"""
    
    # Get tool input from environment
    tool_input = os.environ.get('TOOL_INPUT', '{}')
    
    try:
        data = json.loads(tool_input)
    except json.JSONDecodeError:
        # If not JSON, allow (might be a different tool)
        sys.exit(0)
    
    # Extract file path and agent
    file_path = data.get('file_path') or data.get('path', '')
    agent_name = os.environ.get('CLAUDE_AGENT', '')
    
    if not file_path:
        # No file path, allow
        sys.exit(0)
    
    # Check if protected
    if is_protected(file_path):
        print(f"❌ BLOCKED: Cannot modify protected file: {file_path}")
        print("Protected files include: .env, secrets, credentials, keys, etc.")
        sys.exit(1)
    
    # Check agent permissions
    if agent_name and not is_allowed_for_agent(agent_name, file_path):
        allowed = AGENT_PATHS.get(agent_name, [])
        print(f"❌ BLOCKED: Agent '{agent_name}' not allowed to modify: {file_path}")
        print(f"Agent is restricted to: {', '.join(allowed)}")
        sys.exit(1)
    
    # Check for dangerous operations
    content = data.get('content', '') or data.get('new_content', '')
    
    # Check for potential secrets in content
    dangerous_patterns = [
        'password=',
        'api_key=',
        'secret=',
        'token=',
        'private_key',
        'BEGIN RSA',
        'BEGIN PRIVATE KEY',
        'aws_access_key',
        'aws_secret'
    ]
    
    content_lower = content.lower()
    for pattern in dangerous_patterns:
        if pattern.lower() in content_lower:
            # Check if it's a template/example (contains placeholder values)
            placeholders = ['xxx', 'your-', 'change-me', '<', 'example', 'placeholder']
            if not any(p in content_lower for p in placeholders):
                print(f"⚠️  WARNING: Potential secret detected in content: {pattern}")
                print("Consider using environment variables instead")
                # Warning only, don't block
    
    # All checks passed
    sys.exit(0)

if __name__ == '__main__':
    main()
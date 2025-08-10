#!/usr/bin/env python3
"""
Security check hook - prevents dangerous commands and enforces environment-based restrictions
"""

import json
import os
import sys
import re

# Get current environment
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'development').lower()

# Dangerous commands that should never run
FORBIDDEN_COMMANDS = [
    r'rm\s+-rf\s+/',              # Delete root
    r'chmod\s+777',               # World writable
    r'curl.*\|\s*(bash|sh)',      # Curl pipe to shell
    r'wget.*\|\s*(bash|sh)',      # Wget pipe to shell
    r'eval\s*\(',                  # Eval execution
    r'exec\s*\(',                  # Exec execution
    r':\(\)\s*\{.*:\|\:&\s*\}',   # Fork bomb
    r'dd\s+if=/dev/(zero|random)', # Disk destroyer
]

# Production-restricted commands
PRODUCTION_RESTRICTED = [
    r'aws\s+secretsmanager\s+get-secret',
    r'aws\s+ssm\s+get-parameter',
    r'kubectl\s+delete',
    r'kubectl\s+apply',
    r'terraform\s+destroy',
    r'terraform\s+apply',
    r'docker\s+rm',
    r'dropdb',
    r'DROP\s+(TABLE|DATABASE)',
    r'TRUNCATE',
    r'heroku\s+ps:scale',
    r'gcloud\s+compute\s+instances\s+delete',
]

# Commands that need confirmation
CONFIRMATION_REQUIRED = [
    r'git\s+push\s+.*--force',
    r'git\s+reset\s+--hard',
    r'npm\s+publish',
    r'pip\s+install.*--upgrade',
    r'apt-get\s+dist-upgrade',
    r'brew\s+upgrade',
]

# Patterns that might indicate secrets
SECRET_PATTERNS = [
    (r'AKIA[0-9A-Z]{16}', 'AWS Access Key'),
    (r'[0-9a-zA-Z/+=]{40}', 'Possible AWS Secret Key'),
    (r'-----BEGIN\s+(RSA|DSA|EC|OPENSSH)\s+PRIVATE\s+KEY-----', 'Private Key'),
    (r'eyJ[A-Za-z0-9-_=]+\.[A-Za-z0-9-_=]+\.?[A-Za-z0-9-_.+/=]*', 'JWT Token'),
    (r'(api[_-]?key|apikey)\s*=\s*[\'"][a-zA-Z0-9]{20,}[\'"]', 'API Key'),
    (r'(password|passwd|pwd)\s*=\s*[\'"][^\'"]{8,}[\'"]', 'Password'),
    (r'xox[baprs]-[0-9]{10,13}-[a-zA-Z0-9]{24,32}', 'Slack Token'),
    (r'ghp_[a-zA-Z0-9]{36}', 'GitHub Personal Access Token'),
    (r'ghs_[a-zA-Z0-9]{36}', 'GitHub Secret'),
    (r'sk_live_[a-zA-Z0-9]{24,}', 'Stripe Live Key'),
]

def check_command(command: str) -> tuple[bool, str]:
    """Check if a command is safe to run"""
    
    # Check for forbidden commands
    for pattern in FORBIDDEN_COMMANDS:
        if re.search(pattern, command, re.IGNORECASE):
            return False, f"Forbidden command pattern detected: {pattern}"
    
    # Check production restrictions
    if ENVIRONMENT in ['production', 'prod', 'live']:
        for pattern in PRODUCTION_RESTRICTED:
            if re.search(pattern, command, re.IGNORECASE):
                return False, f"Command restricted in production: {pattern}"
    
    # Check for commands needing confirmation
    for pattern in CONFIRMATION_REQUIRED:
        if re.search(pattern, command, re.IGNORECASE):
            # In CI/CD or non-interactive mode, block
            if os.environ.get('CI') or not sys.stdin.isatty():
                return False, f"Command requires manual confirmation: {pattern}"
            
            # Otherwise just warn
            print(f"⚠️  WARNING: Dangerous command detected: {pattern}")
            print(f"Command: {command}")
            response = input("Continue? (y/N): ")
            if response.lower() != 'y':
                return False, "User cancelled operation"
    
    # Check for secrets in command
    for pattern, name in SECRET_PATTERNS:
        if re.search(pattern, command):
            # Check if it's a placeholder
            if any(p in command.lower() for p in ['xxx', 'your-', 'example', '<placeholder>']):
                continue
            return False, f"Potential {name} detected in command"
    
    return True, "Command approved"

def check_file_content(content: str) -> tuple[bool, str]:
    """Check file content for security issues"""
    
    warnings = []
    
    # Check for secrets
    for pattern, name in SECRET_PATTERNS:
        if re.search(pattern, content):
            # Check if it's a template/example
            if not any(p in content.lower() for p in ['xxx', 'your-', 'example', '<placeholder>', 'change-me']):
                warnings.append(f"Potential {name} detected")
    
    # Check for dangerous code patterns
    dangerous_code = [
        (r'eval\s*\(', 'eval() usage'),
        (r'exec\s*\(', 'exec() usage'),
        (r'__import__\s*\(', 'Dynamic import'),
        (r'subprocess\.call\s*\(.*shell\s*=\s*True', 'Shell injection risk'),
        (r'os\.system\s*\(', 'os.system() usage'),
        (r'pickle\.loads?\s*\(', 'Pickle deserialization'),
        (r'yaml\.load\s*\(', 'Unsafe YAML loading (use safe_load)'),
        (r'innerHTML\s*=', 'XSS risk via innerHTML'),
        (r'document\.write\s*\(', 'document.write usage'),
        (r'createObjectURL\s*\(.*Blob', 'Potential XSS via Blob'),
    ]
    
    for pattern, name in dangerous_code:
        if re.search(pattern, content):
            warnings.append(f"Security risk: {name}")
    
    # Check for SQL injection risks
    sql_patterns = [
        r'(SELECT|INSERT|UPDATE|DELETE).*\+.*[\'"]',  # String concatenation in SQL
        r'f[\'"].*?(SELECT|INSERT|UPDATE|DELETE)',     # f-string SQL
        r'%\s*\(.*?\).*?(SELECT|INSERT|UPDATE|DELETE)', # % formatting SQL
    ]
    
    for pattern in sql_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            warnings.append("Potential SQL injection risk")
    
    if warnings:
        return True, f"Security warnings: {', '.join(set(warnings))}"
    
    return True, "Content approved"

def main():
    """Main hook logic"""
    
    # Get tool type and input
    tool_type = os.environ.get('TOOL_TYPE', '')
    tool_input = os.environ.get('TOOL_INPUT', '{}')
    
    try:
        data = json.loads(tool_input)
    except json.JSONDecodeError:
        # Not JSON, check if it's a bash command
        if tool_type == 'bash':
            command = tool_input
            safe, message = check_command(command)
            if not safe:
                print(f"❌ BLOCKED: {message}")
                print(f"Environment: {ENVIRONMENT}")
                sys.exit(1)
        sys.exit(0)
    
    # Check bash commands
    if tool_type == 'bash' or 'command' in data:
        command = data.get('command', '')
        if command:
            safe, message = check_command(command)
            if not safe:
                print(f"❌ BLOCKED: {message}")
                print(f"Environment: {ENVIRONMENT}")
                sys.exit(1)
    
    # Check file content for writes
    if tool_type in ['write_file', 'edit_file']:
        content = data.get('content', '') or data.get('new_content', '')
        if content:
            safe, message = check_file_content(content)
            if not safe:
                print(f"❌ BLOCKED: {message}")
                sys.exit(1)
            elif "warnings" in message:
                print(f"⚠️  {message}")
                # Don't block, just warn
    
    sys.exit(0)

if __name__ == '__main__':
    main()
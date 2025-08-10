#!/bin/bash
# Pre-task hook: Set up session tracking

# Create session directory
SESSION_ID=$(date +%Y%m%d_%H%M%S)
SESSION_DIR=".claude/sessions/$SESSION_ID"
mkdir -p "$SESSION_DIR"

# Export for other hooks
export CLAUDE_SESSION_ID="$SESSION_ID"
export CLAUDE_SESSION_DIR="$SESSION_DIR"

# Check git status
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
  echo "⚠️  Warning: Uncommitted changes detected"
  echo "Consider committing or stashing before running agent"
fi

# Snapshot current state
if [ -d ".git" ]; then
  cat > "$SESSION_DIR/snapshot.json" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "git_branch": "$(git branch --show-current 2>/dev/null || echo 'unknown')",
  "git_head": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
  "working_files": $(git status --porcelain 2>/dev/null | wc -l || echo 0)
}
EOF
fi

# Log task start
echo "[$(date)] Session $SESSION_ID started" >> .claude/audit.log

exit 0
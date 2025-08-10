#!/bin/bash
# Post-task hook: Save session results

SESSION_DIR="${CLAUDE_SESSION_DIR:-.claude/sessions/current}"

# Save task output
if [ -n "$TOOL_OUTPUT" ]; then
  echo "$TOOL_OUTPUT" > "$SESSION_DIR/output.txt"
fi

# Generate diff if files changed
if [ -d ".git" ]; then
  git diff > "$SESSION_DIR/changes.diff" 2>/dev/null
  
  # If diff exists, save it
  if [ -s "$SESSION_DIR/changes.diff" ]; then
    echo "📝 Changes saved to: $SESSION_DIR/changes.diff"
    echo "Review with: git diff"
    echo "Apply with: git apply $SESSION_DIR/changes.diff"
  fi
fi

# Update audit log
echo "[$(date)] Session completed: $SESSION_DIR" >> .claude/audit.log

exit 0
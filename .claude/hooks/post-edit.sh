#!/bin/bash
# Post-edit hook: Actions after editing files

FILE_PATH=$(echo "$TOOL_OUTPUT" | jq -r '.file_path // ""')

if [ -z "$FILE_PATH" ]; then
  exit 0
fi

# Auto-update imports for Python
if [[ $FILE_PATH == *.py ]]; then
  if command -v isort &> /dev/null; then
    isort "$FILE_PATH" 2>/dev/null || true
  fi
fi

# Update TypeScript types
if [[ $FILE_PATH == *.ts || $FILE_PATH == *.tsx ]]; then
  if [ -f "package.json" ] && command -v npm &> /dev/null; then
    # Generate types if needed
    npm run type-check 2>/dev/null || true
  fi
fi

# Log changes for audit
echo "[$(date)] Edited: $FILE_PATH" >> .claude/audit.log

exit 0
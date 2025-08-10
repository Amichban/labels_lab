#!/bin/bash
# Pre-write hook: Format code before writing

FILE_PATH=$(echo "$TOOL_INPUT" | jq -r '.file_path // .path // ""')

if [ -z "$FILE_PATH" ]; then
  exit 0
fi

# Python - black + isort
if [[ $FILE_PATH == *.py ]]; then
  if command -v black &> /dev/null; then
    black "$FILE_PATH" 2>/dev/null || true
  fi
  if command -v isort &> /dev/null; then
    isort "$FILE_PATH" 2>/dev/null || true
  fi
fi

# TypeScript/JavaScript - prettier
if [[ $FILE_PATH == *.ts || $FILE_PATH == *.tsx || $FILE_PATH == *.js || $FILE_PATH == *.jsx ]]; then
  if command -v prettier &> /dev/null; then
    prettier --write "$FILE_PATH" 2>/dev/null || true
  fi
fi

# JSON - jq
if [[ $FILE_PATH == *.json ]]; then
  if command -v jq &> /dev/null; then
    jq '.' "$FILE_PATH" > "$FILE_PATH.tmp" 2>/dev/null && mv "$FILE_PATH.tmp" "$FILE_PATH" || true
  fi
fi

# YAML - yq
if [[ $FILE_PATH == *.yaml || $FILE_PATH == *.yml ]]; then
  if command -v yq &> /dev/null; then
    yq eval '.' "$FILE_PATH" > "$FILE_PATH.tmp" 2>/dev/null && mv "$FILE_PATH.tmp" "$FILE_PATH" || true
  fi
fi

exit 0
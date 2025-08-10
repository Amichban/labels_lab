#!/bin/bash
# Setup Claude CLI mock for GitHub Actions

set -e

# Create a claude command in PATH
mkdir -p $HOME/.local/bin
cp .github/scripts/claude-mock.sh $HOME/.local/bin/claude
chmod +x $HOME/.local/bin/claude
export PATH="$HOME/.local/bin:$PATH"

echo "Claude mock CLI installed successfully"
echo "PATH=$PATH" >> $GITHUB_ENV
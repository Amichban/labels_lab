#!/usr/bin/env python3
"""Format API changelog from JSON to Markdown."""

import json
import datetime
import os

def format_changelog():
    """Generate markdown changelog from JSON API changes."""
    with open('api-changes.json') as f:
        changes = json.load(f)

    # Generate markdown changelog
    pr_number = os.environ.get('GITHUB_PR_NUMBER', 'N/A')
    changelog = f"""## API Changes - PR #{pr_number}
*Generated: {datetime.datetime.now().isoformat()}*

"""

    if changes.get('breaking_changes'):
        changelog += "### 🚨 Breaking Changes\n"
        for change in changes['breaking_changes']:
            changelog += f"- {change}\n"
        changelog += "\n"

    if changes.get('added'):
        changelog += "### 🆕 New Endpoints\n"
        for endpoint in changes['added']:
            method = endpoint['method']
            path = endpoint['path']
            desc = endpoint['description']
            changelog += f"- `{method} {path}` - {desc}\n"
        changelog += "\n"

    if changes.get('modified'):
        changelog += "### 📝 Modified Endpoints\n"
        for endpoint in changes['modified']:
            method = endpoint['method']
            path = endpoint['path']
            desc = endpoint['description']
            changelog += f"- `{method} {path}` - {desc}\n"
            for detail in endpoint.get('details', []):
                changelog += f"  - {detail}\n"
        changelog += "\n"

    if changes.get('deprecated'):
        changelog += "### 🗑️ Deprecated Endpoints\n"
        for endpoint in changes['deprecated']:
            method = endpoint['method']
            path = endpoint['path']
            reason = endpoint.get('reason', 'Deprecated')
            changelog += f"- `{method} {path}` - {reason}\n"
        changelog += "\n"

    if changes.get('migration_guide'):
        changelog += "### 📋 Migration Guide\n"
        changelog += changes['migration_guide'] + "\n"

    print(changelog)

    # Save to file
    with open('api-changelog.md', 'w') as f:
        f.write(changelog)

if __name__ == "__main__":
    format_changelog()
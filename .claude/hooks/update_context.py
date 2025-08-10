#!/usr/bin/env python3
"""
Context update hook - Automatically maintains PROJECT_STATE.md and context.
Runs after significant actions to keep memory fresh.
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import re


class ContextManager:
    """Manages project context and memory."""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.claude_dir = self.project_root / '.claude'
        self.project_state_file = self.claude_dir / 'PROJECT_STATE.md'
        self.claude_md = self.project_root / 'CLAUDE.md'
        self.volumetrics_file = self.project_root / 'docs' / 'VOLUMETRICS.md'
        self.context_cache = self.claude_dir / 'context_cache.json'
        
    def update_project_state(self, action: str, details: str = None):
        """Update PROJECT_STATE.md with recent action."""
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        
        # Read current state
        if self.project_state_file.exists():
            with open(self.project_state_file, 'r') as f:
                content = f.read()
        else:
            content = self._create_initial_state()
        
        # Parse sections
        sections = self._parse_sections(content)
        
        # Update based on action
        if action == 'task_completed':
            self._move_task_to_completed(sections, details)
        elif action == 'task_started':
            self._add_to_in_progress(sections, details)
        elif action == 'blocked':
            self._add_blocker(sections, details)
        elif action == 'file_changed':
            self._track_file_change(sections, details)
        elif action == 'decision_made':
            self._record_decision(sections, details)
        elif action == 'test_run':
            self._update_test_status(sections, details)
        
        # Add timestamp
        sections['last_updated'] = f"*Last updated: {timestamp}*"
        
        # Write back
        self._write_sections(sections)
        
    def _parse_sections(self, content: str) -> dict:
        """Parse PROJECT_STATE.md into sections."""
        sections = {
            'header': '# Project State\n',
            'current_tasks': '',
            'in_progress': '',
            'completed_today': '',
            'blocked': '',
            'next_up': '',
            'context_notes': '',
            'last_updated': ''
        }
        
        current_section = None
        current_content = []
        
        for line in content.split('\n'):
            if line.startswith('## Current Tasks'):
                current_section = 'current_tasks'
                current_content = [line]
            elif line.startswith('## In Progress'):
                current_section = 'in_progress'
                current_content = [line]
            elif line.startswith('## Completed Today'):
                current_section = 'completed_today'
                current_content = [line]
            elif line.startswith('## Blocked'):
                current_section = 'blocked'
                current_content = [line]
            elif line.startswith('## Next Up'):
                current_section = 'next_up'
                current_content = [line]
            elif line.startswith('## Context'):
                current_section = 'context_notes'
                current_content = [line]
            elif current_section:
                current_content.append(line)
                sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    def _write_sections(self, sections: dict):
        """Write sections back to PROJECT_STATE.md."""
        content = sections['header']
        
        for key in ['current_tasks', 'in_progress', 'completed_today', 
                   'blocked', 'next_up', 'context_notes']:
            if sections.get(key):
                content += '\n' + sections[key] + '\n'
        
        content += '\n' + sections.get('last_updated', '')
        
        with open(self.project_state_file, 'w') as f:
            f.write(content)
    
    def _add_to_in_progress(self, sections: dict, task: str):
        """Add task to in progress section."""
        if '## In Progress' not in sections['in_progress']:
            sections['in_progress'] = '## In Progress\n'
        
        sections['in_progress'] += f"- [ ] {task} (started: {datetime.now().strftime('%H:%M')})\n"
    
    def _move_task_to_completed(self, sections: dict, task: str):
        """Move task from in progress to completed."""
        # Remove from in progress
        if task and sections.get('in_progress'):
            lines = sections['in_progress'].split('\n')
            lines = [l for l in lines if task not in l]
            sections['in_progress'] = '\n'.join(lines)
        
        # Add to completed
        if '## Completed Today' not in sections['completed_today']:
            sections['completed_today'] = '## Completed Today\n'
        
        sections['completed_today'] += f"- ✅ {task} ({datetime.now().strftime('%H:%M')})\n"
    
    def _add_blocker(self, sections: dict, blocker: str):
        """Add a blocker."""
        if '## Blocked' not in sections['blocked']:
            sections['blocked'] = '## Blocked\n'
        
        sections['blocked'] += f"- 🚫 {blocker} (since: {datetime.now().strftime('%Y-%m-%d %H:%M')})\n"
    
    def _track_file_change(self, sections: dict, file_path: str):
        """Track file changes in context."""
        if '## Context' not in sections.get('context_notes', ''):
            sections['context_notes'] = '## Context for Next Session\n'
        
        # Add file change note
        change_type = self._detect_change_type(file_path)
        sections['context_notes'] += f"- {change_type}: {file_path}\n"
    
    def _detect_change_type(self, file_path: str) -> str:
        """Detect type of change from file path."""
        if 'test' in file_path.lower():
            return '🧪 Test updated'
        elif 'api' in file_path.lower():
            return '🔌 API changed'
        elif 'model' in file_path.lower() or 'schema' in file_path.lower():
            return '🗄️ Schema modified'
        elif 'migration' in file_path.lower():
            return '🔄 Migration added'
        else:
            return '📝 File modified'
    
    def _record_decision(self, sections: dict, decision: str):
        """Record an architectural decision."""
        if '## Context' not in sections.get('context_notes', ''):
            sections['context_notes'] = '## Context for Next Session\n'
        
        sections['context_notes'] += f"- 💡 Decision: {decision}\n"
    
    def _update_test_status(self, sections: dict, status: str):
        """Update test status in context."""
        if '## Context' not in sections.get('context_notes', ''):
            sections['context_notes'] = '## Context for Next Session\n'
        
        # Parse test results
        if 'passed' in status.lower():
            emoji = '✅'
        elif 'failed' in status.lower():
            emoji = '❌'
        else:
            emoji = '⚠️'
        
        sections['context_notes'] += f"- {emoji} Tests: {status}\n"
    
    def _create_initial_state(self) -> str:
        """Create initial PROJECT_STATE.md content."""
        return """# Project State

## Current Tasks
- [ ] 

## In Progress

## Completed Today

## Blocked

## Next Up

## Context for Next Session
"""
    
    def summarize_git_changes(self, since_hours: int = 24):
        """Summarize recent git changes for context."""
        try:
            # Get recent commits
            result = subprocess.run(
                ['git', 'log', f'--since={since_hours} hours ago', 
                 '--pretty=format:%h %s', '--stat'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return result.stdout
        except:
            pass
        return ""
    
    def update_claude_md(self, key: str, value: str):
        """Update CLAUDE.md with new information."""
        if not self.claude_md.exists():
            return
        
        with open(self.claude_md, 'r') as f:
            content = f.read()
        
        # Add or update section
        if f"## {key}" in content:
            # Update existing section
            lines = content.split('\n')
            in_section = False
            new_lines = []
            
            for line in lines:
                if line == f"## {key}":
                    in_section = True
                    new_lines.append(line)
                    new_lines.append(value)
                elif line.startswith('## ') and in_section:
                    in_section = False
                    new_lines.append(line)
                elif not in_section:
                    new_lines.append(line)
            
            content = '\n'.join(new_lines)
        else:
            # Add new section
            content += f"\n## {key}\n{value}\n"
        
        with open(self.claude_md, 'w') as f:
            f.write(content)
    
    def cache_context(self, context_type: str, data: dict):
        """Cache frequently used context."""
        cache = {}
        
        if self.context_cache.exists():
            with open(self.context_cache, 'r') as f:
                cache = json.load(f)
        
        cache[context_type] = {
            'data': data,
            'timestamp': datetime.now().isoformat(),
            'expires': (datetime.now().timestamp() + 3600)  # 1 hour cache
        }
        
        with open(self.context_cache, 'w') as f:
            json.dump(cache, f, indent=2)
    
    def get_cached_context(self, context_type: str) -> dict:
        """Get cached context if still valid."""
        if not self.context_cache.exists():
            return None
        
        with open(self.context_cache, 'r') as f:
            cache = json.load(f)
        
        if context_type in cache:
            if cache[context_type]['expires'] > datetime.now().timestamp():
                return cache[context_type]['data']
        
        return None
    
    def prune_old_context(self, max_age_days: int = 7):
        """Remove old context to keep memory fresh."""
        # Clean up completed tasks older than max_age
        if self.project_state_file.exists():
            with open(self.project_state_file, 'r') as f:
                content = f.read()
            
            # Remove old completed items
            lines = content.split('\n')
            new_lines = []
            
            for line in lines:
                # Check if line has old date
                if '202' in line:  # Has a date
                    # Extract date and check age
                    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', line)
                    if date_match:
                        date_str = date_match.group(1)
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                        age = (datetime.now() - date_obj).days
                        
                        if age <= max_age_days or '## ' in line:
                            new_lines.append(line)
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
            
            with open(self.project_state_file, 'w') as f:
                f.write('\n'.join(new_lines))
    
    def generate_session_summary(self) -> str:
        """Generate a summary of the current session."""
        summary = f"## Session Summary - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        
        # Get git changes
        git_changes = self.summarize_git_changes(since_hours=8)
        if git_changes:
            summary += f"### Code Changes\n```\n{git_changes}\n```\n\n"
        
        # Get current state
        if self.project_state_file.exists():
            with open(self.project_state_file, 'r') as f:
                state = f.read()
                
            # Extract key sections
            if '## Completed Today' in state:
                completed = state.split('## Completed Today')[1].split('##')[0]
                summary += f"### Completed\n{completed}\n"
            
            if '## Blocked' in state:
                blocked = state.split('## Blocked')[1].split('##')[0]
                if blocked.strip():
                    summary += f"### ⚠️ Blocked Items\n{blocked}\n"
        
        return summary


def main():
    """Main entry point for hook."""
    parser = argparse.ArgumentParser(description='Update project context')
    parser.add_argument('action', choices=[
        'task_completed', 'task_started', 'blocked', 
        'file_changed', 'decision_made', 'test_run',
        'session_summary', 'prune', 'cache'
    ])
    parser.add_argument('--details', help='Action details')
    parser.add_argument('--type', help='Context type for caching')
    parser.add_argument('--data', help='JSON data for caching')
    
    args = parser.parse_args()
    
    manager = ContextManager()
    
    if args.action == 'session_summary':
        print(manager.generate_session_summary())
    elif args.action == 'prune':
        manager.prune_old_context()
        print("✅ Old context pruned")
    elif args.action == 'cache' and args.type and args.data:
        manager.cache_context(args.type, json.loads(args.data))
        print(f"✅ Context cached: {args.type}")
    else:
        manager.update_project_state(args.action, args.details)
        print(f"✅ Context updated: {args.action}")


if __name__ == '__main__':
    main()
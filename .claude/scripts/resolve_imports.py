#!/usr/bin/env python3
"""
Resolve @import statements in CLAUDE.md files.
Follows Claude Code's import syntax to create a single resolved memory file.
"""

import re
import os
from pathlib import Path
from typing import Set, Optional


class ImportResolver:
    """Resolves @import statements in Claude memory files."""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.resolved_imports: Set[str] = set()
        self.max_depth = 5  # Claude Code's max import depth
        
    def resolve_file(self, file_path: str, depth: int = 0) -> str:
        """
        Recursively resolve imports in a file.
        
        Args:
            file_path: Path to the file to resolve
            depth: Current recursion depth
            
        Returns:
            Resolved content with imports expanded
        """
        if depth >= self.max_depth:
            return f"# ⚠️ Max import depth ({self.max_depth}) reached for: {file_path}\n"
        
        # Resolve path relative to base
        if not os.path.isabs(file_path):
            full_path = self.base_path / file_path
        else:
            full_path = Path(file_path)
        
        # Check if file exists
        if not full_path.exists():
            return f"# ❌ File not found: {file_path}\n"
        
        # Avoid circular imports
        abs_path = str(full_path.resolve())
        if abs_path in self.resolved_imports:
            return f"# ⚠️ Circular import detected: {file_path}\n"
        
        self.resolved_imports.add(abs_path)
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return f"# ❌ Error reading {file_path}: {str(e)}\n"
        
        # Find and resolve @imports
        import_pattern = r'^@(.+?)$'
        
        def replace_import(match):
            import_path = match.group(1).strip()
            
            # Add header to show where content comes from
            header = f"\n# 📁 Imported from: {import_path}\n"
            header += "# " + "=" * 50 + "\n\n"
            
            # Recursively resolve the imported file
            imported_content = self.resolve_file(import_path, depth + 1)
            
            footer = f"\n# " + "=" * 50 + "\n"
            footer += f"# 📁 End of import: {import_path}\n\n"
            
            return header + imported_content + footer
        
        # Replace all @import statements
        resolved_content = re.sub(
            import_pattern, 
            replace_import, 
            content, 
            flags=re.MULTILINE
        )
        
        # Remove this path from resolved set when backtracking
        self.resolved_imports.discard(abs_path)
        
        return resolved_content
    
    def resolve(self, input_file: str = "CLAUDE.md", output_file: str = None) -> str:
        """
        Resolve all imports in the input file.
        
        Args:
            input_file: Path to the input file
            output_file: Optional path to save resolved content
            
        Returns:
            Resolved content
        """
        print(f"📚 Resolving imports in {input_file}...")
        
        # Reset state
        self.resolved_imports.clear()
        
        # Add header
        header = "# Claude Memory - Resolved\n"
        header += f"# Generated from: {input_file}\n"
        header += f"# Generated at: {Path(input_file).stat().st_mtime if Path(input_file).exists() else 'N/A'}\n"
        header += "# " + "=" * 60 + "\n\n"
        
        # Resolve the main file
        resolved_content = header + self.resolve_file(input_file)
        
        # Add statistics footer
        footer = "\n\n# " + "=" * 60 + "\n"
        footer += "# Memory Resolution Statistics\n"
        footer += f"# Files imported: {len(self.resolved_imports)}\n"
        footer += f"# Total size: {len(resolved_content)} characters\n"
        footer += f"# Lines: {resolved_content.count(chr(10))} lines\n"
        
        resolved_content += footer
        
        # Save if output file specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(resolved_content)
            
            print(f"✅ Resolved memory saved to: {output_file}")
            print(f"   Files imported: {len(self.resolved_imports)}")
            print(f"   Total size: {len(resolved_content):,} characters")
            print(f"   Lines: {resolved_content.count(chr(10)):,} lines")
        
        return resolved_content
    
    def validate_imports(self, file_path: str) -> list:
        """
        Validate all imports in a file without resolving.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            List of validation issues
        """
        issues = []
        
        if not Path(file_path).exists():
            issues.append(f"Main file not found: {file_path}")
            return issues
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find all @imports
        import_pattern = r'^@(.+?)$'
        imports = re.findall(import_pattern, content, flags=re.MULTILINE)
        
        for import_path in imports:
            import_path = import_path.strip()
            full_path = self.base_path / import_path
            
            if not full_path.exists():
                issues.append(f"Import not found: {import_path}")
            elif full_path.is_dir():
                issues.append(f"Import is a directory, not a file: {import_path}")
            elif full_path.stat().st_size == 0:
                issues.append(f"Import file is empty: {import_path}")
            elif full_path.stat().st_size > 100000:  # 100KB warning
                size_kb = full_path.stat().st_size / 1024
                issues.append(f"Large import file ({size_kb:.1f}KB): {import_path}")
        
        return issues


def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Resolve @import statements in Claude memory files"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="CLAUDE.md",
        help="Input file to resolve (default: CLAUDE.md)"
    )
    parser.add_argument(
        "-o", "--output",
        default=".claude/CLAUDE_RESOLVED.md",
        help="Output file for resolved content (default: .claude/CLAUDE_RESOLVED.md)"
    )
    parser.add_argument(
        "-v", "--validate",
        action="store_true",
        help="Validate imports without resolving"
    )
    parser.add_argument(
        "-b", "--base",
        default=".",
        help="Base path for resolving relative imports (default: current directory)"
    )
    
    args = parser.parse_args()
    
    resolver = ImportResolver(base_path=args.base)
    
    if args.validate:
        # Validation mode
        print(f"🔍 Validating imports in {args.input}...")
        issues = resolver.validate_imports(args.input)
        
        if issues:
            print(f"\n❌ Found {len(issues)} issue(s):")
            for issue in issues:
                print(f"  - {issue}")
            exit(1)
        else:
            print("✅ All imports are valid!")
            exit(0)
    else:
        # Resolution mode
        try:
            resolver.resolve(args.input, args.output)
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            exit(1)


if __name__ == "__main__":
    main()
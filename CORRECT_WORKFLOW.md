# Correct Claude Code Workflow

## ✅ The Right Way: Agent-First Development

### 1. Start with CLAUDE.md
```bash
# FIRST thing - define project rules
cat > CLAUDE.md << 'EOF'
## Stack
- FastAPI + PostgreSQL + Redis
- Next.js 14 + TypeScript
- Railway deployment

## Rules
- Use Pydantic for validation
- Test coverage 80% minimum
- Async/await everywhere
- Repository pattern
EOF
```

### 2. Define Agents BEFORE Coding
```bash
# Set up agents for your needs
claude init

# Configure agents via YAML
vim .claude/agents/backend.yaml
vim .claude/agents/frontend.yaml
```

### 3. Use Agents to BUILD
```bash
# Let Claude build the initial structure
claude "Create a FastAPI backend with user authentication"

# Claude creates the code following CLAUDE.md rules
# Review the diff
git diff

# Continue building with agents
claude backend "Add portfolio CRUD endpoints"
claude frontend "Create dashboard components"
claude tester "Write tests for authentication"
```

### 4. Iterate with Context
```bash
# Update PROJECT_STATE.md as you go
echo "- [x] Authentication complete" >> .claude/PROJECT_STATE.md
echo "- [ ] Add portfolio management" >> .claude/PROJECT_STATE.md

# Claude sees this context and stays aligned
claude "Continue with the next task in PROJECT_STATE.md"
```

## ❌ The Wrong Way (What We Did Before)

### DON'T Do This:
1. ❌ Write all code manually
2. ❌ Add agents after the fact
3. ❌ Create CLAUDE.md at the end
4. ❌ Use agents for documentation only
5. ❌ Build complex agent.py scripts

### What Went Wrong:
```bash
# We built everything manually
mkdir src
touch src/main.py
# ... 3000 lines of manual coding ...

# THEN tried to add agents
python scripts/agent.py init  # Too late!

# Created CLAUDE.md after building
echo "# Project built manually" > CLAUDE.md  # Wrong!
```

## 🎯 Key Principles

### 1. Agents Build, You Review
- Claude writes the code
- You review diffs
- You decide what to commit
- Claude follows YOUR rules in CLAUDE.md

### 2. Context is King
```bash
# Good context management
.claude/
├── CLAUDE.md           # Concise rules (< 50 lines)
├── PROJECT_STATE.md    # Current work
├── DECISIONS.md        # Why choices were made
└── context-rules.yml   # What agents can see
```

### 3. Use Native Features
```bash
# Native Claude Code commands
claude backend "task"     # Uses YAML agent
claude review            # Code review
claude test              # Generate tests

# NOT custom Python scripts
python agent.py backend "task"  # Don't build this!
```

### 4. Terminal-First Philosophy
```bash
# Pipe outputs to Claude
git diff | claude "Review these changes"
pytest | claude "Explain failures"
curl api/health | claude "Diagnose issue"

# Stream logs
tail -f app.log | claude "Monitor for errors"
```

### 5. GitHub Integration
```markdown
# In GitHub issue:
@claude implement user authentication with JWT

# Claude automatically:
1. Creates branch
2. Implements feature
3. Runs tests
4. Opens PR
```

## 📊 Workflow Comparison

| Step | Wrong Way (Manual) | Right Way (Agent-First) |
|------|-------------------|------------------------|
| 1 | Create directories | Write CLAUDE.md rules |
| 2 | Write code manually | Configure agents |
| 3 | Debug for hours | Claude builds with agents |
| 4 | Add tests maybe | Claude writes tests |
| 5 | Document later | Documentation as you go |
| 6 | Add agents after | Agents from the start |

## 🚀 Quick Start Template

```bash
# 1. Clone template
git clone claude-native-template my-app
cd my-app

# 2. Define YOUR project
vim CLAUDE.md  # Add YOUR rules

# 3. Let Claude build
claude "Build a [your app type] with [your requirements]"

# 4. Review and iterate
git diff
git commit -m "Initial structure by Claude"

# 5. Continue with agents
claude backend "Add [feature]"
claude frontend "Create [UI]"
claude tester "Test [component]"
```

## 📝 Memory Management

### Keep CLAUDE.md Concise
```markdown
# GOOD - Bullet points
- FastAPI with async
- PostgreSQL for data
- Test coverage 80%

# BAD - Verbose documentation
The application uses FastAPI framework with asynchronous
programming patterns throughout. We chose PostgreSQL as our
database because... [500 more lines]
```

### Rotate When Needed
```bash
# When CLAUDE.md gets large (>100 lines)
mv CLAUDE.md CLAUDE_ARCHIVE_$(date +%Y%m%d).md
cp CLAUDE_TEMPLATE.md CLAUDE.md
```

### Use PROJECT_STATE.md for Current Work
```markdown
## Current Sprint
- [x] User auth
- [ ] Portfolio CRUD <- Working on this
- [ ] Market data integration

## Blocked
- Need API key for market data
```

## 🎓 Learning from Mistakes

### Mistake 1: Building First
**Problem**: We built 3000+ lines before adding agents
**Solution**: Start with agents, let them build

### Mistake 2: Complex Agent Scripts
**Problem**: Built elaborate agent.py coordinator
**Solution**: Use Claude's native YAML agents

### Mistake 3: Status in CLAUDE.md
**Problem**: Used CLAUDE.md for project status
**Solution**: CLAUDE.md for rules, PROJECT_STATE.md for status

### Mistake 4: No Context Filtering
**Problem**: Agents saw everything, context overflow
**Solution**: Use context-rules.yml to limit scope

## ✨ The Magic Formula

1. **Define** clear rules in CLAUDE.md
2. **Configure** agents for your stack
3. **Build** with agents from day one
4. **Review** diffs before committing
5. **Iterate** with clear context
6. **Ship** with confidence

Remember: Claude Code is a platform, not something to wrap!
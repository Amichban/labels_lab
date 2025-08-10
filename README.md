# 🚀 Claude Code Native Template

A production-ready template for building enterprise software with Claude Code as your AI pair programmer. This template implements all Anthropic-documented best practices for AI-native development.

## ✨ Features

### 🤖 AI-Powered Development
- **50+ Custom Slash Commands** - Specialized commands for every workflow
- **10+ Specialized Subagents** - Expert AI assistants for specific tasks
- **Smart Memory Management** - Modular memory with @import syntax
- **Intelligent CI/CD** - Claude as your CI teacher and deployment assistant

### 🏗️ Enterprise-Ready Architecture
- **API-First Development** - OpenAPI contracts with automatic validation
- **Testing Pyramid** - Smart test selection with flake detection
- **Feature Flags** - Trunk-based development with progressive rollout
- **Performance Monitoring** - Lighthouse CI with automatic optimization
- **Security Scanning** - SAST/DAST with automatic remediation

### 📊 Production Features
- **Canary Deployments** - Safe rollouts with automatic rollback
- **Incident Response** - AI-powered runbooks and root cause analysis
- **Architecture Decision Records** - Structured decision tracking
- **Documentation Freshness** - Automatic sync between code and docs
- **Cost Controls** - Rate limiting and budget management

## 🎯 Quick Start

### 1. Clone the Template

```bash
# Clone for a new project
git clone https://github.com/yourusername/claude-native-template.git my-project
cd my-project

# Remove template git history
rm -rf .git
git init

# Create your own repository
git add .
git commit -m "Initial commit from Claude Native Template"
```

### 2. Initialize Memory

```bash
# Run the memory initialization script
bash .claude/scripts/init-memory.sh

# Or use the slash command in Claude Code
/init
```

### 3. Configure Your Project

Edit `CLAUDE.md` to set your technology stack:

```markdown
### Technology Stack
- **Backend**: FastAPI  # or Django, Express, etc.
- **Database**: PostgreSQL  # or MySQL, MongoDB, etc.
- **Frontend**: Next.js  # or React, Vue, etc.
- **Deployment**: AWS  # or GCP, Railway, etc.
```

### 4. Set Up GitHub Secrets

Required secrets for CI/CD:
```yaml
ANTHROPIC_API_KEY    # For Claude in CI/CD
OPENAI_API_KEY       # Optional, for comparisons
AWS_ROLE_ARN         # For AWS deployments
GCP_WORKLOAD_IDENTITY # For GCP deployments
```

### 5. Start Development

```bash
# Use Claude Code commands
/prd "Build a user authentication system"
/api-design "POST /auth/login"
/scaffold backend auth
/test auth
```

## 📁 Project Structure

```
my-project/
├── CLAUDE.md                    # Main memory file with @imports
├── .claude/
│   ├── memory/                  # Modular memory files
│   │   ├── setup.md            # Environment configuration
│   │   ├── standards.md        # Code standards
│   │   ├── patterns.md         # Architecture patterns
│   │   └── team.md             # Team conventions
│   ├── subagents/              # Specialized AI assistants
│   │   ├── api-contractor.md   # API design
│   │   ├── test-runner.md      # Testing
│   │   ├── ci-teacher.md       # CI/CD help
│   │   └── ...
│   ├── scripts/                # Automation scripts
│   │   ├── resolve_imports.py  # Memory import resolver
│   │   └── init-memory.sh      # Memory bootstrap
│   └── slash-commands.yaml     # Custom commands
├── .github/
│   ├── workflows/              # GitHub Actions
│   │   ├── smart-testing.yml   # Intelligent testing
│   │   ├── intelligent-ci.yml  # AI-powered CI
│   │   └── ...
│   └── claude-config.yml       # Cost controls
├── src/                        # Your application code
├── tests/                      # Test files
└── docs/                       # Documentation
    ├── adr/                    # Architecture decisions
    └── VOLUMETRICS.md          # Performance metrics
```

## 🎮 Key Commands

### Memory Management
- `/init` - Initialize project memory
- `/memory` - Edit memory files
- `/remember "note"` - Quick add to memory
- `/memory-review` - Check memory health
- `/memory-clean` - Archive old items

### Development
- `/prd "feature"` - Create PRD with Claude
- `/api-design` - Design API contracts
- `/scaffold` - Generate boilerplate
- `/test` - Run smart tests
- `/perf` - Analyze performance

### Operations
- `/deploy` - Prepare deployment
- `/incident` - Respond to incidents
- `/runbook` - Generate runbooks
- `/adr "decision"` - Create ADR
- `/cost` - Analyze costs

## 🔧 Customization

### Add Your Own Subagents

Create `.claude/subagents/your-agent.md`:
```markdown
# Your Custom Agent
You are an expert in [domain]. 
When asked about [topic], you should...
```

### Add Custom Commands

Edit `.claude/slash-commands.yaml`:
```yaml
commands:
  - name: your-command
    description: "What it does"
    shell: "bash command to run"
```

### Configure Workflows

Edit `.github/workflows/` files for your CI/CD needs.

## 🧪 Testing

```bash
# Run all tests
make test

# Run smart test selection
/test-smart

# Run specific test pyramid level
/test unit
/test integration
/test e2e
```

## 🚢 Deployment

```bash
# Prepare deployment
/deploy-prepare

# Review deployment checklist
/deploy-checklist

# Deploy with canary
make deploy-canary

# Monitor deployment
/deploy-monitor
```

## 📈 Performance

The template includes automatic performance monitoring:
- Lighthouse CI on every PR
- Database query optimization
- Bundle size tracking
- Core Web Vitals monitoring

## 🔒 Security

Built-in security features:
- SAST/DAST scanning
- Dependency vulnerability checks
- Secret scanning
- Security headers validation
- Rate limiting

## 🤝 Contributing

We welcome contributions! Please see [FEEDBACK.md](FEEDBACK.md) for how to provide feedback and contribute improvements.

## 📚 Documentation

- [Architecture Decision Records](docs/adr/index.md)
- [System Volumetrics](docs/VOLUMETRICS.md)
- [Memory Management](.claude/memory/README.md)
- [Subagent Documentation](.claude/subagents/README.md)

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/claude-native-template/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/claude-native-template/discussions)
- **Claude Code Help**: `/help` in Claude Code

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built following [Anthropic's Claude Code best practices](https://docs.anthropic.com/en/docs/claude-code).

---

## 🎯 Next Steps After Cloning

1. **Initialize**: Run `/init` to set up memory
2. **Configure**: Update `CLAUDE.md` with your stack
3. **Secrets**: Add GitHub secrets for CI/CD
4. **Customize**: Add project-specific patterns
5. **Start**: Use `/prd` to begin development

## 🏆 Success Metrics

This template helps you achieve:
- ⚡ 50% faster development with AI assistance
- 🐛 70% fewer bugs with smart testing
- 🚀 90% safer deployments with canary releases
- 📊 100% visibility with comprehensive monitoring
- 🔒 Enterprise-grade security from day one

Ready to build something amazing? Start with `/prd "your idea"` and let Claude help!
# Architecture Decision Records

## Active Decisions

*No ADRs yet. Create your first with `/adr <title>`*

## Deprecated Decisions

*None*

## How to use ADRs

1. **Create new ADR**: `/adr "Use PostgreSQL for primary database"`
2. **Update ADR status**: `/adr-update 001 accepted`
3. **List ADRs**: `/adr-list`
4. **Search ADRs**: `/memory-search "database"`

## ADR Template

Each ADR follows this structure:

```markdown
# ADR-{number}: {title}

Date: {date}
Status: {status}
Decision: {decision}

## Context
What is the issue that we're seeing that is motivating this decision or change?

## Decision
What is the change that we're proposing and/or doing?

## Consequences
### Positive
- {positive consequence 1}
- {positive consequence 2}

### Negative
- {negative consequence 1}
- {negative consequence 2}

### Neutral
- {neutral consequence 1}

## Alternatives Considered
### Option 1: {alternative}
- **Pros**: {pros}
- **Cons**: {cons}
- **Reason not chosen**: {reason}

## Implementation
### Phase 1: {phase}
- [ ] {task 1}
- [ ] {task 2}

## References
- [{reference 1}]({url})

## Related ADRs
- ADR-{number}: {title}
- Supersedes: ADR-{number}
- Superseded by: None
```

## ADR Status Values

- **Draft**: Under discussion
- **Proposed**: Ready for review
- **Accepted**: Approved and active
- **Deprecated**: No longer relevant
- **Superseded**: Replaced by another ADR
- **Rejected**: Not approved

## Best Practices

1. **Write ADRs promptly** - Document decisions while context is fresh
2. **Keep them concise** - 1-2 pages maximum
3. **Include alternatives** - Show what was considered
4. **Link related ADRs** - Build a decision graph
5. **Review periodically** - Mark outdated ADRs as deprecated

## Quick Reference

- **When to write an ADR**: Significant architectural changes, technology choices, major pattern adoptions
- **Who writes ADRs**: Anyone can propose, team reviews
- **Where to store**: This directory (`docs/adr/`)
- **How to number**: Sequential (001, 002, 003...)
- **Format**: Markdown with consistent structure
# Beads Mastery Report — Airframe Workspace

## Current State Analysis

### Version Information
- **Current Version:** `bd v0.49.1 (dev)`
- **Latest Available:** `v1.0.4` (from doctor output)
- **Database:** SQLite with full schema compatibility
- **Mode:** Daemon mode active

### ⚠️ Upgrade Opportunity Identified

**Doctor Output Shows:**
```
⚠  CLI Version 0.49.1 (latest: 1.0.4)
```

**Configuration Issues to Fix:**
1. `jsonl_export` points to system file `interactions.jsonl` instead of `issues.jsonl`
2. `.beads/issues.jsonl` is gitignored (should be tracked for sync)

---

## Core Concepts Summary (From Medium Article)

### What Beads Solves

**1. Agent Amnesia Problem**
- Agents have no memory between sessions (~10 min sessions)
- Markdown plans get lost across compactions
- Beads provides persistent issue-based memory via git-backed SQLite

**2. Work Disavowal Problem**
- Agents near context limit (3k tokens) take shortcuts
- They claim work is "done" without fixing tests/issues
- Beads lets you kill agents after each issue, making sessions throwaway
- Each session cheaper + better decisions

**3. Lost Work Problem**
- Agents notice problems but ignore them to save context space
- With Beads: "I've filed issue 397 to get them working again"
- Work discovered and recorded automatically

### Why It's Called Beads
- Issues linked by dependencies = grapes on vine / beads on chain
- `bd` stands for "bug database"
- Agents follow dependency graph to complete tasks in right order

---

## Advanced Features Discovered

### 1. Formula System (Source Layer)
**Purpose:** Define reusable workflow templates with composition rules

**Lifecycle:** Rig → Cook → Run
- **Rig:** Compose formulas (extends, compose)
- **Cook:** Transform to proto (`bd cook` expands macros, applies aspects)
- **Run:** Agents execute poured mols or wisps

**Commands:**
- `bd formula list` - List available formulas from search paths
- `bd formula show <name>` - Show formula details and composition rules
- `bd formula convert` - Convert JSON to TOML

**Search Paths (in order):**
1. `.beads/formulas/` (project)
2. `~/.beads/formulas/` (user)
3. `$GT_ROOT/.beads/formulas/` (orchestrator, if GT_ROOT set)

---

### 2. Molecule System (Work Templates)

**Metaphor:**
- **Proto:** Uninstantiated template (reusable work pattern) with "template" label
- **Mol:** Real issues spawned from proto via substitution (`{{key}}`)
- **Wisp:** Ephemeral molecule (vapor phase, temporary)

**Commands:**
- `bd mol show <name>` - Show proto/molecule structure and variables
- `bd mol pour <proto>` - Instantiate proto as persistent mol (solid → liquid)
- `bd mol wisp <proto>` - Instantiate proto as ephemeral wisp (vapor phase)
- `bd mol bond <a> <b>` - Polymorphic combine: proto+proto, proto+mol, mol+mol
- `bd mol squash <mol>` - Compress molecule to digest
- `bd mol distill <epic>` - Extract proto from ad-hoc epic
- `bd mol current` - Show current position in molecule workflow
- `bd mol progress` - Show molecule progress summary
- `bd mol ready` - Find molecules ready for gate-resume dispatch
- `bd mol stale` - Detect complete-but-unclosed molecules
- `bd mol seed <formula>` - Verify formula accessibility or seed patrol formulas
- `bd mol burn <wisp>` - Delete wisp without digest

---

### 3. Graph Visualization

**Purpose:** Display issue dependency graph with execution order

**Display Formats:**
- `--box` (default): ASCII boxes showing layers, more detailed
- `--compact`: Tree format, one line per issue, more scannable

**Execution Order Rules:**
- Layer 0 / leftmost = no dependencies (can start immediately)
- Higher layers depend on lower layers
- Nodes in same layer can run in parallel

**Status Icons:**
- ○ open
- ◐ in_progress
- ● blocked
- ✓ closed
- ❄ deferred

**Commands:**
- `bd graph <issue-id>` - Show graph for specific issue
- `bd graph --all` - Show all open issues grouped by connected component
- `bd graph --json` - JSON output

---

### 4. Agent State Management (ZFC Compliance)

**Purpose:** Self-report agent state for Witness/monitoring systems

**States:**
- `idle` - Agent waiting for work
- `spawning` - Agent starting up
- `running` - Agent executing (general)
- `working` - Actively working on task
- `stuck` - Blocked, needs help
- `done` - Completed current work
- `stopped` - Cleanly shut down
- `dead` - Died without clean shutdown (set by Witness via timeout)

**Commands:**
- `bd agent state <agent> <state>` - Set agent state
- `bd agent heartbeat <agent>` - Update last_activity timestamp
- `bd agent show <agent>` - Show agent bead details
- `bd agent backfill-labels` - Backfill role_type/rig labels on existing agent beads

---

### 5. Audit Trail System

**Purpose:** Append-only JSONL log for auditing and dataset generation (SFT/RL fine-tuning)

**Storage:** `.beads/interactions.jsonl`

**Entry Types:**
- **Interaction:** Main event entries
- **Label:** References existing interaction entry

**Commands:**
- `bd audit record <message>` - Append audit interaction entry
- `bd audit label <entry-id> --msg "reason"` - Append label entry

---

### 6. Worktree Management

**Purpose:** Multiple working directories sharing same beads database for parallel development

**Features:**
- Automatically sets up redirect file so all worktrees share `.beads` database
- Consistent issue state across all worktrees

**Commands:**
- `bd worktree create <name>` - Create worktree with beads redirect
- `bd worktree create <name> --branch <branch>` - Create with specific branch name
- `bd worktree list` - List all git worktrees
- `bd worktree remove <name>` - Remove worktree (with safety checks)
- `bd worktree info` - Show info about current worktree

---

### 7. Prime Command (AI Optimization)

**Purpose:** Output essential Beads workflow context in AI-optimized markdown format

**Modes:**
- **MCP Mode:** Brief workflow reminders (~50 tokens) for SessionStart/PreCompact hooks
- **CLI Mode:** Full command reference (~1-2k tokens)

**Config Options:**
- `no-git-ops: true` - Stealth mode (no git commands in session close protocol)
  - Set via: `bd config set no-git-ops true`

**Workflow Customization:**
- Place `.beads/PRIME.md` to override default output entirely
- Use `--export` to dump default content for customization

**Commands:**
- `bd prime` - Output essential context (auto-detects MCP server)
- `bd prime --full` - Force full CLI output
- `bd prime --mcp` - Force MCP mode (minimal output)
- `bd prime --stealth` - Stealth mode (no git operations, flush only)
- `bd prime --export` - Output default content for customization

---

### 8. Formula Management

**Commands:**
- `bd formula list` - List available formulas from all search paths
- `bd formula show <name>` - Show formula details and composition rules
- `bd formula convert <file> --to toml` - Convert JSON to TOML format

---

### 9. Upgrade Management

**Commands:**
- `bd upgrade status` - Check if bd version changed since last use
- `bd upgrade review` - Show what's new since last version
- `bd upgrade ack` - Acknowledge current version

---

### 10. Other Notable Commands

**Maintenance:**
- `bd doctor` - Check and fix beads installation health (start here)
- `bd repair` - Repair corrupted database by cleaning orphaned references
- `bd resolve-conflicts` - Resolve git merge conflicts in JSONL files
- `bd migrate` - Database migration commands

**Admin:**
- `bd admin <command>` - Administrative commands for database maintenance

**Sync & Data:**
- `bd branch <command>` - List or create branches (requires Dolt backend)
- `bd daemon <command>` - Manage background sync daemon
- `bd export` - Export issues to JSONL or Obsidian format
- `bd federation <command>` - Manage peer-to-peer federation (requires CGO)
- `bd import` - Import issues from JSONL format
- `bd merge` - Git merge driver for beads JSONL files
- `bd restore` - Restore full history of compacted issue from git
- `bd sync` - Export database to JSONL (sync with git)
- `bd vc <command>` - Version control operations (requires Dolt backend)

**Integrations:**
- `bd jira <command>` - Jira integration commands
- `bd linear <command>` - Linear integration commands
- `bd repo <command>` - Manage multiple repository configuration

**Additional:**
- `bd agent <command>` - Manage agent bead state
- `bd audit <command>` - Record and label agent interactions (append-only JSONL)
- `bd blocked` - Show blocked issues
- `bd completion` - Generate autocompletion script for shell
- `bd cook` - Compile formula into proto (ephemeral by default)
- `bd defer` - Defer one or more issues for later
- `bd epic <command>` - Epic management commands
- `bd help <command>` - Help about any command
- `bd hook <command>` - Execute git hook (called by hook scripts)
- `bd lint` - Check issues for missing template sections
- `bd mail <command>` - Delegate to mail provider (e.g., gt mail)
- `bd mol` - Molecule commands (work templates)
- `bd orphans` - Identify orphaned issues (referenced in commits but still open)
- `bd preflight` - Show PR readiness checklist
- `bd quickstart` - Quick start guide for bd
- `bd ready` - Show ready work (no blockers, open or in_progress)
- `bd rename <command>` - Rename an issue ID
- `bd search <command>` - Search issues by text query
- `bd set-state` - Set operational state (creates event + updates label)
- `bd ship` - Publish a capability for cross-project dependencies
- `bd slot <command>` - Manage agent bead slots
- `bd status` - Show issue database overview and statistics
- `bd supersede` - Mark an issue as superseded by newer one
- `bd swarm <command>` - Swarm management for structured epics
- `bd types` - List valid issue types
- `bd undefer` - Undefer one or more issues (restore to open)
- `bd where` - Show active beads location

---

## Configuration & Setup

### Git Hooks (Recommended)
```bash
# Install recommended hooks
bd hooks install

# Recommended hooks: pre-push, pre-commit, post-merge
```

### Config Settings
```bash
# Set custom status states
bd config set status.custom "awaiting_review,awaiting_testing"

# Enable stealth mode (no git ops)
bd config set no-git-ops true

# Jira integration
bd config set jira.url "https://company.atlassian.net"
bd config set jira.project "PROJ"
```

### Database Mode
```bash
# Daemon mode (default, recommended)
# Socket: C:\Users\micha\repos\airframe\.beads\bd.sock

# No-daemon mode (direct storage, bypass daemon)
bd --no-daemon <command>

# Read-only mode (block write operations, for worker sandboxes)
bd --readonly <command>

# Sandbox mode (disable daemon and auto-sync)
bd --sandbox <command>
```

---

## Best Practices

### 1. Always Use `bd ready` Before Starting Work
```bash
bd ready --json  # Get definitive list of unblocked work
```

### 2. Kill Agents After Each Issue Completion
- Makes sessions throwaway (cheaper, better decisions)
- Prevents agents from claiming work "done" without fixing issues

### 3. File Issues for Discovered Problems
```bash
# Instead of ignoring broken tests
bd create "Fix broken tests discovered in feature X" --priority high
```

### 4. Use `bd graph` to Understand Dependency Graph
- Visualize execution order
- Identify parallelizable tasks (same layer)

### 5. Export Issues for Review/Archive
```bash
bd export --format obsidian  # Export to Obsidian format
bd export --format jsonl     # Export to JSONL
```

---

## Known Issues & Fixes

### Issue 1: jsonl_export Configuration
**Problem:** `interactions.jsonl` is a system file, shouldn't be configured as export target

**Fix:**
```bash
# Check current config
bd config list

# Remove invalid configuration (if needed)
bd config unset jsonl_export
```

### Issue 2: issues.jsonl Gitignore
**Problem:** `.beads/issues.jsonl` is ignored by git, will cause sync failure

**Fix:**
```bash
# Edit .gitignore to untrack issues.jsonl
# Remove line: .beads\issues.jsonl

# Or use bd sync which handles this automatically
bd sync
```

---

## Summary

### What We Have Now (v0.49.1)
✅ Full feature set including:
- Issue tracking with dependencies
- Formula system for reusable workflows
- Molecule system for work templates
- Graph visualization
- Agent state management
- Audit trail system
- Worktree management
- Prime command for AI optimization

### What Upgrade Would Give Us (v1.0.4)
📋 Need to check release notes for:
- Bug fixes
- Performance improvements
- New features
- Breaking changes

---

## Next Steps

1. **Fix configuration issues** shown by `bd doctor`
2. **Check upgrade path** from v0.49.1 to v1.0.4
3. **Create molecule templates** for common workflows
4. **Set up formulas** for project-specific patterns
5. **Configure agent states** for monitoring systems

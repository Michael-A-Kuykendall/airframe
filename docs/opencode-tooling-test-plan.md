# Airframe Opencode Tooling Upgrade Comparison & Test Plan

## Status Summary

**dzero-cas upgrade plan phases:**
- Phase 1 (AST tools, terminal superpowers) - ✅ COMPLETE in airframe
- Phase 2 (Semantic index with LanceDB) - ⏸️ SKIPPED (needs separate setup)
- Phase 3 (Evals & snapshots) - ⏸️ NOT IMPLEMENTED

**Airframe current state:** Already has dzero-cas Phase 1 tools integrated!

---

## Current Tooling Comparison

### ✅ Already Installed/Configured in Airframe

| Tool | Status | Location | Notes |
|------|--------|----------|-------|
| `ast-grep` (alias: `ag`) | ✅ Configured | opencode.json line 119 | AST-aware pattern matching for Rust |
| `fd` (file finder) | ✅ Configured | opencode.json via terminal-triage | Fast file replacement for find |
| `fzf` (fuzzy finder) | ✅ Configured | opencode.json via terminal-triage | Fuzzy terminal navigation |
| `bat` (syntax cat) | ✅ Configured | opencode.json via terminal-triage | Syntax-highlighted cat replacement |
| `eza` (ls replacement) | ⚠️ Not configured | - | Could add as alias |
| `zoxide` (smart cd) | ⚠️ Not configured | - | Could add as alias |

### ❌ Not Yet Implemented

| Feature | dzero-cas Phase | Status | Notes |
|---------|-----------------|--------|-------|
| Semantic code index | Phase 2 | ⏸️ Skip | Requires LanceDB setup (out of scope) |
| Evals suite | Phase 3 | ❌ Not done | Could add tests/evals/ directory |
| Repo snapshots via repomix | Phase 3 | ❌ Not done | Could add .github/workflows/snapshot.yml |

---

## Testing Plan

### Test 1: Verify AST-Grep Works
```bash
# Test pattern matching on Rust code
ag -p "pub fn" src/backend/bindless/
ag -p "fn.*run_" src/backend/bindless/pipeline/
```

### Test 2: Verify Fuzzy Finder Setup
```powershell
# Test terminal-triage command (from opencode.json)
$env:PATH = "C:\Program Files\Git\fzf\bin;" + $env:PATH
$env:FZF_DEFAULT_OPTS = "--height 40% --bind 'ctrl-t:select' --bind 'tab:down,shift-tab:up'"

# Run triage
& "terminal-triage" | Select-Object -First 5
```

### Test 3: Verify File Finder (fd)
```powershell
# Test fd replacement for find
fd .rs src/backend/bindless/ | Select-Object -First 10
```

### Test 4: Add Missing Tools (Optional)
```powershell
# Install eza if not present
winget install eza-community.eza --silent --accept-source-agreements

# Install zoxide if not present  
winget install zoxide --silent --accept-source-agreements

# Create aliases in PowerShell profile
function New-Aliases {
    # ls alias
    $null = Register-ArgumentProcessor -Name ls -ScriptBlock {
        eza "$args[0]" @args
    }
    
    # find alias (use fd)
    $null = Register-ArgumentProcessor -Name find -ScriptBlock {
        fd "$args[0]" @args
    }
}

# Add to profile
"function New-Aliases {" | Out-File -Append -Path $PROFILE
"    New-Aliases" | Out-File -Append -Path $PROFILE
"}" | Out-File -Append -Path $PROFILE
```

---

## Recommended Actions

### Immediate (5 minutes)
1. ✅ Verify `ast-grep` is installed and working
2. ✅ Test `terminal-triage` command with fzf
3. ✅ Test `fd` file finder
4. ⚠️ Optionally add eza/zoxide aliases

### Short-term (optional)
- Add evals suite in `tests/evals/`
- Add repomix snapshot workflow to CI

### Skip for now
- Semantic index (Phase 2) - requires LanceDB, out of scope

---

## Next Steps

1. Run tests above to verify tools work
2. If all pass, update AGENTS.md to reflect tooling upgrade completion
3. Optionally add missing eza/zoxide aliases
4. Consider adding evals suite (Phase 3) if desired

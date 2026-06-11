---
inclusion: manual
---

# Hotfix Deployment Workflow — Airframe + Shimmy

## Strategy
Private first → CI green → public merge. Never push directly to public without private CI passing.

## Airframe Hotfix

### 1. Create clean branch from release base (NOT from feature branches)
```bash
cd /c/Users/micha/repos/airframe
git checkout release/v0.2.2-clean   # or latest release base
git checkout -b fix/<description>   # fix/** and hotfix/** both trigger CI
# cherry-pick ONLY the fix commit(s) — no feature work
git cherry-pick <fix-commit-sha>
```

### 2. Bump version + update CHANGELOG
- `Cargo.toml`: increment patch version (0.2.2 → 0.2.3)
- `CHANGELOG.md`: add entry at top with date and fix description

### 3. Verify locally
```bash
cargo build
cargo test --lib                         # must be 0 failures
cargo test --test <relevant_test>        # must be 0 failures
```

### 4. Push to private, wait for CI green
```bash
git push private fix/<description> -u
# Watch CI at: https://github.com/Michael-A-Kuykendall/airframe-private/actions
```

### 5. Once CI green — merge to release branch on private, push public
```bash
git checkout release/v0.2.3    # create from base if needed
git merge --no-ff fix/<description>
git push private release/v0.2.3
git push public release/v0.2.3
```

### 6. Tag
```bash
git tag v0.2.3 && git push private v0.2.3 && git push public v0.2.3
```

---

## Shimmy Commensurate Bump

When airframe bumps, shimmy gets a matching bump if inference is affected.

```bash
cd /c/Users/micha/repos/shimmy
git checkout release/v2.2-cleanup
git checkout -b fix/airframe-v0.2.3-bump
# Update: airframe = { version = "0.2.3" } in Cargo.toml
cargo build && cargo test
git add Cargo.toml Cargo.lock && git commit -m "chore: bump airframe to v0.2.3"
git push origin fix/airframe-v0.2.3-bump   # shimmy-private CI
# After CI green:
git push public fix/airframe-v0.2.3-bump
```

---

## Remotes Reference
- airframe `private` → airframe-private.git
- shimmy `origin` → shimmy-private.git  
- shimmy `public` → shimmy.git

## CI Trigger Branches (airframe)
`master`, `feat/**`, `release/**`, `fix/**`, `hotfix/**`, `hygiene/**`

## Rules
- NEVER push to public without private CI green
- Cherry-pick individual commits — no feature work in hotfix
- Hotfix branch: fix commit(s) + version bump only

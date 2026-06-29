# Build Plan
Creating a central build plan to document all tooling enhancements.

## Scope
- Add `.vscode/extensions.json` with recommended Rust ecosystem tools
- Extend `[.gitignore]` with common dev artifact globs (keeping the existing list intact)
- Establish `.clinerules` for Windows and PowerShell ergonomics

## Verification
After each step, verify no existing content was deleted and the project remains buildable via cargo check.
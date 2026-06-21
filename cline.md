# 🚀 CLINE WORKSPACE CONTINUITY & TOOL GUIDE

This document serves as the definitive guide for all local development processes, tool usage, and operational mandates within the airframe repository. It synthesizes information from cloud directives, project dependencies, and available agent capabilities.

---

## ⚙️ I. OPERATIONAL MANDATES (Cloud Work Mode)

*Sourced from: .kiro/cloud-work-mode.md*

**Core Directives:**
1.  **Execution Style:** Execute tasks exactly as specified.
2.  **Verbosity:** Minimal. Only what's necessary to confirm completion.
3.  **Deviation Policy:** No deviations. No commentary unless explicitly requested.
4.  **Focus:** Single task, single file, single test at a time.
5.  **Git Workflow (Mandatory):** All code changes *must* be committed and pushed. The work product is the git history. Commit after each change, and push to the private remote immediately.

---

## 🛠️ II. CLINE TOOL REFERENCE GUIDE & DEMONSTRATION

This section inventories all available tools for agent execution. Always invoke these via tool calls rather than assuming functionality.

### 1. `skills` (Tool Invocation)
*   **Purpose:** Executes predefined, high-level tasks/workflows (e.g., running a specific review or build). This is the primary way to trigger complex actions.
*   **Usage Example:** `skill: "review-pr", args: "123"`
*   **Available Skills:** `review-team`, `find-skills`.

### 2. `read_files` (File System Read)
*   **Purpose:** Reads content from specific, known files at absolute paths. Essential for reading source code or configuration data.
*   **Usage Example:** Reading a file: `skill: "read_files", args: {"files": [{"path": "/path/to/file.txt"}]}`
*   **Advanced Usage:** Can read specific line ranges (e.g., lines 10 to 20).

### 3. `search_codebase` (Code Pattern Search)
*   **Purpose:** Performs powerful regex searches across the entire codebase to find patterns, function definitions, or class names.
*   **Usage Example:** Finding all usages of a variable: `skill: "search_codebase", args: {"queries": ["\bmyVariable\b"]}`

### 4. `run_commands` (Shell Execution)
*   **Purpose:** Executes arbitrary shell commands from the root directory (Windows environment). Used for build steps, git status checks, or listing directories.
*   **Usage Example:** Running a command: `skill: "run_commands", args: {"commands": ["git status"]}`

### 5. `editor` (File Modification)
*   **Purpose:** The controlled mechanism for making precise edits to files, either replacing content (`old_text`/`new_text`) or inserting text at a specific line number (`insert_line`). **This is the preferred method over shell redirection.**
*   **Usage Example (Replacement):** `skill: "editor", args: {"path": "file.py", "old_text": "old_func()", "new_text": "new_func()"}`

### 6. Other Utility Tools
*   `fetch_web_content`: For retrieving external documentation or API references from URLs.
*   `ask_question`: To pause execution and gather necessary clarification/options from the user.
*   `spawn_agent`: To delegate specialized, complex tasks to a sub-agent with a custom system prompt.

---

## 📂 III. PROJECT CONTEXT & DEPENDENCIES

**A. Core Dependencies (From .opencode/package.json):**
*   The project currently depends on: `@opencode-ai/plugin`: `1.17.7`.

**B. Directory Structure Notes:**
*   **.kiro/:** Contains workflow and operational mode documentation (`cloud-work-mode.md`). This directory dictates *how* work should be done.
*   **.opencode/:** Contains core project assets, dependencies (`node_modules`), and skill definitions/metadata (e.g., `package.json`, `skills`).

---

**NEXT STEPS:** Always prioritize following the mandates in Section I when making changes.
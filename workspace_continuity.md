# 🚀 CLINE WORKSPACE CONTINUITY & TOOL GUIDE

This document serves as the definitive guide for all local development processes, tool usage, and operational mandates within the airframe repository. It synthesizes information from cloud directives, project dependencies, and available agent capabilities.

---

## 📂 III. PROJECT CONTEXT & DEPENDENCIES

**A. Core Dependencies (From .opencode/package.json):**
*   The project currently depends on: `@opencode-ai/plugin`: `1.17.7`.

**B. Directory Structure Notes:**
*   **.kiro/:** Contains workflow and operational mode documentation (`cloud-work-mode.md`). This directory dictates *how* work should be done.
*   **.opencode/:** Contains core project assets, dependencies (`node_modules`), and skill definitions/metadata (e.g., `package.json`, `skills`).

---

## 🚀 IV. WORKSPACE CONTINUITY PROCESS

### For Airframe Project:

This document enables full local workspace continuity for the Airframe project by integrating all tools and processes:

1. **Tool Access Methodology:**
   - All Cline tools can be invoked via their respective tool calls as shown in this documentation.
   - Each tool call follows strict JSON syntax (as shown above) and uses absolute paths where needed.

2. **Workspace Integration Strategy:**
   - *Tool Demonstrations:*
     * `skills` for complex operations like review tasks
     * `read_files` to access any specific files at precise paths
     * `search_codebase` to find code patterns and references
     * `run_commands` for shell operations within the Airframe directory
     * `editor` to modify files with precision and safety

### For Shimmy Project:

This document demonstrates processes that can be applied to any related project (Shimmy):

1. **Cross-Project Integration:**
   - The same Cline tools can be used for accessing information in your Shimmy directory.
   - Apply the tool catalog and operational mandates consistently across both projects.

### Directory Structure Overview:

Airframe Project:
```
C:\Users\micha\repos\airframe
├── cline.md (this file)
├── .kiro/ (contains cloud-work-mode.md)
└── .opencode/ (includes package.json and other metadata)
```

Shimmy Project:
```
C:\Users\micha\repos\shimmy
└── (project files)
```

### Key Integration Points:

1. **Tool Usage Examples:**
   - `skills`: Accesses operational capabilities seamlessly across projects
   - `read_files`: Retrieves any file content from both Airframe and Shimmy repositories
   - `search_codebase`: Queries patterns within any project directory
   - `run_commands`: Executes shell operations within the respective workspaces
   - `editor`: Makes precise modifications to files in any workspace

### Operational Mandates Application:

1. **Execution Style:**
   - Exact compliance with cloud directives for each task execution
2. **Verbosity Policy:**
   - Minimal output to confirm completion of operations
3. **Deviation Policy:**
   - No deviations from operational mandates; always follow cloud instructions
4. **Focus Strategy:**
   - Single task, single file at a time with consistent process application
5. **Git Workflow Requirements:
   - Commit and push all changes immediately after any modification

---

**NEXT STEPS:** Always prioritize following the mandates in Section I when making changes.
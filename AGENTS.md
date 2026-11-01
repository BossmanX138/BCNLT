# BossCrafts Universal Project Isolation Policy

**MANDATORY: Applies to every BossCrafts project, past, present, and future.**

1. Before operations, determine and record the canonical active project root. If ambiguous, STOP.
2. All project writes, edits, generated files, logs, caches, temporary files, downloads, artifacts, and destructive actions MUST remain inside that project root. Never modify another project or any external path.
3. Treat other projects as strictly READ-ONLY. COPY any needed code, assets, configurations, models, templates, or other resources into the active root *before use*. Never move or edit sources. Preserve provenance and licensing.
4. Do not circumvent isolation using symlinks, hardlinks, junctions, submodules, mounts, path traversal, inherited output paths, or external build scripts. Validate resolved destinations.
5. Prefer relative paths inside root. Never depend on mutable files in another BossCrafts checkout. Bundle project-owned resources for whole-folder portability.
6. Configure tools, agents, subprocesses, caches, temp dirs, and output dirs to write only within root. If this cannot be guaranteed, STOP rather than executing.
7. Preserve existing project-specific instructions. Audit historical projects individually; do not assume compliance.
8. Before declaring a project isolated, document root, source provenance, copied dependency hashes, unavoidable external OS/runtime prerequisites, and results of a clean-folder portability test in the project.
9. If a request conflicts with this policy, STOP and report the conflict. No silent exceptions.

**Important:** Instructions alone are not a security boundary. Enforce with filesystem sandboxing/permissions and automated path-validation checks. External OS/runtime reads are permitted, but project-data writes outside root are forbidden.


## BossCrafts Controlled Sharing Amendment (2026-11-01; supersedes conflicting restrictions above)

**ABSOLUTE INVARIANT: No project may modify another project's files, directories, configurations, caches, private data, or source.** This includes direct writes, indirect subprocess writes, deletes, renames, moves, and writes through linked paths. No user-facing integration changes this rule.

**Approved opt-in sharing:** A project MAY read immutable, version-pinned centrally managed libraries, dependency environments (e.g. PyAudio), models, and assets from a separately governed shared-resource root. A consuming project MUST NOT write to that root. Each opted-in resource must be declared in an in-project manifest with resource ID, version/hash, access scope, license, and resolution path. No implicit access. Shared-library changes can only be made by an independent authorized maintenance workflow, never as part of a consuming project's work.

**Cross-project features:** Use documented, versioned APIs or message protocols with explicit capabilities and least privilege. Invoking BossForgeOS from AgentForge does not grant AgentForge direct write access to BossForgeOS or its state. The receiving service may manage only its own state under its own authorization and policy. No source-tree mounts, shared-writable working directories, or hidden cross-project write paths.

**Writes:** All project-owned outputs, caches, logs, builds, temp data, and checkpoints MUST remain within the active project's verified canonical root. Dedicated shared services may write *their own* service-owned storage under separate service authorization, not at the direction of a consumer's unrestricted filesystem access. Validate real paths, symlinks, junctions, subprocess and tool outputs; stop on boundary violation.

**Portability:** Integrated mode declares external shared resources and reproducible resolution. Standalone release bundles redistributable components or a verified installer/dependency lock. No silent runtime dependency on other projects' private directories. Do not copy a dependency merely because it is accessed from the approved shared library; the earlier copy-before-use requirement remains in force for borrowing from *other project roots*, not the shared-resource hub.

**Enforcement:** AGENTS.md is advisory, not a sandbox. Use OS permissions, capability tokens, path checks, integration tests, and explicit opt-in manifests. Fail closed on uncertain authorization. Never update another project's AGENTS.md as a side effect of working inside a different project; universal-policy rollout is a separately authorized administrative operation.

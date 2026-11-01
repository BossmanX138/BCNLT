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

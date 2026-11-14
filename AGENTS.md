# BossCrafts Cross-Project Resource Policy

This policy supersedes all earlier BossCrafts project-isolation and controlled-sharing rules.

- Confirm the active project root before writing files.
- When this project needs code, assets, configuration, models, templates, tooling, or other resources from another ordinary project, copy the required resources into this project's own root before modifying or integrating them. Copy; never move.
- Treat the originating project as read-only during consumer-project work. Changes, generated output, caches, logs, and temporary files belong to the consuming project or its declared user-selected workspace.
- Record copied-resource provenance, including the source project, version or commit when available, applicable license, and update method.
- Do not use links, mounts, submodules, path traversal, or subprocess output paths to turn another ordinary project into a hidden writable dependency.
- Shared-purpose resources are the exception. Databases, durable state stores, services, APIs, registries, model or package stores, shared-resource hubs, and projects explicitly designed to serve other projects may be used in place through their owner-approved interface or declared contract.
- A shared-purpose resource may be written only when its ownership contract explicitly grants that operation. The grant covers only the declared API, database, state, or companion files; it does not grant general write access to the provider's source tree or unrelated state.
- Maintenance of a provider project's implementation must be performed as explicit work in that provider project, not as an accidental side effect of consumer-project work.
- Database migrations and repairs must preserve data, use recoverable procedures where supported, validate integrity, and never replace a canonical database with an empty substitute.
- Keep credentials and encryption keys in approved secret storage.
- Preserve stricter project-specific safety and quality rules unless they conflict with this policy. When they conflict, this policy controls cross-project resource use.

`AGENTS.md` is operational guidance, not an operating-system security boundary. Use permissions, capability checks, path validation, and tests where enforcement matters.

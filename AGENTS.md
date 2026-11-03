# BossCrafts Project Borrowing Rule

- If this project needs a resource, function, source file, asset, configuration, model, template, or other implementation from another project, copy it into this project's own directory before using or modifying it.
- Do not modify files in another project's workspace. Preserve the copied item's provenance and license when applicable.
- Do not depend at runtime on mutable implementation files in another project's workspace; keep the project's required copy with this project.
- A specifically designated shared operational data file that outside systems must continually query, such as a database, may be read or modified only under its explicit ownership and write contract.
- The shared-data exception is limited to the designated data file and any required companion state files. It does not authorize changing the source code, configuration, migrations, tools, or other files that govern that data.
- If ownership, designation, or the allowed file scope is unclear, preserve the existing files and ask before writing.

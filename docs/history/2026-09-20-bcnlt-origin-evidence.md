# BCNLT origin evidence: September 20 prototype

The earliest recovered BCNLT conversation was supplied on October 28, 2026 as
`Pasted text.txt` with SHA-256
`dafa9fe0d2283691a64cae9efc964c1d6785763da70f0ec817474a4fa2e822b1`.
This note records the architectural requirements used from that source without
copying the full conversation into the repository.

The opening specification establishes these original requirements:

- contextual semantic events rather than raw text tokens;
- uncertainty scoring from structural ambiguity;
- UYR expansion and targeted retrieval followed by reevaluation;
- a wildcard resolver that classifies unfamiliar constructs, forms provisional
  hypotheses, records evidence, and assigns confidence;
- one compact Common Coding Core for universal programming concepts;
- language-specific oddities kept outside that core and handled through
  extensions and wildcard learning;
- preprocessing, uncertainty, UYR, wildcard, IR, adapters, and CLI kept as
  separable modules;
- parallel preprocessing and incremental reprocessing as explicit design goals.

Later passages in the recovered record describe wildcard memory as an
append-only, self-growing lab notebook. Repeated sightings increase evidence,
and stable repeated patterns may be promoted into named extensions. That is the
historical basis for the current learned-definition store, autonomous promotion
rules, and the separation between universal Common Core concepts and learned
language-specific extensions.

The recovered prototype implemented BCNLT as semantic preprocessing. The
current tokenizer project carries that semantic stage forward into a native
model-facing hybrid tokenizer: BCNLT structural records receive stable token
IDs and dedicated model rows, while the preserved base tokenizer continues to
encode ordinary text, raw-code compatibility mode, and field payloads.

# Runeforge Mk2 7B BCNLT integration

This profile connects the preserved merged Runeforge Mk2 7B checkpoint to BCNLT through an independently generated Ollama Q4_K_M runtime. BCNLT supplies semantic context before inference while Runeforge retains its native tokenizer.

The Ollama template uses the Alpaca instruction/input format used for Runeforge Mk2 training and evaluation. `profile.json` records source, converter, F16, and Q4 hashes. `scripts/connect_runeforge_mk2.ps1` validates those pins, registers the runtime, requires a real definition-line readiness completion, and writes BCNLT smoke evidence.

The Q4 runtime permits a like-for-like local comparison with Wizard-Phi v0.2 Q4_K_M. The head-to-head measures definition-line extraction from raw versus BCNLT semantic contexts; it is not a general model-quality benchmark.

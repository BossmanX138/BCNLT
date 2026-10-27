# BCNLT + Wizard-Phi 3.5 v0.2

This profile runs the promoted, merged Wizard-Phi v0.2 checkpoint through Ollama and supplies BCNLT semantic context before inference. Wizard-Phi keeps its native Phi tokenizer; BCNLT is a semantic preprocessing and retrieval layer.

The source checkpoint and separate Q4_K_M GGUF are pinned by SHA-256 values in `profile.json`. The runtime copy is sized for the local RTX 3070. The preserved Transformers release remains unchanged.

From the BCNLT repository:

```powershell
.\scripts\connect_wizard_phi_v02.ps1
```

The command validates the source checkpoint, creates `wizard-phi-3.5:v0.2` when needed, performs a real completion readiness check, runs BCNLT's model-context evaluation, and writes evidence under `outputs/wizard-phi-v02/`.

If the GGUF runtime copy needs to be reconstructed, run:

```powershell
.\scripts\build_wizard_phi_v02_gguf.ps1
```

To rebuild the Ollama model after an intentional source update:

```powershell
.\scripts\connect_wizard_phi_v02.ps1 -Rebuild
```

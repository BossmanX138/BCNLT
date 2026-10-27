[CmdletBinding()]
param([switch]$Force)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Profile = Get-Content -LiteralPath (Join-Path $Root "integrations/runeforge_mk2/profile.json") -Raw | ConvertFrom-Json
$WizardRoot = "D:\Wizard-Phi-3.5-v0.1"
$Source = $Profile.source_model -replace '/', '\'
$RuntimeDirectory = Join-Path $WizardRoot "outputs\runtime-runeforge-mk2"
$F16 = Join-Path $RuntimeDirectory "Runeforge-Mk2-7B-F16.gguf"
$Q4 = Join-Path $RuntimeDirectory "Runeforge-Mk2-7B-Q4_K_M.gguf"
$ConverterRoot = Join-Path $WizardRoot "tools\llama.cpp"
$Converter = Join-Path $ConverterRoot "convert_hf_to_gguf.py"
$Python = Join-Path $WizardRoot ".venv-gpu\Scripts\python.exe"
$Quantizer = Join-Path $env:LOCALAPPDATA "Programs\Ollama\lib\ollama\llama-quantize.exe"

$ConfigHash = (Get-FileHash -LiteralPath (Join-Path $Source "config.json") -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ConfigHash -ne $Profile.source_config_sha256) { throw "Runeforge Mk2 7B config hash mismatch: $ConfigHash" }
if (-not (Test-Path -LiteralPath $Converter)) { throw "Pinned llama.cpp checkout is missing: $ConverterRoot" }
$ConverterCommit = (& git -C $ConverterRoot rev-parse HEAD).Trim()
if ($ConverterCommit -ne $Profile.converter.commit) { throw "llama.cpp commit mismatch: $ConverterCommit" }
if (-not (Test-Path -LiteralPath $Python)) { throw "Conversion Python is missing: $Python" }
if (-not (Test-Path -LiteralPath $Quantizer)) { throw "Ollama quantizer is missing: $Quantizer" }

New-Item -ItemType Directory -Path $RuntimeDirectory -Force | Out-Null
if ($Force -or -not (Test-Path -LiteralPath $F16)) {
    & $Python $Converter $Source --outfile $F16 --outtype f16 --use-temp-file --model-name "Runeforge-Mk2-7B"
    if ($LASTEXITCODE -ne 0) { throw "F16 GGUF conversion failed" }
}
$F16Hash = (Get-FileHash -LiteralPath $F16 -Algorithm SHA256).Hash.ToLowerInvariant()
if ($F16Hash -ne $Profile.converter.f16_sha256) { throw "F16 GGUF hash mismatch: $F16Hash" }

if ($Force -or -not (Test-Path -LiteralPath $Q4)) {
    & $Quantizer $F16 $Q4 Q4_K_M 4
    if ($LASTEXITCODE -ne 0) { throw "Q4_K_M quantization failed" }
}
$Q4Hash = (Get-FileHash -LiteralPath $Q4 -Algorithm SHA256).Hash.ToLowerInvariant()
if ($Q4Hash -ne $Profile.runtime_model_sha256) { throw "Q4_K_M GGUF hash mismatch: $Q4Hash" }
Write-Host "Verified Runeforge Mk2 7B GGUF runtime: $Q4"


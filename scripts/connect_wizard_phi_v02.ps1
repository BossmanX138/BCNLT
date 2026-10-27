[CmdletBinding()]
param(
    [string]$InputPath = "fixtures/sample.py",
    [string]$ReportPath = "outputs/wizard-phi-v02/model-context.json",
    [int]$MaxProbes = 5,
    [switch]$Rebuild
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$ProfileDirectory = Join-Path $Root "integrations/wizard_phi_v02"
$ProfilePath = Join-Path $ProfileDirectory "profile.json"
$ModelfilePath = Join-Path $ProfileDirectory "Modelfile"
$Profile = Get-Content -LiteralPath $ProfilePath -Raw | ConvertFrom-Json
$SourceModel = $Profile.source_model -replace '/', '\'
$SourceConfig = Join-Path $SourceModel "config.json"
$RuntimeModelFile = $Profile.runtime_model_file -replace '/', '\'

if (-not (Test-Path -LiteralPath $SourceConfig)) {
    throw "Wizard-Phi v0.2 source model is missing: $SourceModel"
}

$ActualConfigHash = (Get-FileHash -LiteralPath $SourceConfig -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ActualConfigHash -ne $Profile.source_config_sha256) {
    throw "Wizard-Phi v0.2 config hash mismatch: $ActualConfigHash"
}

if (-not (Test-Path -LiteralPath $RuntimeModelFile)) {
    throw "Wizard-Phi v0.2 GGUF runtime is missing: $RuntimeModelFile. Run scripts/build_wizard_phi_v02_gguf.ps1 first."
}
$ActualRuntimeHash = (Get-FileHash -LiteralPath $RuntimeModelFile -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ActualRuntimeHash -ne $Profile.runtime_model_sha256) {
    throw "Wizard-Phi v0.2 GGUF hash mismatch: $ActualRuntimeHash"
}

$Installed = @(ollama list 2>&1 | Select-String -SimpleMatch $Profile.runtime_model)
if ($Rebuild -or $Installed.Count -eq 0) {
    & ollama create $Profile.runtime_model --file $ModelfilePath
    if ($LASTEXITCODE -ne 0) {
        throw "Ollama failed to create $($Profile.runtime_model)"
    }
}

$Input = if ([System.IO.Path]::IsPathRooted($InputPath)) { $InputPath } else { Join-Path $Root $InputPath }
$Report = if ([System.IO.Path]::IsPathRooted($ReportPath)) { $ReportPath } else { Join-Path $Root $ReportPath }
$ReportDirectory = Split-Path -Parent $Report
New-Item -ItemType Directory -Path $ReportDirectory -Force | Out-Null

$SmokeBody = @{
    model = $Profile.runtime_model
    stream = $false
    options = @{ temperature = 0; num_predict = 32 }
    messages = @(
        @{ role = "system"; content = "Answer only from the provided context. For definition-line questions, answer in the exact form 'line N'." },
        @{ role = "user"; content = 'Question: What line defines `thing`? Context: L7: DEFINE thing scope=module' }
    )
} | ConvertTo-Json -Depth 5

$Smoke = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/chat" -Method Post -ContentType "application/json" -Body $SmokeBody -TimeoutSec 180
$ReadinessCompletion = ([string]$Smoke.message.content).Trim()
if ($ReadinessCompletion -notmatch '(?i)^line\s+7$') {
    throw "Wizard-Phi v0.2 readiness answer was not 'line 7': $ReadinessCompletion"
}

Push-Location $Root
try {
    python -m bcnlt.cli $Input `
        --model-eval-report $Report `
        --model-eval-provider ollama `
        --model-eval-model $Profile.runtime_model `
        --model-eval-ollama-url "http://127.0.0.1:11434" `
        --model-eval-max-probes $MaxProbes
    if ($LASTEXITCODE -ne 0) {
        throw "BCNLT model evaluation failed"
    }
}
finally {
    Pop-Location
}

$ReportPayload = Get-Content -LiteralPath $Report -Raw | ConvertFrom-Json
$Evidence = [ordered]@{
    schema_version = 1
    integration = $Profile.integration
    runtime_model = $Profile.runtime_model
    source_model = $Profile.source_model
    source_config_sha256 = $ActualConfigHash
    runtime_model_file = $Profile.runtime_model_file
    runtime_model_sha256 = $ActualRuntimeHash
    quantization = $Profile.quantization
    readiness_completion = $ReadinessCompletion
    input = $Input
    report = $Report
    probe_count = $ReportPayload.probe_count
    standard_accuracy = $ReportPayload.standard_accuracy
    bcnlt_accuracy = $ReportPayload.bcnlt_accuracy
    completed_at = [DateTimeOffset]::UtcNow.ToString("o")
}
$EvidencePath = Join-Path $ReportDirectory "integration-smoke.json"
$Evidence | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $EvidencePath -Encoding utf8
Write-Host "BCNLT is connected to Wizard-Phi v0.2. Evidence: $EvidencePath"


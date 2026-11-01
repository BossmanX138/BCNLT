[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$TokenizerArtifact,
    [Parameter(Mandatory = $true)][string]$PackedDataset,
    [Parameter(Mandatory = $true)][string]$SourceSample,
    [Parameter(Mandatory = $true)][string]$PackedAudit,
    [Parameter(Mandatory = $true)][string]$CompatibilityLineage,
    [Parameter(Mandatory = $true)][string]$SmokeLineage,
    [Parameter(Mandatory = $true)][string]$Output,
    [string]$BaseModel = '/home/bossm/models/Runeforge-Mk2-7B',
    [string]$Python = '/home/bossm/.venvs/wizard-phi/bin/python',
    [string]$Distribution = 'Ubuntu'
)

$ErrorActionPreference = 'Stop'
$repoWindows = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$repoWsl = (& wsl.exe -d $Distribution -- wslpath -a ($repoWindows.Replace('\', '/'))).Trim()
if (-not $repoWsl) { throw 'Could not resolve the repository path inside WSL.' }
$compatibilityConfig = "$repoWsl/configs/runeforge_mk2_bcnlt_smoke_v1.json"
$smokeConfig = "$repoWsl/configs/runeforge_mk2_bcnlt_multilingual_smoke_v1.json"
$compatibilityOutput = "$Output-one-step"

& wsl.exe -d $Distribution -- $Python -c "import json,sys; a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2])); assert a.get('passed') is True, 'source sample failed'; assert b.get('passed') is True, 'packed multilingual audit failed'" $SourceSample $PackedAudit
if ($LASTEXITCODE -ne 0) { throw 'Source-sample or packed-corpus gate failed.' }

& wsl.exe -d $Distribution --cd $repoWsl -- $Python -m integrations.runeforge_mk2_tokenizer.train_bcnlt_adapter `
    --base-model $BaseModel --tokenizer-artifact $TokenizerArtifact --packed-dataset $PackedDataset `
    --training-config $compatibilityConfig --expected-lineage $CompatibilityLineage `
    --output $compatibilityOutput --maximum-optimizer-steps 1 --engine unsloth
if ($LASTEXITCODE -ne 0) { throw 'One-step multilingual VRAM compatibility check failed.' }

& wsl.exe -d $Distribution --cd $repoWsl -- $Python -m integrations.runeforge_mk2_tokenizer.train_bcnlt_adapter `
    --base-model $BaseModel --tokenizer-artifact $TokenizerArtifact --packed-dataset $PackedDataset `
    --training-config $smokeConfig --expected-lineage $SmokeLineage --output $Output --engine unsloth
if ($LASTEXITCODE -ne 0) { throw 'The 128-step multilingual smoke failed.' }

& wsl.exe -d $Distribution -- $Python -c "import json,sys; x=json.load(open(sys.argv[1])); assert x.get('promoted') is True, x.get('failure_codes')" "$Output/smoke_promotion.json"
if ($LASTEXITCODE -ne 0) { throw 'The 128-step smoke completed but did not pass promotion.' }
Write-Host "One-step adapter: $compatibilityOutput"
Write-Host "Promoted 128-step adapter: $Output"

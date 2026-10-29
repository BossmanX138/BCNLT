[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$TokenizerArtifact,
    [Parameter(Mandatory = $true)][string]$PackedDataset,
    [Parameter(Mandatory = $true)][string]$ExpectedLineage,
    [Parameter(Mandatory = $true)][string]$Output,
    [string]$BaseModel = '/home/bossm/models/Runeforge-Mk2-7B',
    [string]$Python = '/home/bossm/.venvs/wizard-phi/bin/python',
    [string]$Distribution = 'Ubuntu',
    [int]$MaximumOptimizerSteps = 1
)

$ErrorActionPreference = 'Stop'
$repoWindows = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$repoWsl = (& wsl.exe -d $Distribution -- wslpath -a ($repoWindows.Replace('\', '/'))).Trim()
if (-not $repoWsl) { throw 'Could not resolve the repository path inside WSL.' }
$config = "$repoWsl/configs/runeforge_mk2_bcnlt_smoke_v1.json"

& wsl.exe -d $Distribution -- bash -lc "set -euo pipefail; cd '$repoWsl'; test -x '$Python'; test -d '$BaseModel'; test -d '$TokenizerArtifact'; test -f '$PackedDataset/dataset_manifest.json'; test -f '$ExpectedLineage'; test ! -e '$Output'; '$Python' -m integrations.runeforge_mk2_tokenizer.train_bcnlt_adapter --base-model '$BaseModel' --tokenizer-artifact '$TokenizerArtifact' --packed-dataset '$PackedDataset' --training-config '$config' --expected-lineage '$ExpectedLineage' --output '$Output' --maximum-optimizer-steps '$MaximumOptimizerSteps' --engine auto"
if ($LASTEXITCODE -ne 0) { throw "BCNLT smoke training failed with exit code $LASTEXITCODE." }

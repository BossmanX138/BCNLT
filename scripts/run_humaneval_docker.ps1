[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Samples,

    [Parameter(Mandatory = $true)]
    [string]$Output,

    [string]$Dataset
)

$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$python = Join-Path $project '.venv\Scripts\python.exe'
$evaluator = Join-Path $PSScriptRoot 'evaluate_humaneval_sandboxed.py'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Locked project interpreter is missing: $python"
}
if (-not (Test-Path -LiteralPath $evaluator -PathType Leaf)) {
    throw "Sandboxed HumanEval evaluator is missing: $evaluator"
}

$arguments = @(
    $evaluator,
    '--samples', (Resolve-Path -LiteralPath $Samples).Path,
    '--output', [System.IO.Path]::GetFullPath($Output)
)
if ($Dataset) {
    $arguments += @('--dataset', (Resolve-Path -LiteralPath $Dataset).Path)
}

& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Sandboxed HumanEval evaluation failed with code $LASTEXITCODE."
}

$receipt = Join-Path ([System.IO.Path]::GetFullPath($Output)) 'receipt.json'
if (-not (Test-Path -LiteralPath $receipt -PathType Leaf)) {
    throw 'Sandboxed HumanEval evaluator did not publish its receipt.'
}
Get-Content -LiteralPath $receipt -Raw

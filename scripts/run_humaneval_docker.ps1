[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Samples,

    [Parameter(Mandatory = $true)]
    [string]$Output,

    [string]$Image = 'bosscrafts-humaneval:openai-b9453e8'
)

$ErrorActionPreference = 'Stop'
$samplesPath = (Resolve-Path -LiteralPath $Samples).Path
if ([System.IO.Path]::GetExtension($samplesPath) -ne '.jsonl') {
    throw 'Samples must be a JSONL file.'
}
$outputPath = [System.IO.Path]::GetFullPath($Output)
[System.IO.Directory]::CreateDirectory($outputPath) | Out-Null

$dockerInfo = & docker info --format '{{.ServerVersion}}' 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Docker is not ready: $dockerInfo"
}

$arguments = @(
    'run', '--rm',
    '--network', 'none',
    '--read-only',
    '--tmpfs', '/tmp:rw,noexec,nosuid,size=512m',
    '--cap-drop', 'ALL',
    '--security-opt', 'no-new-privileges',
    '--pids-limit', '512',
    '--memory', '3g',
    '--cpus', '6',
    '--mount', "type=bind,source=$samplesPath,target=/input/samples.jsonl,readonly",
    '--mount', "type=bind,source=$outputPath,target=/output",
    $Image
)

& docker @arguments
if ($LASTEXITCODE -ne 0) {
    throw "HumanEval container exited with code $LASTEXITCODE."
}

$metrics = Join-Path $outputPath 'metrics.txt'
$plainResults = Join-Path $outputPath 'samples.jsonl_results.jsonl'
$gzipResults = Join-Path $outputPath 'samples.jsonl_results.jsonl.gz'
$results = if (Test-Path -LiteralPath $plainResults) { $plainResults } else { $gzipResults }
if (-not (Test-Path -LiteralPath $metrics) -or -not (Test-Path -LiteralPath $results)) {
    throw 'HumanEval evaluator did not produce its expected evidence files.'
}

[pscustomobject]@{
    image = $Image
    metrics = $metrics
    metrics_sha256 = (Get-FileHash -LiteralPath $metrics -Algorithm SHA256).Hash.ToLowerInvariant()
    results = $results
    results_sha256 = (Get-FileHash -LiteralPath $results -Algorithm SHA256).Hash.ToLowerInvariant()
    samples_sha256 = (Get-FileHash -LiteralPath $samplesPath -Algorithm SHA256).Hash.ToLowerInvariant()
} | ConvertTo-Json -Depth 3

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Handoff,
    [string]$Distribution = 'Ubuntu',
    [string]$Python = '/home/bossm/.venvs/wizard-phi-eval/bin/python',
    [string]$ResumeCheckpoint,
    [string]$ResumeSha256,
    [switch]$StartTraining
)

$ErrorActionPreference = 'Stop'
$repoWindows = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$repoWsl = (& wsl.exe -d $Distribution -- wslpath -a ($repoWindows -replace '\', '/')).Trim()
if (-not $repoWsl) { throw 'Could not resolve the repository path inside WSL.' }
$disk = (& wsl.exe -d $Distribution -- bash -lc "df -B1 --output=avail /home | tail -1").Trim()
$gpu = (& wsl.exe -d $Distribution -- nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | Select-Object -First 1).Trim()
if (-not ($disk -match '^\d+$')) { throw 'Could not determine WSL disk availability.' }
if (-not ($gpu -match '^\d+$')) { throw 'Could not determine GPU memory.' }

$arguments = @(
    '-m', 'bcnlt.full_run_gate',
    '--handoff', $Handoff,
    '--disk-free-bytes', $disk,
    '--gpu-memory-mib', $gpu
)
if ($ResumeCheckpoint) {
    if (-not $ResumeSha256) { throw '-ResumeSha256 is required with -ResumeCheckpoint.' }
    $arguments += @('--resume', $ResumeCheckpoint, '--resume-sha256', $ResumeSha256)
}
if ($StartTraining) { $arguments += '--start-training' }

& wsl.exe -d $Distribution --cd $repoWsl -- $Python @arguments
if ($LASTEXITCODE -ne 0) { throw "Full-run gate failed with exit code $LASTEXITCODE." }

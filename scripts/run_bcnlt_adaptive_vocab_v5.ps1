[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Spec,
    [switch]$Resume,
    [switch]$StartTraining,
    [string]$Distribution = "Ubuntu",
    [string]$Python = "/home/bossm/.venvs/wizard-phi-eval/bin/python"
)

$ErrorActionPreference = "Stop"
$Repository = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$SpecPath = (Resolve-Path $Spec).Path
$WslRepository = (& wsl.exe -d $Distribution -- wslpath -a $Repository.Replace('\', '/')).Trim()
$WslSpec = (& wsl.exe -d $Distribution -- wslpath -a $SpecPath.Replace('\', '/')).Trim()
if (-not $WslRepository -or -not $WslSpec) {
    throw "Unable to resolve repository or specification into WSL."
}
$WslScript = "$($WslRepository.TrimEnd('/'))/scripts/run_bcnlt_adaptive_vocab_v5.py"

$Arguments = @(
    "-d", $Distribution, "--", $Python,
    $WslScript,
    "--spec", $WslSpec
)
if ($Resume) {
    $Arguments += "--resume"
}
if ($StartTraining) {
    $Arguments += "--start-training"
}

& wsl.exe @Arguments
exit $LASTEXITCODE

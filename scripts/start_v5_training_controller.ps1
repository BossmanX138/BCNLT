$ErrorActionPreference = 'Stop'

$project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$controller = Join-Path $project 'scripts\run_bosscrafts_v5_training_until_complete.py'
$wrapper = Join-Path $project 'scripts\run_v5_training_controller.cmd'
$status = Join-Path $project 'artifacts\checkpoints\bosscrafts-engineering-v5-clean\training-progress.json'
$logRoot = Join-Path $project 'artifacts\logs\bosscrafts-engineering-v5-clean'
$launchManifest = Join-Path $logRoot 'controller-launch.json'

$existing = @(
    Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -like '*run_bosscrafts_v5_training_until_complete.py*'
    }
)
if ($existing.Count -gt 0) {
    $existing | Select-Object ProcessId, Name, CreationDate, CommandLine |
        ConvertTo-Json -Depth 3
    exit 0
}

New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$stdout = Join-Path $logRoot "$stamp.stdout.log"
$stderr = Join-Path $logRoot "$stamp.stderr.log"
$process = Start-Process `
    -FilePath $env:ComSpec `
    -ArgumentList @('/d', '/c', $wrapper) `
    -WorkingDirectory $project `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -PassThru

$payload = [ordered]@{
    schema_version = 'bosscrafts-v5-controller-launch-v1'
    launched_utc = (Get-Date).ToUniversalTime().ToString('o')
    process_id = $process.Id
    controller = $controller
    status = $status
    stdout = $stdout
    stderr = $stderr
    expected_resume_step = 100
}
$temporary = "$launchManifest.tmp"
$payload | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $temporary -Encoding utf8
Move-Item -LiteralPath $temporary -Destination $launchManifest -Force
$payload | ConvertTo-Json -Depth 4

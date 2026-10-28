[CmdletBinding()]
param(
    [string]$ConfigPath = "configs/runeforge_mk2_bcnlt_v1.json",
    [string]$EvidencePath = "outputs/runeforge-mk2-bcnlt-v1/preflight.json",
    [string]$ProbeJsonPath
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

function Resolve-FromRoot([string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
    return Join-Path $Root $Path
}

function Stop-Preflight([string]$Message) {
    throw "Runeforge BCNLT preflight failed: $Message"
}

function Get-LiveProbe($Config) {
    $Distribution = [string]$Config.base.wsl_distribution
    $Distributions = @(
        wsl.exe --list --quiet 2>$null |
            ForEach-Object { ([string]$_).Replace([string][char]0, [string]'').Trim() } |
            Where-Object { $_ }
    )
    if ($LASTEXITCODE -ne 0 -or $Distribution -notin $Distributions) {
        return [ordered]@{ wsl_available = $false; wsl_distribution = $Distribution }
    }

    $ModelPath = [string]$Config.base.wsl_path
    wsl.exe -d $Distribution -- test -d $ModelPath
    $ModelExists = $LASTEXITCODE -eq 0

    $BaseHashes = [ordered]@{}
    if ($ModelExists) {
        foreach ($Property in $Config.base.required_file_sha256.PSObject.Properties) {
            $FilePath = "$ModelPath/$($Property.Name)"
            $HashOutput = @(wsl.exe -d $Distribution -- sha256sum $FilePath 2>$null)
            if ($LASTEXITCODE -eq 0 -and $HashOutput.Count -gt 0) {
                $BaseHashes[$Property.Name] = ([string]$HashOutput[0]).Split(' ')[0].ToLowerInvariant()
            }
        }
    }

    $Gpu = [ordered]@{
        name = ""
        memory_total_mib = 0
        driver_version = ""
        cuda_available = $false
    }
    $GpuLine = @(wsl.exe -d $Distribution -- nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits 2>$null)
    if ($LASTEXITCODE -eq 0 -and $GpuLine.Count -gt 0) {
        $Parts = ([string]$GpuLine[0]).Split(',') | ForEach-Object { $_.Trim() }
        if ($Parts.Count -ge 3) {
            $Gpu.name = $Parts[0]
            $Gpu.memory_total_mib = [int]$Parts[1]
            $Gpu.driver_version = $Parts[2]
            $Gpu.cuda_available = $true
        }
    }

    $DiskLines = @(wsl.exe -d $Distribution -- df -B1 --output=avail $ModelPath 2>$null)
    $DiskFree = 0L
    if ($LASTEXITCODE -eq 0) {
        $NumericLines = @($DiskLines | ForEach-Object { ([string]$_).Trim() } | Where-Object { $_ -match '^\d+$' })
        if ($NumericLines.Count -gt 0) { $DiskFree = [long]$NumericLines[-1] }
    }

    $Python = [string]$Config.preflight.python_executable
    $Packages = [ordered]@{}
    foreach ($Package in $Config.preflight.required_packages) {
        $PackageName = [string]$Package
        if ($PackageName -notmatch '^[A-Za-z0-9_.-]+$') { Stop-Preflight "invalid package name $PackageName" }
        $Code = "import importlib.metadata as m; print(m.version('$PackageName'))"
        $Version = @(wsl.exe -d $Distribution -- $Python -c $Code 2>$null)
        $Packages[$PackageName] = if ($LASTEXITCODE -eq 0 -and $Version.Count -gt 0) {
            ([string]$Version[-1]).Trim()
        } else { "MISSING" }
    }

    $DatasetAccess = [ordered]@{}
    $WhoAmI = @(wsl.exe -d $Distribution -- $Python -m huggingface_hub.commands.huggingface_cli whoami 2>$null)
    $LoggedIn = $LASTEXITCODE -eq 0 -and (($WhoAmI -join " ") -notmatch 'Not logged in')
    foreach ($Source in $Config.hf_sources) {
        $Dataset = [string]$Source.dataset
        $DatasetAccess[$Dataset] = $false
        if ($LoggedIn) {
            $Revision = [string]$Source.revision
            $ProbePath = [string]$Source.probe_path
            $Code = @"
from huggingface_hub import HfApi
api = HfApi()
info = api.dataset_info(repo_id='$Dataset', revision='$Revision')
items = api.list_repo_tree(repo_id='$Dataset', repo_type='dataset', revision=info.sha, path_in_repo='$ProbePath', recursive=False)
print('ok' if next(iter(items), None) is not None else 'empty')
"@
            $Access = @(wsl.exe -d $Distribution -- $Python -c $Code 2>$null)
            $DatasetAccess[$Dataset] = $LASTEXITCODE -eq 0 -and (($Access -join " ") -match 'ok')
        }
    }

    return [ordered]@{
        wsl_available = $true
        wsl_distribution = $Distribution
        model_path_exists = $ModelExists
        resolved_model_path = $ModelPath
        gpu = $Gpu
        disk_free_bytes = $DiskFree
        python_packages = $Packages
        base_file_sha256 = $BaseHashes
        hf_dataset_access = $DatasetAccess
    }
}

$ResolvedConfig = Resolve-FromRoot $ConfigPath
$ResolvedEvidence = Resolve-FromRoot $EvidencePath
if (-not (Test-Path -LiteralPath $ResolvedConfig -PathType Leaf)) {
    Stop-Preflight "configuration is missing: $ResolvedConfig"
}
if (Test-Path -LiteralPath $ResolvedEvidence) {
    Stop-Preflight "evidence output already exists: $ResolvedEvidence"
}
$Config = Get-Content -LiteralPath $ResolvedConfig -Raw | ConvertFrom-Json
if ($Config.schema_version -ne "runeforge-mk2-bcnlt-v1") {
    Stop-Preflight "unsupported configuration schema"
}

$Probe = if ($ProbeJsonPath) {
    Get-Content -LiteralPath (Resolve-FromRoot $ProbeJsonPath) -Raw | ConvertFrom-Json
} else {
    Get-LiveProbe $Config
}
$Probe = $Probe | ConvertTo-Json -Depth 8 | ConvertFrom-Json

if (-not $Probe.wsl_available) { Stop-Preflight "WSL distribution is unavailable" }
if ([string]$Probe.wsl_distribution -ne [string]$Config.base.wsl_distribution) {
    Stop-Preflight "WSL distribution mismatch"
}
if (-not $Probe.model_path_exists) { Stop-Preflight "Runeforge model path is unavailable" }
if (-not $Probe.gpu.cuda_available) { Stop-Preflight "CUDA GPU is unavailable" }
if ([int]$Probe.gpu.memory_total_mib -lt [int]$Config.preflight.min_gpu_memory_mib) {
    Stop-Preflight "GPU memory is below the configured smoke requirement"
}
if ([long]$Probe.disk_free_bytes -lt [long]$Config.preflight.min_disk_free_bytes) {
    Stop-Preflight "disk space is below the configured requirement"
}

foreach ($Property in $Config.base.required_file_sha256.PSObject.Properties) {
    $ActualProperty = $Probe.base_file_sha256.PSObject.Properties[$Property.Name]
    if ($null -eq $ActualProperty -or ([string]$ActualProperty.Value).ToLowerInvariant() -ne ([string]$Property.Value).ToLowerInvariant()) {
        Stop-Preflight "base checkpoint hash mismatch for $($Property.Name)"
    }
}
foreach ($Package in $Config.preflight.required_packages) {
    $VersionProperty = $Probe.python_packages.PSObject.Properties[[string]$Package]
    if ($null -eq $VersionProperty -or [string]$VersionProperty.Value -eq "MISSING") {
        Stop-Preflight "required Python package is missing: $Package"
    }
}
foreach ($Source in $Config.hf_sources) {
    $AccessProperty = $Probe.hf_dataset_access.PSObject.Properties[[string]$Source.dataset]
    if ($null -eq $AccessProperty -or -not [bool]$AccessProperty.Value) {
        Stop-Preflight "Hugging Face dataset access has not been accepted: $($Source.dataset)"
    }
}

$Evidence = [ordered]@{
    schema_version = "runeforge-mk2-bcnlt-preflight-v1"
    status = "passed"
    config_sha256 = (Get-FileHash -LiteralPath $ResolvedConfig -Algorithm SHA256).Hash.ToLowerInvariant()
    wsl_distribution = [string]$Probe.wsl_distribution
    resolved_model_path = [string]$Probe.resolved_model_path
    gpu = $Probe.gpu
    disk_free_bytes = [long]$Probe.disk_free_bytes
    python_packages = $Probe.python_packages
    base_file_sha256 = $Probe.base_file_sha256
    hf_dataset_access = $Probe.hf_dataset_access
    hf_probe_host = if ($Probe.PSObject.Properties['hf_probe_host']) { [string]$Probe.hf_probe_host } else { 'wsl' }
    hf_authenticated_identity = if ($Probe.PSObject.Properties['hf_authenticated_identity']) { [string]$Probe.hf_authenticated_identity } else { '' }
    completed_at = [DateTimeOffset]::UtcNow.ToString("o")
}
$EvidenceDirectory = Split-Path -Parent $ResolvedEvidence
if ($EvidenceDirectory) { New-Item -ItemType Directory -Path $EvidenceDirectory -Force | Out-Null }
$TemporaryEvidence = "$ResolvedEvidence.tmp"
$Evidence | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $TemporaryEvidence -Encoding utf8
Move-Item -LiteralPath $TemporaryEvidence -Destination $ResolvedEvidence
Write-Output ($Evidence | ConvertTo-Json -Depth 8 -Compress)

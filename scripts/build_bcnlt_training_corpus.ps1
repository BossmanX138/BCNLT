[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$TokenizerArtifact,
    [Parameter(Mandatory = $true)][string]$OutputRoot,
    [string]$BaseTokenizer = '/home/bossm/models/Runeforge-Mk2-7B',
    [string]$HoldoutHashes,
    [string]$Distribution = 'Ubuntu',
    [string]$Python = '/home/bossm/.venvs/wizard-phi-eval/bin/python',
    [int]$SourceLimit = 0,
    [switch]$Resume
)

$ErrorActionPreference = 'Stop'
$repoWindows = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$repoWsl = (& wsl.exe -d $Distribution -- wslpath -a ($repoWindows.Replace('\', '/'))).Trim()
if (-not $repoWsl) { throw 'Could not resolve the repository path inside WSL.' }

function Invoke-WslPhase([string]$Name, [string]$Command, [string]$Output) {
    wsl.exe -d $Distribution -- test -e $Output
    $exists = $LASTEXITCODE -eq 0
    if ($exists -and -not $Resume) { throw "Phase $Name output already exists: $Output" }
    if ($exists -and $Resume) {
        Write-Host "Verified phase output exists; downstream manifests will re-check its hash: $Name"
        return
    }
    & wsl.exe -d $Distribution -- bash -lc "set -euo pipefail; cd '$repoWsl'; $Command"
    if ($LASTEXITCODE -ne 0) { throw "Phase $Name failed with exit code $LASTEXITCODE." }
    & wsl.exe -d $Distribution -- test -e $Output
    if ($LASTEXITCODE -ne 0) { throw "Phase $Name did not create $Output." }
}

& (Join-Path $PSScriptRoot 'preflight_bcnlt_runeforge.ps1')

& wsl.exe -d $Distribution -- test -d $OutputRoot
if ($LASTEXITCODE -ne 0) { & wsl.exe -d $Distribution -- mkdir -p $OutputRoot }
$database = "$OutputRoot/corpus.sqlite"
$curriculum = "$OutputRoot/curriculum.jsonl"
$wildcardMemory = "$OutputRoot/learned-definitions.jsonl"
$validation = "$OutputRoot/corpus_validation.json"
$accepted = "$OutputRoot/accepted.jsonl"
$packed = "$OutputRoot/packed"
$registry = "$repoWsl/configs/bcnlt_corpus_sources_v1.json"

$holdoutArg = if ($HoldoutHashes) { " --holdout-hashes '$HoldoutHashes'" } else { '' }
$limitArg = if ($SourceLimit -gt 0) { " --limit $SourceLimit" } else { '' }
Invoke-WslPhase 'ingestion' "'$Python' -m scripts.ingest_bcnlt_corpus --config '$registry' --database '$database'$holdoutArg$limitArg" $database
Invoke-WslPhase 'curriculum' "'$Python' -m scripts.generate_bcnlt_curriculum --database '$database' --output '$curriculum' --wildcard-memory '$wildcardMemory'" $curriculum

& wsl.exe -d $Distribution -- test -e $validation
$validationExists = $LASTEXITCODE -eq 0
if ($validationExists -and -not $Resume) { throw "Validation output already exists: $validation" }
if (-not $validationExists) {
    & wsl.exe -d $Distribution -- bash -lc "set -euo pipefail; cd '$repoWsl'; '$Python' -m scripts.validate_bcnlt_corpus --database '$database' --curriculum '$curriculum' --tokenizer-artifact '$TokenizerArtifact' --base-tokenizer '$BaseTokenizer' --report '$validation' --accepted-output '$accepted'$holdoutArg"
    if ($LASTEXITCODE -ne 0) { throw 'Corpus admission gates did not pass; full-run materialization stopped.' }
}
Invoke-WslPhase 'packing' "'$Python' -m scripts.export_bcnlt_training_shards --records '$accepted' --tokenizer-artifact '$TokenizerArtifact' --base-tokenizer '$BaseTokenizer' --output '$packed'" "$packed/dataset_manifest.json"

$manifest = "$OutputRoot/corpus_pipeline_manifest.json"
& wsl.exe -d $Distribution -- $Python -c "import hashlib,json,pathlib,sys; root=pathlib.Path(sys.argv[1]); names=['corpus.sqlite','curriculum.jsonl','learned-definitions.jsonl','corpus_validation.json','accepted.jsonl','packed/dataset_manifest.json']; h=lambda p: hashlib.sha256(p.read_bytes()).hexdigest(); payload={'schema_version':'bcnlt-corpus-pipeline-v1','artifacts':{n:{'path':str(root/n),'sha256':h(root/n)} for n in names}}; out=root/'corpus_pipeline_manifest.json'; out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')" $OutputRoot
if ($LASTEXITCODE -ne 0) { throw 'Could not write the corpus pipeline manifest.' }
Write-Host "Corpus pipeline manifest: $manifest"

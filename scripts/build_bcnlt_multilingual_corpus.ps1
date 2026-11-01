[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$TokenizerArtifact,
    [Parameter(Mandatory = $true)][string]$OutputRoot,
    [string]$BaseTokenizer = '/home/bossm/models/Runeforge-Mk2-7B',
    [string]$HoldoutHashes,
    [ValidateSet('smoke', 'full')][string]$PolicyProfile = 'smoke',
    [string]$Distribution = 'Ubuntu',
    [string]$Python = '/home/bossm/.venvs/wizard-phi-eval/bin/python',
    [int]$RowsPerLanguage = 100,
    [int]$SourceLimit = 4000,
    [switch]$Resume
)

$ErrorActionPreference = 'Stop'
$repoWindows = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$repoWsl = (& wsl.exe -d $Distribution -- wslpath -a ($repoWindows.Replace('\', '/'))).Trim()
if (-not $repoWsl) { throw 'Could not resolve the repository path inside WSL.' }
$registryWindows = Join-Path $repoWindows 'configs\bcnlt_corpus_sources_v2.json'
$registryHash = (Get-FileHash -LiteralPath $registryWindows -Algorithm SHA256).Hash.ToLowerInvariant()
$trainingConfigName = if ($PolicyProfile -eq 'full') {
    'runeforge_mk2_bcnlt_multilingual_full_v1.json'
} else {
    'runeforge_mk2_bcnlt_multilingual_smoke_v1.json'
}
$trainingConfigWindows = Join-Path $repoWindows "configs\$trainingConfigName"
$trainingConfigHash = (Get-FileHash -LiteralPath $trainingConfigWindows -Algorithm SHA256).Hash.ToLowerInvariant()
$tokenizerHash = (& wsl.exe -d $Distribution -- sha256sum "$TokenizerArtifact/bcnlt_tokenizer_manifest.json").Split(' ')[0]
if (-not ($tokenizerHash -match '^[0-9a-f]{64}$')) { throw 'Could not hash the tokenizer manifest.' }
$holdoutHash = 'none'
if ($HoldoutHashes) {
    $holdoutHash = (& wsl.exe -d $Distribution -- sha256sum $HoldoutHashes).Split(' ')[0]
    if (-not ($holdoutHash -match '^[0-9a-f]{64}$')) { throw 'Could not hash the holdout list.' }
}
$codeRevision = (& git -C $repoWindows rev-parse HEAD).Trim()
$identityBytes = [Text.Encoding]::UTF8.GetBytes(
    "$registryHash|$tokenizerHash|$trainingConfigHash|$codeRevision|$PolicyProfile|$SourceLimit|$RowsPerLanguage|$holdoutHash"
)
$identityHash = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($identityBytes)).ToLowerInvariant()
$runRoot = "$($OutputRoot.TrimEnd('/'))/multilingual-$PolicyProfile-$($identityHash.Substring(0, 16))"

function Invoke-WslPhase([string]$Name, [string]$Command, [string]$Output) {
    & wsl.exe -d $Distribution -- test -e $Output
    $exists = $LASTEXITCODE -eq 0
    if ($exists -and -not $Resume) { throw "Phase $Name output already exists: $Output" }
    if ($exists) {
        Write-Host "Reusing content-addressed phase output: $Name"
        return
    }
    & wsl.exe -d $Distribution -- bash -lc "set -euo pipefail; cd '$repoWsl'; $Command"
    if ($LASTEXITCODE -ne 0) { throw "Phase $Name failed with exit code $LASTEXITCODE." }
    & wsl.exe -d $Distribution -- test -e $Output
    if ($LASTEXITCODE -ne 0) { throw "Phase $Name did not create $Output." }
}

& wsl.exe -d $Distribution -- mkdir -p $runRoot
if ($LASTEXITCODE -ne 0) { throw 'Could not create the content-addressed run root.' }
$registry = "$repoWsl/configs/bcnlt_corpus_sources_v2.json"
$trainingConfig = "$repoWsl/configs/$trainingConfigName"
$sample = "$runRoot/source-sample.json"
$database = "$runRoot/corpus.sqlite"
$curriculum = "$runRoot/curriculum.jsonl"
$wildcardMemory = "$runRoot/learned-definitions"
$validation = "$runRoot/corpus-validation.json"
$accepted = "$runRoot/accepted.jsonl"
$packed = "$runRoot/packed"
$packedAudit = "$runRoot/per-language-validation.json"
$semanticSpecs = "$runRoot/semantic-specs.json"
$holdoutArg = if ($HoldoutHashes) { " --holdout-hashes '$HoldoutHashes'" } else { '' }
$limitArg = if ($SourceLimit -gt 0) { " --limit $SourceLimit" } else { '' }

Invoke-WslPhase 'source-sample' "'$Python' -m scripts.sample_bcnlt_multilingual_sources --config '$registry' --rows-per-language $RowsPerLanguage --minimum-admitted 25 --output '$sample'" $sample
Invoke-WslPhase 'ingestion' "'$Python' -m scripts.ingest_bcnlt_corpus --config '$registry' --database '$database'$holdoutArg$limitArg" $database
Invoke-WslPhase 'curriculum' "'$Python' -m scripts.generate_bcnlt_curriculum --database '$database' --output '$curriculum' --wildcard-memory '$wildcardMemory'" $curriculum

& wsl.exe -d $Distribution -- test -e $validation
if ($LASTEXITCODE -ne 0) {
    & wsl.exe -d $Distribution -- bash -lc "set -euo pipefail; cd '$repoWsl'; '$Python' -m scripts.validate_bcnlt_corpus --database '$database' --curriculum '$curriculum' --tokenizer-artifact '$TokenizerArtifact' --base-tokenizer '$BaseTokenizer' --report '$validation' --accepted-output '$accepted' --policy-profile '$PolicyProfile'$holdoutArg"
    if ($LASTEXITCODE -ne 0) { throw 'Multilingual corpus admission gates failed.' }
} elseif (-not $Resume) {
    throw "Validation output already exists: $validation"
}
Invoke-WslPhase 'packing' "'$Python' -m scripts.export_bcnlt_training_shards --records '$accepted' --tokenizer-artifact '$TokenizerArtifact' --base-tokenizer '$BaseTokenizer' --output '$packed'" "$packed/dataset_manifest.json"
Invoke-WslPhase 'packed-audit' "'$Python' -m bcnlt.multilingual_training --packed '$packed' --config '$trainingConfig' --output '$packedAudit'" $packedAudit
& wsl.exe -d $Distribution -- test -e $semanticSpecs
$semanticExists = $LASTEXITCODE -eq 0
if (-not $semanticExists) {
    $runRootWindows = (& wsl.exe -d $Distribution -- wslpath -w $runRoot).Trim()
    if (-not $runRootWindows) { throw 'Could not map the run root back to Windows.' }
    $registryObject = Get-Content -LiteralPath $registryWindows -Raw | ConvertFrom-Json
    $hashes = [ordered]@{}
    foreach ($source in $registryObject.sources) {
        if ($source.name -eq 'bigcode/the-stack-dedup') {
            $hashes[$source.canonical_language] = $source.semantic_spec_sha256
        }
    }
    $payload = [ordered]@{
        schema_version = 'bcnlt-semantic-spec-lineage-v1'
        semantic_spec_sha256 = $hashes
    }
    $semanticWindows = Join-Path $runRootWindows 'semantic-specs.json'
    [IO.File]::WriteAllText(
        $semanticWindows,
        (($payload | ConvertTo-Json -Depth 5) + "`n"),
        [Text.UTF8Encoding]::new($false)
    )
} elseif (-not $Resume) {
    throw "Semantic-spec lineage already exists: $semanticSpecs"
}

$manifest = "$runRoot/corpus-pipeline-manifest.json"
& wsl.exe -d $Distribution -- $Python -c "import hashlib,json,pathlib,sys; root=pathlib.Path(sys.argv[1]); names=['source-sample.json','corpus.sqlite','curriculum.jsonl','corpus-validation.json','accepted.jsonl','packed/dataset_manifest.json','per-language-validation.json','semantic-specs.json']; h=lambda p: hashlib.sha256(p.read_bytes()).hexdigest(); payload={'schema_version':'bcnlt-multilingual-corpus-pipeline-v1','identity':sys.argv[2],'artifacts':{n:{'path':str(root/n),'sha256':h(root/n)} for n in names}}; (root/'corpus-pipeline-manifest.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')" $runRoot $identityHash
if ($LASTEXITCODE -ne 0) { throw 'Could not write the multilingual corpus manifest.' }
Write-Host "Multilingual corpus root: $runRoot"
Write-Host "Pipeline manifest: $manifest"

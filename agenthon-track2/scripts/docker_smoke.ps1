# Harness-exact container smoke test (Windows / Docker Desktop).
#   powershell -ExecutionPolicy Bypass -File scripts\docker_smoke.ps1 [-Image agenthon-t2:1.0.0] [-All]
# Mirrors DEVELOPMENT-RUNTIME.md: --network=none, read-only root, uid 65534, 64 MiB noexec /tmp,
# 256 PIDs, unit dir mounted read-only at /input, fresh /output; then the OFFICIAL gates g0-g3.
param([string]$Image = "agenthon-t2:1.0.0", [switch]$All)

$dockerBin = "$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin"
if (Test-Path $dockerBin) { $env:Path = "$dockerBin;$env:Path" }
$repo = Split-Path -Parent $PSScriptRoot
$root = Split-Path -Parent $repo
$unitsDir = Join-Path $root "_reference\track2-forecasting-public\units"
$py = Join-Path $root ".venv313\Scripts\python.exe"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"

if ($All) {
    $names = Get-ChildItem $unitsDir -Directory | Where-Object { Test-Path (Join-Path $_.FullName "card.toml") } |
        ForEach-Object { $_.Name }
} else {
    $names = @("t2-F3-bear-flattener-2022", "t2-F1-cpi-glidepath-2023", "t2-F2-cnh-stress-2015",
               "t2-F4-covid-mkt-2020", "t2-EXAMPLE-ust-curve-1m")
}

$pass = 0; $fail = @()
foreach ($name in $names) {
    $unitPath = Join-Path $unitsDir $name
    $outPath = Join-Path $repo "artifacts\docker_out\$stamp\$name"
    New-Item -ItemType Directory -Path $outPath -Force | Out-Null
    $asof = (Select-String -Path (Join-Path $unitPath "card.toml") -Pattern '^data_cutoff\s*=\s*"(.+)"').Matches[0].Groups[1].Value
    $t0 = Get-Date
    $log = docker run --rm --platform linux/amd64 --network=none --read-only --user 65534:65534 `
        --cap-drop=ALL --security-opt no-new-privileges --pids-limit 256 `
        --tmpfs "/tmp:rw,noexec,nosuid,nodev,size=64m" --cpus=4 --memory=8g `
        -e QFBENCH_SEED=0 -e QFBENCH_NETWORK=none `
        -v "${unitPath}:/input:ro" -v "${outPath}:/output" $Image `
        forecast --panels /input/panels/ --text /input/text/ --asof $asof --out /output/forecast.parquet 2>&1 |
        ForEach-Object { "$_" }
    $secs = ((Get-Date) - $t0).TotalSeconds
    $files = (Get-ChildItem $outPath | ForEach-Object { $_.Name }) -join ","
    $gate = & $py (Join-Path $repo "scripts\validate_submission.py") --card (Join-Path $unitPath "card.toml") --out $outPath | Select-Object -Last 1
    $line = ($log | Select-String "agenthon\] $name" | Select-Object -Last 1)
    "{0,-36} {1,5:N1}s  {2}  | {3}" -f $name, $secs, $gate, $line
    if ($gate -match "admissible 1/1") { $pass++ } else { $fail += $name; $log | Select-Object -Last 15 }
}
"`nCONTAINER SMOKE: $pass/$($names.Count) admissible" + $(if ($fail) { "  FAILED: $($fail -join ', ')" } else { "" })

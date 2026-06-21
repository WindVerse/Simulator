<#
.SYNOPSIS
    Builds the Wind Visualization System Windows installer with Inno Setup.

.DESCRIPTION
    Bundles the application, its portable Python runtime (python\), ML model
    files and the OpenFOAM sample dataset into a single distributable installer.
    Re-run this after code changes (bump -Version) to cut a new release.

.PARAMETER Version
    Version stamped into the installer and output filename. Default: 1.0.0

.PARAMETER Fast
    Use fast (lower-ratio) compression for quicker test builds.

.EXAMPLE
    pwsh installer\build.ps1 -Version 1.0.0
    pwsh installer\build.ps1 -Version 1.0.1 -Fast
#>
[CmdletBinding()]
param(
    [string]$Version = "1.0.0",
    [switch]$Fast
)

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Resolve paths (script lives in <repo>\installer)
# ---------------------------------------------------------------------------
$InstallerDir = $PSScriptRoot
$RepoRoot     = Split-Path -Parent $InstallerDir
$IssPath      = Join-Path $InstallerDir "WindVisualizationSystem.iss"
$OutputDir    = Join-Path $InstallerDir "Output"

Write-Host "=== Wind Visualization System - installer build ===" -ForegroundColor Cyan
Write-Host "Repo root : $RepoRoot"
Write-Host "Version   : $Version"
Write-Host "Mode      : $(if ($Fast) { 'fast compression' } else { 'max compression' })"

# ---------------------------------------------------------------------------
# Locate the Inno Setup compiler (ISCC.exe)
# ---------------------------------------------------------------------------
$isccCandidates = @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles}\Inno Setup 6\ISCC.exe",
    "${env:LOCALAPPDATA}\Programs\Inno Setup 6\ISCC.exe"  # winget user-scope install
)
$Iscc = $null
foreach ($c in $isccCandidates) {
    if ($c -and (Test-Path $c)) { $Iscc = $c; break }
}
if (-not $Iscc) {
    $cmd = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($cmd) { $Iscc = $cmd.Source }
}
if (-not $Iscc) {
    Write-Host ""
    Write-Host "ERROR: Inno Setup compiler (ISCC.exe) was not found." -ForegroundColor Red
    Write-Host "Install Inno Setup 6, then re-run this script:" -ForegroundColor Yellow
    Write-Host "    winget install --id JRSoftware.InnoSetup -e" -ForegroundColor Yellow
    Write-Host "(or download from https://jrsoftware.org/isdl.php )" -ForegroundColor Yellow
    exit 1
}
Write-Host "ISCC      : $Iscc"

# ---------------------------------------------------------------------------
# Sanity-check the payload before a long compile
# ---------------------------------------------------------------------------
$required = @(
    "python\python.exe",
    "python\pythonw.exe",
    "main.py",
    "models\best_model_1.pth",
    "objects\flag.obj",
    "wind_data\sample_openfoam_output\postProcessing\surfaces"
)
$missing = @()
foreach ($r in $required) {
    if (-not (Test-Path (Join-Path $RepoRoot $r))) { $missing += $r }
}
if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Host "ERROR: required payload is missing from the working copy:" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "    - $_" -ForegroundColor Red }
    Write-Host "Ensure the portable 'python\' folder and sample data are present (see run.bat)." -ForegroundColor Yellow
    exit 1
}
Write-Host "Payload   : OK (python runtime, models, sample data present)"

# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------
$isccArgs = @("/DMyAppVersion=$Version")
if ($Fast) { $isccArgs += "/DFastBuild" }
$isccArgs += $IssPath

Write-Host ""
Write-Host "Compiling installer (this can take many minutes for ~6.7 GB)..." -ForegroundColor Cyan
& $Iscc @isccArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: ISCC failed with exit code $LASTEXITCODE." -ForegroundColor Red
    exit $LASTEXITCODE
}

# ---------------------------------------------------------------------------
# Report output
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "=== Build succeeded ===" -ForegroundColor Green
$produced = Get-ChildItem -Path $OutputDir -Filter "WindVisualizationSystem-Setup-$Version*" -ErrorAction SilentlyContinue
if ($produced) {
    foreach ($f in $produced) {
        $sizeGB = [math]::Round($f.Length / 1GB, 2)
        Write-Host ("    {0}  ({1} GB)" -f $f.FullName, $sizeGB)
    }
    Write-Host ""
    Write-Host "Distribute ALL of the above files together (the .exe needs its .bin slices)." -ForegroundColor Yellow
} else {
    Write-Host "    (output not found under $OutputDir - check ISCC log above)" -ForegroundColor Yellow
}

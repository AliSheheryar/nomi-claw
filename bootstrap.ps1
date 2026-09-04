# bootstrap.ps1 — install EVERYTHING from a fresh Windows install.
#
# Assumes: Windows 10/11, admin rights (right-click PowerShell -> Run as Administrator).
# Installs: winget (if missing), Git, GitHub CLI, Python 3.11 (+launcher),
#           ffmpeg, Visual C++ Build Tools, NVIDIA driver check, then runs setup.ps1.
#
# Usage (from an ADMIN PowerShell, in the folder you want the project):
#     Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
#     .\bootstrap.ps1
#
# Reboots may be required after Build Tools / NVIDIA driver install.

$ErrorActionPreference = "Stop"

function Test-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    (New-Object Security.Principal.WindowsPrincipal($id)).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator
    )
}

function Refresh-Path {
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path","User")
}

function Have($cmd) { [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }

function WingetInstall($id, $name) {
    Write-Host "  installing $name ..."
    winget install -e --id $id --accept-source-agreements --accept-package-agreements --silent
    Refresh-Path
}

# ---------------------------------------------------------------------------
if (-not (Test-Admin)) {
    throw "Run this from an ADMIN PowerShell (right-click -> Run as Administrator)."
}

Write-Host "==[0/8]== TLS 1.2 for downloads"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# ---------------------------------------------------------------------------
Write-Host "==[1/8]== winget check"
if (-not (Have winget)) {
    Write-Host "  winget missing. Installing App Installer from Microsoft..."
    $url = "https://aka.ms/getwinget"
    $out = "$env:TEMP\AppInstaller.msixbundle"
    Invoke-WebRequest -Uri $url -OutFile $out -UseBasicParsing
    Add-AppxPackage -Path $out
    Refresh-Path
    if (-not (Have winget)) { throw "winget still missing after App Installer install. Reboot and rerun." }
}
Write-Host "  winget: $(winget --version)"

# ---------------------------------------------------------------------------
Write-Host "==[2/8]== Git"
if (-not (Have git)) { WingetInstall "Git.Git" "Git" }
Write-Host "  git: $(git --version)"

# ---------------------------------------------------------------------------
Write-Host "==[3/8]== GitHub CLI"
if (-not (Have gh)) { WingetInstall "GitHub.cli" "GitHub CLI" }
Write-Host "  gh:  $(gh --version | Select-Object -First 1)"

# ---------------------------------------------------------------------------
Write-Host "==[4/8]== Python 3.11 (+ launcher)"
$needPy = -not (Have python) -or -not ((python --version 2>&1) -match "Python 3\.11")
if ($needPy) {
    WingetInstall "Python.Python.3.11" "Python 3.11"
}
if (-not (Have py)) { WingetInstall "Python.Launcher" "Python Launcher" }
Refresh-Path
Write-Host "  python: $(python --version 2>&1)"

# ---------------------------------------------------------------------------
Write-Host "==[5/8]== ffmpeg"
if (-not (Have ffmpeg)) { WingetInstall "Gyan.FFmpeg" "ffmpeg" }
Write-Host "  ffmpeg: $((ffmpeg -version 2>&1 | Select-Object -First 1))"

# ---------------------------------------------------------------------------
Write-Host "==[6/8]== Visual C++ Build Tools (needed for autoawq / native wheels)"
$vswhere = "$env:ProgramFiles(x86)\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    Write-Host "  installing Build Tools (this can take several minutes)..."
    winget install -e --id Microsoft.VisualStudio.2022.BuildTools --silent `
        --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended" `
        --accept-source-agreements --accept-package-agreements
} else {
    Write-Host "  Visual Studio installer already present."
}

# ---------------------------------------------------------------------------
Write-Host "==[7/8]== NVIDIA GPU visibility check"
if (Have nvidia-smi) {
    nvidia-smi | Select-Object -First 12
} else {
    Write-Warning "  nvidia-smi not found. Install/refresh drivers from:"
    Write-Warning "    https://www.nvidia.com/download/index.aspx"
    Write-Warning "  After install, reboot, then rerun this script."
}

# ---------------------------------------------------------------------------
Write-Host "==[8/8]== Running project setup.ps1"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here
if (-not (Test-Path .\setup.ps1)) {
    throw "setup.ps1 not found in $here. Are you running this from the repo root?"
}
& .\setup.ps1

Write-Host ""
Write-Host "===================================================================="
Write-Host " Bootstrap complete."
Write-Host " Activate venv:   . .\.venv\Scripts\Activate.ps1"
Write-Host " Run pipeline:    python nomi_claw.py `"C:\path\to\video.mp4`""
Write-Host "===================================================================="

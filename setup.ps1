# setup.ps1 — one-shot setup for the wedding-clip pipeline on Windows + RTX 4060
# Run from D:\nomi-claw:   powershell -ExecutionPolicy Bypass -File .\setup.ps1

$ErrorActionPreference = "Stop"

Write-Host "==[1/7]== Python 3.11 check"
$PyCmd = $null
try { $null = & py -3.11 -V 2>$null; if ($LASTEXITCODE -eq 0) { $PyCmd = @("py","-3.11") } } catch {}
if (-not $PyCmd) {
    try {
        $ver = & python --version 2>&1
        if ($ver -match "Python 3\.1[1-9]") { $PyCmd = @("python") }
    } catch {}
}
if (-not $PyCmd) { throw "Python 3.11+ not found. Install: winget install -e --id Python.Python.3.11  then reopen shell." }
Write-Host "      using: $($PyCmd -join ' ')"

Write-Host "==[2/7]== ffmpeg check"
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    throw "ffmpeg not on PATH. Install: winget install Gyan.FFmpeg  then reopen shell."
}

Write-Host "==[3/7]== venv"
if (-not (Test-Path .\.venv)) { & $PyCmd[0] $PyCmd[1..($PyCmd.Length-1)] -m venv .venv }
. .\.venv\Scripts\Activate.ps1

Write-Host "==[4/7]== pip + torch (CUDA 12.4 wheels)"
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

Write-Host "==[5/7]== python deps"
pip install `
    "transformers>=4.49,<4.52" `
    accelerate `
    autoawq `
    qwen-vl-utils `
    av `
    decord `
    ffmpeg-python `
    "scenedetect[opencv]" `
    pillow

Write-Host "==[6/7]== CUDA visibility check"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible to torch'; print('CUDA OK:', torch.cuda.get_device_name(0), '| VRAM GB:', round(torch.cuda.get_device_properties(0).total_memory/1e9,1))"

Write-Host "==[7/7]== Pre-downloading Qwen2.5-VL-7B-Instruct-AWQ to .\models  (this is ~5-6 GB)"
$env:HF_HOME = (Resolve-Path .\models).Path
python -c "from transformers import AutoProcessor, AutoModelForImageTextToText; m='Qwen/Qwen2.5-VL-7B-Instruct-AWQ'; AutoProcessor.from_pretrained(m); AutoModelForImageTextToText.from_pretrained(m); print('model cached')"

Write-Host ""
Write-Host "Setup complete."
Write-Host "Activate later:  . .\.venv\Scripts\Activate.ps1"
Write-Host "Run:             python ground_video.py path\to\video.mp4"

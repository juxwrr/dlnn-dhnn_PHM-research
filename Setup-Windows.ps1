$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"
py -3.12 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw "Install Python 3.12 from python.org first." }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip setup failed." }
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
& .\.venv\Scripts\python.exe -m conservative_training.portable check --device cpu
if ($LASTEXITCODE -ne 0) { throw "JAX CPU check failed." }
Write-Host "Setup complete. Download and extract training-transfer.zip from Releases, then run Start-Training.ps1."

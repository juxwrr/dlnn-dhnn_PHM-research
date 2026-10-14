param([ValidateSet("one-term","coupled","both")][string]$Model="both",[int[]]$Seeds=@(0,1,2,3,4),[switch]$Controls)
$ErrorActionPreference="Stop"
Set-Location $PSScriptRoot
$env:PYTHONUTF8="1"
$arguments=@("-m","conservative_training.portable","study","--model",$Model,"--device","cpu","--seeds")+$Seeds
if ($Controls) { $arguments += "--controls" }
& .\.venv\Scripts\python.exe @arguments
if ($LASTEXITCODE -ne 0) { throw "Training stopped. Check training_results/study_status.json and the task log." }

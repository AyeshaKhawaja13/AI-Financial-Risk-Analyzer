# Usage: .\run.ps1 setup | eda | train | demo | api | dashboard | test   (uses the `python` on your PATH)
param([Parameter(Mandatory=$true)][string]$task)
switch ($task) {
  "setup"     { python -m pip install -r requirements.txt }
  "eda"       { python -m src.eda }
  "train"     { python -m src.train }
  "demo"      { python -m src.eda --demo; python -m src.train --demo --n-iter 5 }
  "api"       { python -m uvicorn api.main:app --reload }
  "dashboard" { python -m streamlit run app/dashboard.py }
  "test"      { python -m pytest -q }
  default     { Write-Host "Unknown task: $task" }
}

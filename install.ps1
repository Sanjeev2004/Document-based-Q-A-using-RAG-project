Write-Host "Installing dependencies..." -ForegroundColor Green
Write-Host ""

Write-Host "Step 1: Installing numpy (with pre-built wheel)..." -ForegroundColor Yellow
pip install numpy>=2.0.0

Write-Host ""
Write-Host "Step 2: Installing remaining dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host ""
Write-Host "Installation complete!" -ForegroundColor Green

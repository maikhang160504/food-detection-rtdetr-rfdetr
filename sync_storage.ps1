# PowerShell Script: Tự động đồng bộ Checkpoints & Logs từ Modal Volume về máy tính
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  DONG BO KET QUA HUAN LUYEN TU MODAL VOLUME VE MAY TINH" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

$VolumeName = "food-detection-training"
$DestFolder = ".\synced_results"

# Tạo các thư mục nhận dữ liệu
New-Item -ItemType Directory -Force -Path "$DestFolder\checkpoints" | Out-Null
New-Item -ItemType Directory -Force -Path "$DestFolder\logs" | Out-Null
New-Item -ItemType Directory -Force -Path "$DestFolder\outputs" | Out-Null

Write-Host "`n[1/3] Dang tai Checkpoints (best.pt, best.pth, last.pt, last.pth)..." -ForegroundColor Yellow
modal volume get $VolumeName /checkpoints/ "$DestFolder\checkpoints"

Write-Host "`n[2/3] Dang tai Logs CSV va Bao cao so sanh..." -ForegroundColor Yellow
modal volume get $VolumeName /logs/ "$DestFolder\logs"

Write-Host "`n[3/3] Dang tai Ket qua danh gia tren tap Test..." -ForegroundColor Yellow
modal volume get $VolumeName /outputs/ "$DestFolder\outputs"

Write-Host "`n================================================================" -ForegroundColor Green
Write-Host "  DONG BO HOAN TAT! Tat ca du lieu nam tai: $DestFolder" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green

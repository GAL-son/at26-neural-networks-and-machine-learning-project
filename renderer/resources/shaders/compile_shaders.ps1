# Ensure the target directory exists
if (-not (Test-Path "target")) {
    New-Item -ItemType Directory -Path "target" | Out-Null
}

Write-Host "Starting shader compilation..." -ForegroundColor Cyan

# Find all files in the source folder
Get-ChildItem -Path "source" -File | ForEach-Object {
    $filename = $_.Name
    # Append .spv to the exact filename
    $outname = "$filename.spv"
    
    # Run glslc
    glslc $_.FullName -o "target\$outname"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[SUCCESS] Compiled: $filename -> target\$outname" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] Failed to compile: $filename" -ForegroundColor Red
    }
}

Write-Host "Shader compilation finished!" -ForegroundColor Cyan
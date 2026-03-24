<#
.SYNOPSIS
    Kill stale Streamlit processes occupying port 8501 (or any Streamlit process).
.DESCRIPTION
    Finds all running Streamlit processes and terminates them cleanly,
    freeing port 8501 for a fresh start.
#>

Write-Host "Looking for running Streamlit processes..." -ForegroundColor Cyan

# Check for anything holding port 8501
$portPids = @()
try {
    $connections = Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue
    if ($connections) {
        $portPids = $connections | Select-Object -ExpandProperty OwningProcess -Unique
    }
} catch {
    # Ignore errors
}

$allProcs = @()

foreach ($procId in $portPids) {
    try {
        $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
        if ($proc) {
            $allProcs += $proc
        }
    } catch {
        # Ignore
    }
}

# Also look for streamlit-named processes
$stProcs = Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -like "*streamlit*" }
if ($stProcs) {
    $allProcs += $stProcs
}

# Deduplicate
$allProcs = $allProcs | Sort-Object Id -Unique

if ($allProcs.Count -eq 0) {
    Write-Host "No Streamlit processes found. Port 8501 is free." -ForegroundColor Green
    exit 0
}

Write-Host "Found $($allProcs.Count) process(es) to kill:" -ForegroundColor Yellow
foreach ($p in $allProcs) {
    Write-Host "  PID $($p.Id) - $($p.ProcessName)" -ForegroundColor Yellow
}

foreach ($p in $allProcs) {
    try {
        Stop-Process -Id $p.Id -Force -ErrorAction Stop
        Write-Host "  Killed PID $($p.Id)" -ForegroundColor Red
    } catch {
        Write-Host "  Could not kill PID $($p.Id): $_" -ForegroundColor DarkYellow
    }
}

Write-Host "Done. Port 8501 should now be available." -ForegroundColor Green

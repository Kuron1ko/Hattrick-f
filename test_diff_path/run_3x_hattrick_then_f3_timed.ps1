[CmdletBinding()]
param(
    [double]$HoursPerMethod = 3.0,
    [int]$Seed = 490,
    [int]$Epochs = 40,
    [string]$Device = "cuda",
    [string]$Python = "D:\kuroresearch\.venv-hattrick\Scripts\python.exe",
    [switch]$SkipHattrick,
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"

if ($HoursPerMethod -le 0) { throw "HoursPerMethod must be positive." }
if ($Epochs -le 0) { throw "Epochs must be positive." }
if ($Device -notin @("auto", "cpu", "cuda")) {
    throw "Device must be auto, cpu, or cuda."
}

$repo = Split-Path -Parent $PSScriptRoot
$hattrickScript = Join-Path $repo "test_diff_path\full_geant_3x_hattrick\run_full_experiment.py"
$f3Script = Join-Path $repo "test_diff_path\full_geant_3x_hattrick_f3\run_full_experiment.py"
$hattrickRun = Join-Path $repo "test_diff_path\full_geant_3x_hattrick\artifacts\hattrick\seed_$Seed"
$f3Run = Join-Path $repo "test_diff_path\full_geant_3x_hattrick_f3\artifacts\seed_$Seed"
$logDir = Join-Path $repo "test_diff_path\timed_training_logs"

foreach ($required in @($Python, $hattrickScript, $f3Script)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required file does not exist: $required"
    }
}
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

$otherTraining = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(?:\.exe)?$' -and
    $_.CommandLine -match 'full_geant_3x_(hattrick|hattrick_f3)[\\/]run_full_experiment\.py' -and
    $_.CommandLine -match '(--stage\s+train|--stage\s+all)'
}
if ($otherTraining) {
    $details = ($otherTraining | ForEach-Object {
        "PID=$($_.ProcessId) $($_.CommandLine)"
    }) -join [Environment]::NewLine
    throw "A 3x Hattrick training process is already running:`n$details"
}

if ($CheckOnly) {
    Write-Host "[check] Python: $Python"
    Write-Host "[check] Hattrick runner: $hattrickScript"
    Write-Host "[check] Hattrick-f3 runner: $f3Script"
    Write-Host "[check] Hattrick-f3 state: $f3Run"
    Write-Host "[check] duration: $HoursPerMethod hour(s) per method"
    Write-Host "[check] skip Hattrick: $SkipHattrick"
    Write-Host "[check] no conflicting 3x training process found"
    return
}

function Test-RunComplete {
    param([string]$RunDirectory)
    $completePath = Join-Path $RunDirectory "complete.json"
    if (-not (Test-Path -LiteralPath $completePath -PathType Leaf)) {
        return $false
    }
    $complete = Get-Content -LiteralPath $completePath -Raw | ConvertFrom-Json
    return $complete.status -eq "COMPLETE" -and [int]$complete.epochs -ge $Epochs
}

function Get-EpochLineCount {
    param([string]$LogPath, [string]$Method)
    if (-not (Test-Path -LiteralPath $LogPath -PathType Leaf)) { return 0 }
    $escaped = [regex]::Escape($Method)
    return @(
        Select-String -LiteralPath $LogPath -Pattern "^\[3x $escaped\] epoch="
    ).Count
}

function Show-LogTail {
    param([string]$StdoutLog, [string]$StderrLog)
    if (Test-Path -LiteralPath $StdoutLog) {
        Get-Content -LiteralPath $StdoutLog -Tail 20
    }
    if ((Test-Path -LiteralPath $StderrLog) -and
        (Get-Item -LiteralPath $StderrLog).Length -gt 0) {
        Write-Host "[stderr tail]"
        Get-Content -LiteralPath $StderrLog -Tail 20
    }
}

function Invoke-TrainingSegment {
    param(
        [string]$Method,
        [string]$Script,
        [string]$RunDirectory,
        [string[]]$MethodArguments
    )

    if (Test-RunComplete -RunDirectory $RunDirectory) {
        Write-Host "[skip] $Method is already complete through epoch $Epochs."
        return
    }

    $arguments = @(
        $Script,
        "--stage", "train",
        "--seed", "$Seed",
        "--epochs", "$Epochs",
        "--top-k", "5",
        "--save-every", "5",
        "--batch-size", "8",
        "--learning-rate", "0.0005",
        "--device", $Device
    ) + $MethodArguments

    $resumeState = Join-Path $RunDirectory "resume_state.pt"
    if (Test-Path -LiteralPath $resumeState -PathType Leaf) {
        $arguments += "--resume"
        Write-Host "[resume] $Method from $resumeState"
    }
    else {
        Write-Host "[start] $Method has no resume state; starting its first epoch."
    }

    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $safeMethod = $Method -replace '[^A-Za-z0-9_-]', '_'
    $stdoutLog = Join-Path $logDir "${stamp}_${safeMethod}_seed_${Seed}.log"
    $stderrLog = Join-Path $logDir "${stamp}_${safeMethod}_seed_${Seed}.err.log"
    Write-Host "[run] $Method for about $HoursPerMethod hour(s)."
    Write-Host "[log] $stdoutLog"

    $process = Start-Process `
        -FilePath $Python `
        -ArgumentList $arguments `
        -WorkingDirectory $repo `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog `
        -WindowStyle Hidden `
        -PassThru

    $deadline = (Get-Date).AddHours($HoursPerMethod)
    $nextStatus = (Get-Date).AddMinutes(5)
    while (-not $process.HasExited -and (Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 10
        $process.Refresh()
        if ((Get-Date) -ge $nextStatus) {
            Write-Host "[running] $Method PID=$($process.Id); deadline=$($deadline.ToString('s'))"
            $nextStatus = (Get-Date).AddMinutes(5)
        }
    }

    $stoppedAtBoundary = $false
    if (-not $process.HasExited) {
        $linesAtDeadline = Get-EpochLineCount -LogPath $stdoutLog -Method $Method
        Write-Host "[limit] $Method reached its time limit; waiting for the current epoch to finish safely."
        while (-not $process.HasExited) {
            Start-Sleep -Seconds 5
            $process.Refresh()
            $lineCount = Get-EpochLineCount -LogPath $stdoutLog -Method $Method
            if ($lineCount -gt $linesAtDeadline) {
                Write-Host "[boundary] Checkpoint and Top-K update completed."
                if (-not $process.WaitForExit(5000)) {
                    Stop-Process -Id $process.Id -Force
                    $process.WaitForExit()
                    $stoppedAtBoundary = $true
                    Write-Host "[boundary] $Method stopped before starting another full epoch."
                }
                break
            }
        }
    }
    else {
        $process.WaitForExit()
    }

    Show-LogTail -StdoutLog $stdoutLog -StderrLog $stderrLog
    if ($stoppedAtBoundary) {
        if (-not (Test-Path -LiteralPath $resumeState -PathType Leaf)) {
            throw "$Method was stopped at an epoch boundary but has no resume_state.pt."
        }
        Write-Host "[paused] $Method is safely resumable."
        Start-Sleep -Seconds 2
        return
    }

    $process.Refresh()
    $exitCode = $process.ExitCode
    if ($exitCode -ne 0) {
        $exitCodeLabel = if ($null -eq $exitCode) { "unknown" } else { "$exitCode" }
        throw "$Method exited with code $exitCodeLabel. The next method will not be started. See $stderrLog"
    }
    if (-not (Test-RunComplete -RunDirectory $RunDirectory)) {
        throw "$Method exited successfully without a complete.json for epoch $Epochs."
    }
    Write-Host "[done] $Method reached epoch $Epochs."
}

Push-Location $repo
try {
    if ($SkipHattrick) {
        Write-Host "[skip] Hattrick skipped by request; starting with Hattrick-f3."
    }
    else {
        Invoke-TrainingSegment `
            -Method "Hattrick" `
            -Script $hattrickScript `
            -RunDirectory $hattrickRun `
            -MethodArguments @()
    }

    Invoke-TrainingSegment `
        -Method "Hattrick-f3" `
        -Script $f3Script `
        -RunDirectory $f3Run `
        -MethodArguments @("--anneal-epochs", "12", "--low-budget", "0.03")
}
finally {
    Pop-Location
}

Write-Host "[ok] Sequential timed training finished."

param(
    [string]$Config = 'configs/cm-dgseg/cm_dgseg_b0_cityscapes.py',
    [int]$Gpus = 1,
    [string]$WorkDir = '',
    [string]$Checkpoint = 'latest.pth'
)

if (-not $WorkDir) {
    $WorkDir = Join-Path 'work_dirs' ([IO.Path]::GetFileNameWithoutExtension($Config))
}

$logDir = Join-Path $WorkDir 'logs'
$smiLog = Join-Path $logDir 'nvidia_smi.csv'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

$job = $null
if (Get-Command 'nvidia-smi' -ErrorAction SilentlyContinue) {
    "timestamp,index,name,utilization.gpu [%],utilization.memory [%],memory.total [MiB],memory.used [MiB],memory.free [MiB]" | Set-Content $smiLog
    $job = Start-Job -ScriptBlock {
        param($Path)
        & nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free `
            --format=csv,noheader -l 60 >> $Path
    } -ArgumentList $smiLog
} else {
    Write-Warning 'nvidia-smi not available; skipping telemetry.'
}

try {
    $trainArgs = @('tools/train.py', $Config, '--work-dir', $WorkDir, '--auto-resume')
    if ($Gpus -gt 1) {
        $trainArgs += @('--launcher', 'pytorch', '--devices', $Gpus)
    }
    & python @trainArgs

    $ckptPath = Join-Path $WorkDir $Checkpoint
    & python 'tools/test.py' $Config $ckptPath '--eval' 'mIoU'
}
finally {
    if ($job) {
        Stop-Job $job -ErrorAction SilentlyContinue | Out-Null
        Receive-Job $job | Out-Null
        Remove-Job $job | Out-Null
    }
}

# Explicitly authorized E3 device fault injection. Run elevated only after the
# background resource gate has ended and the isolated product test is ready.
[CmdletBinding()]
param([ValidateSet('PnpDevice', 'PnPUtil')][string]$DeviceCommand = 'PnpDevice')

$ErrorActionPreference = 'Stop'
$observerCameraId = 'USB\VID_5986&PID_2130&MI_00\6&E71527D&0&0000'
$observerProjectRoot = Split-Path -Parent $PSScriptRoot
$observerReportRoot = Join-Path $observerProjectRoot 'artifacts\phase-e'
$observerReportName = 'e3-camera-cycle-' + [datetime]::UtcNow.ToString('yyyyMMddTHHmmssfff') + '.json'
$observerReportPath = Join-Path $observerReportRoot $observerReportName
$observerResult = [ordered]@{
    instance_id = $observerCameraId
    started_utc = [datetime]::UtcNow.ToString('o')
    classification = 'system_device_disable_enable_not_physical_unplug'
    device_command = $DeviceCommand
    disable_attempted = $false
    disabled_verified = $false
    restored_verified = $false
    restart_required = $false
    error = $null
    restore_error = $null
}
$observerRestoreRequired = $false

try {
    $observerPrincipal = [Security.Principal.WindowsPrincipal]::new(
        [Security.Principal.WindowsIdentity]::GetCurrent())
    if (-not $observerPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw 'Administrator permission is required; use normal Windows UAC approval.'
    }
    $observerLongTests = @(Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -match '\brun_e5_background\.py\b' })
    if ($observerLongTests.Count -gt 0) {
        throw 'A real background test is still running; do not disturb its camera.'
    }
    $observerCamera = @(Get-PnpDevice -PresentOnly -InstanceId $observerCameraId)
    if ($observerCamera.Count -ne 1 -or $observerCamera[0].Class -ne 'Camera' -or
        $observerCamera[0].FriendlyName -ne 'Integrated Camera' -or
        $observerCamera[0].Status -ne 'OK') {
        throw 'The exact authorized camera must be present and enabled before testing.'
    }
    # DN_NEED_RESTART (Cfg.h) can be set even when Status=OK / Problem=0.
    $observerResult.initial_devnode_status = [uint32](Get-PnpDeviceProperty -InstanceId $observerCameraId -KeyName 'DEVPKEY_Device_DevNodeStatus' -ErrorAction Stop).Data
    if (($observerResult.initial_devnode_status -band 0x100) -ne 0) {
        $observerResult.restart_required = $true
        throw 'The exact camera already requires a system restart; no device operation will be attempted.'
    }
    New-Item -ItemType Directory -Path $observerReportRoot -Force | Out-Null
    # Mark before the operation: even a partially successful disable must restore.
    $observerRestoreRequired = $true
    $observerResult.disable_attempted = $true
    if ($DeviceCommand -eq 'PnPUtil') {
        $observerResult.disable_output = @(& 'C:\Windows\System32\pnputil.exe' /disable-device $observerCameraId 2>&1 | ForEach-Object { "$_" })
        $observerResult.disable_exit_code = $LASTEXITCODE
        if ($LASTEXITCODE -ne 0) {
            throw "PnPUtil disable returned $LASTEXITCODE; no force or reboot will be attempted."
        }
    } else {
        Disable-PnpDevice -InstanceId $observerCameraId -Confirm:$false -ErrorAction Stop
    }
    $observerDisabled = Get-PnpDevice -InstanceId $observerCameraId -ErrorAction Stop
    $observerResult.disabled_utc = [datetime]::UtcNow.ToString('o')
    $observerResult.disabled_problem = [int]$observerDisabled.Problem
    $observerResult.disabled_verified = ([int]$observerDisabled.Problem -eq 22)
    if (-not $observerResult.disabled_verified) {
        throw 'Windows did not verify the disabled device state (problem code 22).'
    }
    Start-Sleep -Seconds 12
} catch {
    $observerResult.error = $_.Exception.Message
    $observerResult.error_id = $_.FullyQualifiedErrorId
    $observerResult.error_category = $_.CategoryInfo.ToString()
    $observerResult.error_hresult = $_.Exception.HResult
} finally {
    if ($observerRestoreRequired) {
        $observerResult.restore_attempts = @()
        for ($observerAttempt = 1; $observerAttempt -le 3; $observerAttempt++) {
            $observerAttemptResult = [ordered]@{ attempt = $observerAttempt; enable_error = $null; query_error = $null }
            try {
                if ($DeviceCommand -eq 'PnPUtil') {
                    $observerAttemptResult.enable_output = @(& 'C:\Windows\System32\pnputil.exe' /enable-device $observerCameraId 2>&1 | ForEach-Object { "$_" })
                    $observerAttemptResult.enable_exit_code = $LASTEXITCODE
                    if ($LASTEXITCODE -ne 0) {
                        throw "PnPUtil enable returned $LASTEXITCODE."
                    }
                } else {
                    Enable-PnpDevice -InstanceId $observerCameraId -Confirm:$false -ErrorAction Stop
                }
            } catch {
                $observerAttemptResult.enable_error = $_.Exception.Message
            }
            $observerRestoreDeadline = [datetime]::UtcNow.AddSeconds(10)
            do {
                try {
                    $observerRestored = Get-PnpDevice -PresentOnly -InstanceId $observerCameraId -ErrorAction Stop
                    $observerResult.restore_devnode_status = [uint32](Get-PnpDeviceProperty -InstanceId $observerCameraId -KeyName 'DEVPKEY_Device_DevNodeStatus' -ErrorAction Stop).Data
                    if (($observerResult.restore_devnode_status -band 0x100) -ne 0) {
                        $observerResult.restart_required = $true
                        $observerResult.restore_error = 'Windows requires a system restart for this camera; PnP status alone does not verify recovery.'
                        break
                    }
                    if ($observerRestored.Status -eq 'OK' -and [int]$observerRestored.Problem -eq 0) {
                        $observerResult.restored_verified = $true
                        $observerResult.restored_utc = [datetime]::UtcNow.ToString('o')
                        break
                    }
                } catch {
                    $observerAttemptResult.query_error = $_.Exception.Message
                }
                Start-Sleep -Milliseconds 250
            } while ([datetime]::UtcNow -lt $observerRestoreDeadline)
            $observerResult.restore_attempts += $observerAttemptResult
            if ($observerResult.restored_verified -or $observerResult.restart_required) {
                break
            }
        }
        $observerResult.restore_attempts_finished_utc = [datetime]::UtcNow.ToString('o')
        if (-not $observerResult.restored_verified) {
            if (-not $observerResult.restore_error) {
                $observerResult.restore_error = 'Camera restoration failed after three bounded attempts.'
            }
            if ($observerResult.restart_required) {
                [Console]::Error.WriteLine('URGENT: Windows requires a system restart for the camera. No system restart was initiated; save work and obtain user approval.')
            } else {
                [Console]::Error.WriteLine("URGENT: In administrator PowerShell run: Enable-PnpDevice -InstanceId '$observerCameraId' -Confirm:`$false")
            }
        }
    }
    $observerResult.finished_utc = [datetime]::UtcNow.ToString('o')
    try {
        New-Item -ItemType Directory -Path $observerReportRoot -Force | Out-Null
        $observerResult | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $observerReportPath -Encoding utf8
        Write-Output $observerReportPath
    } catch {
        $observerResult.report_error = $_.Exception.Message
        [Console]::Error.WriteLine(($observerResult | ConvertTo-Json -Depth 4))
    }
}

if ($observerResult.error -or $observerResult.restore_error -or $observerResult.report_error -or
    -not $observerResult.disabled_verified -or -not $observerResult.restored_verified) {
    exit 1
}
exit 0

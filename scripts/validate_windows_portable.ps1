param(
    [string]$Candidate = (Join-Path $PSScriptRoot 'Observer'),
    [string]$OutputRoot = (Join-Path $PSScriptRoot 'validation-results'),
    [switch]$IndependentMachine
)
# Windows PowerShell 5.1 / PowerShell 7. No Python, repository or admin required.
$ErrorActionPreference = 'Stop'
if ($env:OS -ne 'Windows_NT' -or -not [Environment]::Is64BitOperatingSystem -or -not [Environment]::Is64BitProcess) {
    throw 'Run with 64-bit Windows PowerShell on Windows x64.'
}
$candidatePath = (Resolve-Path -LiteralPath $Candidate).Path
$outputPath = [IO.Path]::GetFullPath($OutputRoot)
if ($outputPath.Equals($candidatePath, [StringComparison]::OrdinalIgnoreCase) -or
    $outputPath.StartsWith($candidatePath + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'OutputRoot must be outside the candidate directory.'
}
$runId = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ') + '-' + [Guid]::NewGuid().ToString('N').Substring(0,8)
$runPath = Join-Path $outputPath $runId
if (Test-Path -LiteralPath $runPath) { throw 'Refusing to overwrite existing evidence.' }
New-Item -ItemType Directory -Path $runPath -Force | Out-Null
$utf8 = New-Object Text.UTF8Encoding($false)
function Write-Json($Path, $Value) {
    [IO.File]::WriteAllText($Path, ($Value | ConvertTo-Json -Depth 15) + [Environment]::NewLine, $utf8)
}
function Assert-Inventory {
    $inventory = Get-Content -LiteralPath (Join-Path $candidatePath 'files.sha256.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $expected = @{}
    foreach ($row in $inventory) {
        $relative = [string]$row.path
        $file = [IO.Path]::GetFullPath((Join-Path $candidatePath $relative))
        if (-not $file.StartsWith($candidatePath + '\', [StringComparison]::OrdinalIgnoreCase)) {
            throw 'Manifest path escaped candidate directory.'
        }
        if ($expected.ContainsKey($relative)) { throw "Duplicate manifest path: $relative" }
        $expected[$relative] = $true
        if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing file: $relative" }
        if ((Get-Item -LiteralPath $file).Length -ne $row.bytes -or
            (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant() -ne $row.sha256) {
            throw "File integrity mismatch: $relative"
        }
    }
    $actual = @(Get-ChildItem -LiteralPath $candidatePath -Recurse -File)
    foreach ($file in $actual) {
        $relative = $file.FullName.Substring($candidatePath.Length + 1).Replace('\','/')
        if ($relative -ne 'files.sha256.json' -and -not $expected.ContainsKey($relative)) {
            throw "Unrecorded file: $relative"
        }
    }
    if ($actual.Count -ne $expected.Count + 1) { throw 'Candidate inventory count mismatch.' }
    return $expected.Count
}
function Quote-Argument([string]$Value) {
    # CommandLineToArgvW quoting, including backslashes before a quote or end.
    return '"' + [regex]::Replace([regex]::Replace($Value, '(\\*)"', '$1$1\"'), '(\\+)$', '$1$1') + '"'
}
$localAppData = Join-Path $runPath 'isolated-appdata'
$dataPath = Join-Path $localAppData 'Observer\product'
$cwdPath = Join-Path $runPath 'arbitrary working directory'
$tempPath = Join-Path $runPath 'temp'
New-Item -ItemType Directory -Path $cwdPath,$tempPath | Out-Null
$summary = [ordered]@{
    started_utc = [DateTime]::UtcNow.ToString('o')
    status = 'running'
    scope = 'camera-free automated internal candidate checks'
    independent_machine_declared = [bool]$IndependentMachine
    independent_machine_verified_automatically = $false
    windows_version = [Environment]::OSVersion.VersionString
    powershell_version = $PSVersionTable.PSVersion.ToString()
    process_64bit = [Environment]::Is64BitProcess
    elevated_token = (New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    sanitized_path = (Join-Path $env:SystemRoot 'System32')
    camera_opened = $false
    runs = @()
    manual_checks = 'pending; see manual-results.json'
}
function Invoke-Candidate([string]$ExeName, [string]$Label, [switch]$LoadNative) {
    $reportPath = Join-Path $runPath ($Label + '.json')
    $arguments = @('--self-test')
    if ($LoadNative) { $arguments += '--load-native' }
    $arguments += @('--data-dir', $dataPath, '--report', $reportPath)
    $info = New-Object Diagnostics.ProcessStartInfo
    $info.FileName = Join-Path $candidatePath $ExeName
    $info.Arguments = ($arguments | ForEach-Object { Quote-Argument $_ }) -join ' '
    $info.WorkingDirectory = $cwdPath
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    foreach ($key in @('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','CONDA_PREFIX','QT_PLUGIN_PATH','QT_QPA_PLATFORM_PLUGIN_PATH','QML2_IMPORT_PATH')) {
        $info.EnvironmentVariables.Remove($key)
    }
    $info.EnvironmentVariables['PATH'] = $summary.sanitized_path
    $info.EnvironmentVariables['LOCALAPPDATA'] = $localAppData
    $info.EnvironmentVariables['TEMP'] = $tempPath
    $info.EnvironmentVariables['TMP'] = $tempPath
    $info.EnvironmentVariables['QT_QPA_PLATFORM'] = 'windows'
    $process = New-Object Diagnostics.Process
    $process.StartInfo = $info
    $started = [DateTime]::UtcNow.ToString('o')
    try {
        if (-not $process.Start()) { throw "Could not launch $ExeName" }
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit(120000)) {
            $process.Kill() # Only this runner's owned test child.
            $process.WaitForExit()
            throw "$Label exceeded 120 seconds; owned test child stopped."
        }
        $process.WaitForExit()
        [IO.File]::WriteAllText((Join-Path $runPath ($Label + '.stdout.log')), $stdout.Result, $utf8)
        [IO.File]::WriteAllText((Join-Path $runPath ($Label + '.stderr.log')), $stderr.Result, $utf8)
        $summary.runs += [ordered]@{ executable=$ExeName; arguments=$arguments; started_utc=$started; finished_utc=[DateTime]::UtcNow.ToString('o'); exit_code=$process.ExitCode }
        if ($process.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $reportPath)) { throw "$Label failed; inspect reports and logs." }
        $result = Get-Content -LiteralPath $reportPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($result.status -ne 'passed' -or $result.desktop.qt_platform -ne 'windows' -or
            -not $result.desktop.no_argument_entry_wake -or $result.camera_opened) { throw "$Label did not meet native desktop criteria." }
        return $result
    } finally { $process.Dispose() }
}
try {
    $summary.inventory_files = Assert-Inventory
    $summary.manifest_sha256 = (Get-FileHash -LiteralPath (Join-Path $candidatePath 'files.sha256.json') -Algorithm SHA256).Hash.ToLowerInvariant()
    $first = Invoke-Candidate 'Observer-diagnostic.exe' 'diagnostic-native' -LoadNative
    $second = Invoke-Candidate 'Observer.exe' 'windowed-native'
    foreach ($id in $first.desktop.session_ids) {
        if ($id -notin $second.desktop.session_ids) { throw 'History lost on windowed reopening.' }
    }
    foreach ($hash in $first.desktop.evidence_hashes.PSObject.Properties) {
        if ($second.desktop.evidence_hashes.($hash.Name) -ne $hash.Value) { throw 'Synthetic evidence changed on reopening.' }
    }
    if ((Assert-Inventory) -ne $summary.inventory_files) { throw 'Inventory changed after execution.' }
    $summary.build_id = $first.resources.identity.build_id
    $summary.source_tree_sha256 = $first.resources.identity.source_tree_sha256
    $summary.native_versions = $first.resources.native
    $summary.status = 'passed'
    Write-Json (Join-Path $runPath 'manual-results.json') ([ordered]@{
        status='pending'; build_id=$summary.build_id
        independent_windows_without_repository_or_python='pending'
        first_gui_launch_via_shortcut_in_demo='pending'
        close_hides_to_tray='pending'; tray_restore_and_exit='pending'
        notification_visible_and_click_restores='pending'; sound_audible='pending'
        camera_and_long_run='deferred per E handoff'
        notes='Record what was actually observed. Automated notification calls do not confirm delivery or audibility.'
    })
    Write-Host "Automated checks passed. Manual checks remain pending. Evidence: $runPath"
} catch {
    $summary.status = 'failed'
    $summary.error = $_.Exception.Message
    Write-Host "Checks failed. Evidence: $runPath"
    throw
} finally {
    $summary.finished_utc = [DateTime]::UtcNow.ToString('o')
    Write-Json (Join-Path $runPath 'summary.json') $summary
}

param(
    [Parameter(Mandatory=$true)][string]$Candidate,
    [Parameter(Mandatory=$true)][string]$RunId
)
$ErrorActionPreference = 'Stop'
if ($RunId -notmatch '^[A-Za-z0-9_-]+$') { throw 'RunId must be a simple directory name' }
$root = Split-Path -Parent $PSScriptRoot
$source = (Resolve-Path -LiteralPath $Candidate).Path
$testBase = Join-Path $root 'build\phase-f-tests'
$target = [IO.Path]::GetFullPath((Join-Path $testBase "$RunId\Readonly Observer"))
if (-not $target.StartsWith(([IO.Path]::GetFullPath($testBase) + '\'), [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Read-only test target escaped the isolated workspace'
}
if (Test-Path -LiteralPath $target) { throw 'Use a fresh RunId; do not overwrite prior evidence' }
$evidence = Join-Path $root "artifacts\phase-f\$RunId"
New-Item -ItemType Directory -Path $evidence | Out-Null
New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
Copy-Item -LiteralPath $source -Destination $target -Recurse
$acl = Get-Acl -LiteralPath $target
$originalSddl = $acl.Sddl
$originalSddl | Set-Content -LiteralPath (Join-Path $evidence 'original-acl.txt')
$sid = [Security.Principal.WindowsIdentity]::GetCurrent().User
$rule = [Security.AccessControl.FileSystemAccessRule]::new(
    $sid, [Security.AccessControl.FileSystemRights]::Write,
    [Security.AccessControl.InheritanceFlags]'ContainerInherit,ObjectInherit',
    [Security.AccessControl.PropagationFlags]::None, [Security.AccessControl.AccessControlType]::Deny)
$exitCode = 1
$denied = $false
$restored = $false
try {
    $acl.AddAccessRule($rule)
    Set-Acl -LiteralPath $target -AclObject $acl
    try {
        [IO.File]::WriteAllText((Join-Path $target 'must-not-write.txt'), 'read-only probe')
        throw 'Read-only setup failed: target still accepts writes'
    } catch [UnauthorizedAccessException] {
        $denied = $true
    }
    $env:QT_QPA_PLATFORM = 'offscreen'
    $env:PATH = Join-Path $env:SystemRoot 'System32'
    Remove-Item Env:PYTHONPATH,Env:PYTHONHOME,Env:QT_PLUGIN_PATH,Env:QT_QPA_PLATFORM_PLUGIN_PATH -ErrorAction SilentlyContinue
    $env:LOCALAPPDATA = Join-Path (Split-Path -Parent $target) 'isolated-appdata'
    $dataDir = Join-Path $env:LOCALAPPDATA 'Observer\product'
    & (Join-Path $target 'Observer-diagnostic.exe') --self-test --load-native --data-dir $dataDir --report (Join-Path $evidence 'readonly-desktop.json') *> (Join-Path $evidence 'readonly-desktop.log')
    $exitCode = $LASTEXITCODE
} finally {
    $restoreAcl = Get-Acl -LiteralPath $target
    $restoreAcl.SetSecurityDescriptorSddlForm($originalSddl)
    Set-Acl -LiteralPath $target -AclObject $restoreAcl
    $restored = (Get-Acl -LiteralPath $target).Sddl -eq $originalSddl
    @{ target=$target; write_probe_denied=$denied; exit_code=$exitCode; original_acl_restored=$restored } |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $evidence 'summary.json') -Encoding UTF8
}
if (-not $restored) { throw 'ACL restoration must be resolved before leaving this test' }
if ($exitCode -ne 0) { throw "Candidate failed read-only test: $exitCode" }

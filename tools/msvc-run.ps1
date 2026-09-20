param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string] $Command,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $CommandArgs
)

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"

$vs = & $vswhere `
    -latest `
    -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath

if (-not $vs) {
    throw "Visual Studio C++ tools not found"
}

$devshell = Join-Path $vs "Common7\Tools\Launch-VsDevShell.ps1"

if (-not (Test-Path $devshell)) {
    throw "Launch-VsDevShell.ps1 not found: $devshell"
}

& $devshell `
    -Arch amd64 `
    -HostArch amd64 `
    -SkipAutomaticLocation 6>$null

if ($LASTEXITCODE) {
    exit $LASTEXITCODE
}

& $Command @CommandArgs
exit $LASTEXITCODE
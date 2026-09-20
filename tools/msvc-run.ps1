param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string] $MsvcRunExecutable,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $MsvcRunArguments
)

# Some hosts inject both spellings into the otherwise case-insensitive Windows
# environment. MSBuild materializes the block into a case-insensitive dictionary
# and fails before starting cl.exe when both keys survive. Keep the canonical
# mixed-case entry that Visual Studio's developer shell expects.
if (Test-Path Env:PATH) {
    $msvcRunPath = $env:PATH
    Remove-Item Env:PATH
    $env:Path = $msvcRunPath
}

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

if (-not $env:VSCMD_VER) {
    & $devshell `
        -Arch amd64 `
        -HostArch amd64 `
        -SkipAutomaticLocation 6>$null

    if ($LASTEXITCODE) {
        exit $LASTEXITCODE
    }
}

& $MsvcRunExecutable @MsvcRunArguments
exit $LASTEXITCODE

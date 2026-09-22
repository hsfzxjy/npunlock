# Getting MoviTools

[Documentation index](README.md)

`npunlock` currently uses Intel/Movidius MoviTools to compile user C into
ACT-SHAVE code. These proprietary tools are not included in this repository
and are not downloaded automatically.

The known-good files are available inside an older Lenovo Intel NPU driver
package. You only extract that package to obtain its compiler payload. **Do not
install the legacy driver and do not replace or downgrade your installed Intel
NPU driver.** The installed driver remains responsible for Level Zero,
graph compilation, and NPU execution.

## Package details

| Field | Value |
| --- | --- |
| Package | Intel Vision Processing Unit Driver |
| Version | `31.0.100.1688` |
| Filename | `r2gvp04w_v2.exe` |
| Size shown by Lenovo | 60.12 MB |
| Release date shown by Lenovo | 2024-04-16 |
| SHA-256 | `cf767eea31debaa199b084cce06eb82443aa251d5963ef97030f5219b9fe28db` |

Download the package from Lenovo:

- [Official direct download](https://download.lenovo.com/pccbbs/mobiles/r2gvp04w_v2.exe)
- [Official Lenovo support page](https://support.lenovo.com/lc/en/downloads/ds568542-intel-vision-processing-unit-driver-for-windows-11-version-22h2-or-later-10-version-21h2-or-later-thinkpad-l13-2-in-1-gen-5-type-21lm-21ln-l13-gen-5-type-21lb-21lc)

The direct URL was reachable when this guide was prepared. If it later stops
working, use the Lenovo support page rather than an unofficial mirror.

## Verify the download

Open PowerShell in the download directory and run:

```powershell
Get-FileHash .\r2gvp04w_v2.exe -Algorithm SHA256
```

The hash must be:

```text
CF767EEA31DEBAA199B084CCE06EB82443AA251D5963EF97030F5219B9FE28DB
```

Do not continue if the value differs.

## Extract without installing the driver

The file is a Lenovo self-extracting executable. The following command was
tested successfully; 7-Zip does not directly expose the useful payload:

```powershell
.\r2gvp04w_v2.exe /VERYSILENT /DIR="C:\path\to\npu-driver-dir" /EXTRACT=YES
```

This extracts the package without asking you to install the legacy driver.

Locate the `MVC_DEPEND` directory:

```powershell
Get-ChildItem C:\path\to\npu-driver-dir -Directory -Recurse -Filter MVC_DEPEND
```

It should contain at least:

```text
MVC_DEPEND\
  bin\
    moviAsm64.dll
    moviCompile64.dll
    moviLLD64.dll
  lib\
    mlibc_lite.a
    mlibc_lite_ext.a
    mlibcrt.a
    mlibcrt_mini.a
    mlibcxx.a
    mlibm.a
    mlibVecUtils.a
```

The package's driver manifest lists these files in this layout.

## Configure npunlock

Point `npunlock` at the `MVC_DEPEND` root, not at individual DLL files. For the
current PowerShell session:

```powershell
$env:NPUNLOCK_MOVITOOLS_DIR = 'C:\path\to\MVC_DEPEND'
```

Python can configure the same path explicitly:

```python
import npunlock as npu

npu.configure(movi_dll_dir=r"C:\path\to\MVC_DEPEND")
```

The CLI accepts it directly as well:

```powershell
npurun build --movi-dll-dir C:\path\to\MVC_DEPEND ...
```

The `bin` directory itself is not accepted. The root is required because the
toolchain uses both the DLLs and the `lib\mlibm.a` math archive needed by
kernels such as GELU.

## Keep the two driver roles separate

```text
installed Intel NPU driver
  -> graph compilation, Level Zero, and hardware execution

extracted Lenovo 31.0.100.1688 package
  -> source of MoviTools compiler files only
  -> not installed
```

Extracting MoviTools does not change the driver used at runtime.

## Licensing and provenance

MoviTools is obtained by the user from Lenovo's original Intel NPU driver
package. `npunlock` does not redistribute the DLLs or Intel/Movidius archive
files. The package contains third-party notices for some components; those
notices must not be interpreted as a license for the MoviTools binaries as a
whole.

[Back to documentation index](README.md)

# Fixture provenance

The fixtures in `npu3720/` are the smallest retained inputs and golden outputs
needed to exercise the confirmed MVP path. They are redistributed under this
project's Apache License 2.0 unless a more specific origin is listed below.
No Intel/OEM compiler DLL, driver binary, or OpenVINO runtime component is
included.

| File | SHA-256 | Origin and purpose |
| --- | --- | --- |
| `abs.xml` | `02be7ab43b45c3ab80e11adf796f3c31e17c9d6c95e8748e74cce888f80c9a31` | Project research artifact: minimal static FP16 OpenVINO-format Abs IR used as `ir2blob` input. |
| `abs.bin` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | Empty weights associated with `abs.xml`. |
| `add1-fp16.c` | `72de0a6b2996a436f7a60cee60e0fc3f5208d75543a41cc697b864fd0850a531` | Project add-one ACT-SHAVE source used by the opt-in `shavecc` test. |
| `shave_kernel.ld` | `4b7faf5233e425c6d75a63b18d7f8ea7e06b3b023cbf3b39731de64294f975da` | Copied from Intel's Apache-2.0 `npu_compiler` repository at commit `0b38f7d42113ff329ac2bdd33583d123de4ccf2f`; used as linker input. That checkout has no `NOTICE` file. |
| `add1-fp16.elf` | `3f9d52273870c2e911000da3c3bd474c0514bd297d29539a14e935b2b01de6d5` | Golden linked output produced from `add1-fp16.c` and `shave_kernel.ld` with caller-owned MoviTools; used by offline ELF/patch tests. |
| `abs-add-1x16-tile1.blob` | `d4abc3c09cabdb14fd5090fa5d2e4d33f4aa755aad2bcad4f078acc891d26e07` | Project research artifact: unmodified NPU3720 ACT carrier graph used by the offline patch test. |
| `add1-shared-1x16.blob` | `2010915f2d21e07d2afee613ffe8f38a9ce6e39523ab98e0762c1bd40e04d95f` | Golden graph produced by applying the confirmed two-range patch to the carrier; also used by the opt-in execution test. |

The native graph blobs are narrow evidence fixtures for the observed NPU3720
ABI. Their presence does not imply support for arbitrary graph compiler or
driver versions.

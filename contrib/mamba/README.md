# Mamba-1 inference on ACT-SHAVE

This contribution runs the input-dependent selective scan from the official
`state-spaces/mamba-130m-hf` checkpoint on Intel Meteor Lake NPU ACT-SHAVE
cores through npunlock. It uses the original Safetensors weights without
retraining or replacing the recurrence with a fixed-parameter approximation.

For every channel and state dimension, the custom kernel evaluates the actual
Mamba-1 update:

```text
dt = softplus(delta + delta_bias)
state = exp(dt * A) * state + dt * B * u
y = (sum(C * state) + D * u) * silu(z)
```

Sequence output and final FP32 state are both returned. The next prefill chunk
or decoded token receives that state explicitly, so the CPU never calculates
the recurrence.

## Try it

From the repository root, create an environment and install the dependencies:

```powershell
python -m venv build\mamba-venv
$python = '.\build\mamba-venv\Scripts\python.exe'
& $python -m pip install -r contrib\mamba\requirements.txt
& $python -m pip install .
$env:PYTHONPATH = "$(Resolve-Path python);$(Resolve-Path contrib\mamba)"
```

Download the pinned checkpoint and run three greedy tokens through the complete
ACT path:

```powershell
& $python contrib\mamba\scripts\prepare_model.py `
  --output-dir build\mamba-model

& $python contrib\mamba\scripts\run_act_model.py `
  --model-dir build\mamba-model `
  --movi-dll-dir C:\path\to\MVC_DEPEND `
  --native-dir build\windows\Debug `
  --new-tokens 3
```

The checkpoint is pinned to revision
`40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37`; the recorded
`model.safetensors` SHA-256 is
`1a5ed29c492ef4d485df3b7c2c8109771696589855b2162ad1ba618b6067cbea`.

## What was verified

On Windows 11, NPU device `0x7d1d`, Level Zero graph extension 1.18, and graph
compiler 8.3:

- physical-NPU full-sequence, chunked, and repeated single-token scans matched
  an independent FP32 reference, including nonzero initial and final state;
- all 24 pretrained layers, normalization, projections, cached convolution and
  SiLU, residuals, selective scans, and the vocabulary projection executed on
  ACT-SHAVE;
- the prompt `The future of artificial intelligence is` produced the same
  next-token top five and greedy continuation `not yet clear` as Transformers;
- maximum/mean prefill-logit error was `1.83e-4 / 4.34e-5`, and restoring the
  recurrent cache reproduced continuation logits bit-for-bit; and
- median six-token prefill was 0.66 seconds and median three-token generation
  was 1.26 seconds (`2.38 token/s`). The matching official PyTorch CPU medians
  were 123.5 ms and 236.4 ms (`12.69 token/s`), so this is a programmability
  demonstration rather than a CPU speedup.

The host performs tokenization, embedding lookup, tensor packing and slicing,
submission orchestration, and greedy argmax. It does not execute a dense,
convolutional, or recurrent neural layer. No DPU kernels are used by the
complete path.

## State and partitioning

Each ACT invocation owns complete recurrent histories for an independent
channel shard; no cross-invocation communication or synchronization is
assumed. The complete path uses two validated groups of four 65,536-element
FP32 invocations and one shared SHAVE image. Up to 16 positions are processed
per submission. Longer prompts are split into ordered chunks connected only by
the explicit returned SSM and convolution state.

The 24-layer cache contains 2,359,296 bytes of FP32 SSM state and 589,824 bytes
of host-orchestrated FP32 convolution state.

## Public smoke test

The public suite checks recurrence chunking, incremental-state continuity, and
the transport layout without an NPU, MoviTools, OpenVINO, or model download:

```powershell
$env:PYTHONPATH = "$(Resolve-Path python);$(Resolve-Path contrib\mamba)"
& $python -m unittest discover -s contrib\mamba\tests -p 'test_mamba.py' -v
```

## Limits

- Windows x64, Meteor Lake/NPU3720, graph compiler 8.3, batch size one.
- FP32 transport and accumulation; state size at most 16.
- At most 16 positions per NPU submission; longer sequences use explicit host
  orchestration between chunks.
- The observed ACT group sizes and two-input layout are carrier-specific, not a
  general NPU ABI.
- Separate submissions require host-visible state transfer. The state update
  itself remains on the NPU.
- Stock OpenVINO 2026.3.1 compiled static Mamba state, block, and full-model
  cases explicitly for NPU on the recorded machine, but returned non-finite
  values. No `AUTO`, `HETERO`, or CPU fallback result is presented as NPU
  execution.

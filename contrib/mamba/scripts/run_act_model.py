from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from transformers import AutoTokenizer

from mamba_npu import ActMambaRunner, NpuActLinear
from mamba_npu.model import compare_tensors, load_official_model


def add_timings(target: dict[str, float], source: dict[str, float]) -> None:
    for name, value in source.items():
        target[name] = target.get(name, 0.0) + value


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate with all Mamba neural arithmetic on one ACT graph")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--movi-dll-dir", required=True)
    parser.add_argument("--native-dir")
    parser.add_argument("--prompt", default="The future of artificial intelligence is")
    parser.add_argument("--new-tokens", type=int, default=3)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.new_tokens < 1:
        parser.error("--new-tokens must be positive")
    torch.manual_seed(20261025)
    model = load_official_model(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        official_logits = model(input_ids).logits[:, -1:, :]
        official_ids = model.generate(input_ids, max_new_tokens=args.new_tokens, do_sample=False)
    with NpuActLinear(movi_dll_dir=args.movi_dll_dir, native_dir=args.native_dir) as runtime:
        runner = ActMambaRunner(model, runtime)
        started = perf_counter()
        logits, cache, timings = runner.prefill(input_ids.numpy())
        prefill_wall_ms = (perf_counter() - started) * 1000.0
        prefill_logits = logits.copy()
        saved_cache = cache.clone()
        generated = input_ids[0].tolist()
        first_continuation = None
        for index in range(args.new_tokens):
            token_id = int(np.argmax(logits[0]))
            generated.append(token_id)
            if index + 1 < args.new_tokens:
                logits, cache, step_timings = runner.step(token_id, cache)
                add_timings(timings, step_timings)
                if index == 0:
                    first_continuation = logits.copy()
        first_token = generated[len(input_ids[0])]
        restored_logits, restored_cache, _ = runner.step(first_token, saved_cache)
        if first_continuation is None:
            first_continuation = restored_logits
        generated_tensor = torch.tensor([generated], dtype=torch.int64)
        state_bytes = sum(value.nbytes for value in cache.convolution_states) + sum(
            value.nbytes for value in cache.ssm_states
        )
        report = {
            "prompt": args.prompt,
            "input_token_ids": input_ids[0].tolist(),
            "official_generated_ids": official_ids[0].tolist(),
            "npu_generated_ids": generated,
            "generated_text": tokenizer.decode(generated_tensor[0], skip_special_tokens=True),
            "greedy_tokens_match": generated == official_ids[0].tolist(),
            "prefill_last_logits_error": asdict(
                compare_tensors(torch.from_numpy(prefill_logits).unsqueeze(1), official_logits)
            ),
            "official_top5_next": torch.topk(official_logits[0, -1], 5).indices.tolist(),
            "npu_top5_next": torch.topk(torch.from_numpy(prefill_logits[0]), 5).indices.tolist(),
            "restored_cache_logits_error": asdict(compare_tensors(restored_logits, first_continuation)),
            "restored_cache_state_bytes": int(
                sum(value.nbytes for value in restored_cache.convolution_states)
                + sum(value.nbytes for value in restored_cache.ssm_states)
            ),
            "persistent_state_bytes": int(state_bytes),
            "prefill_wall_ms": prefill_wall_ms,
            "timings_ms": timings,
            "npu_program_build_ms": runtime.build_ms,
            "execution_split": {
                "npu": "RMSNorm, all projections, cached depthwise convolution/SiLU, residuals, all selective scans, and LM head",
                "cpu": "tokenization, embedding-table lookup, tensor packing/slicing, orchestration, and argmax",
            },
        }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

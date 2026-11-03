"""Run on the rebuilt Linux CPU backend; no model weights or NPU required.

python kt-kernel/tests/check_mxfp4_reuse_moe.py

Compare reuse OFF/ON in fresh processes: C++ caches environment switches.
Both use group_min_len=2 to isolate the dot change from the grouping policy.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def write_outputs(path):
    import torch
    from kt_kernel import kt_kernel_ext as ext

    torch.manual_seed(1024)
    experts, topk, hidden, intermediate = 4, 2, 256, 256
    weights = [torch.randint(0, 256, (experts, intermediate, hidden // 32, 17), dtype=torch.uint8) for _ in range(3)]
    for weight in weights:
        weight[..., 0] = 120
    mask = torch.zeros(experts, dtype=torch.bool)
    mask[0] = True
    pool_config = ext.WorkerPoolConfig()
    pool_config.subpool_count = 1
    pool_config.subpool_numa_map = [0]
    pool_config.subpool_thread_count = [2]
    pool = ext.CPUInfer(pool_config)
    config = ext.moe.MOEConfig(experts, topk, hidden, intermediate, mask.data_ptr())
    config.pool = pool.backend_
    config.m_block = 32
    config.max_len = config.group_max_len = 64
    config.group_min_len = 2
    config.gate_type = config.up_type = config.down_type = 39
    config.hidden_type = 30  # GGML_TYPE_BF16
    config.gate_proj, config.up_proj, config.down_proj = [w.data_ptr() for w in weights]
    moe = ext.moe.MOE(config)
    mapping = torch.arange(experts, dtype=torch.int64)
    pool.submit(moe.load_weights_task(mapping.data_ptr()))
    pool.sync()
    outputs = {}
    for rows in (1, 2, 3, 4, 5, 8, 17, 32, 48, 64):
        for pattern in ("shared", "disjoint", "mixed", "no_cpu", "zero"):
            x = (torch.randn(rows, hidden) * 0.01).bfloat16()
            ids = torch.tensor([1, 2], dtype=torch.int64).repeat(rows, 1)
            if pattern == "disjoint":
                ids = torch.arange(rows * topk, dtype=torch.int64).reshape(rows, topk) % 3 + 1
            elif pattern == "mixed":
                ids[::2, 0] = 0  # NPU-owned route
                ids[1::2, 0] = -1  # vLLM masks migrated/resident routes this way
            elif pattern == "no_cpu":
                ids[:, 0], ids[:, 1] = 0, -1
            elif pattern == "zero":
                x.zero_()
            routing = torch.rand(rows, topk, dtype=torch.float32)
            routing /= routing.sum(dim=1, keepdim=True)
            count = torch.tensor([rows], dtype=torch.int32)
            output = torch.full_like(x, float("nan"))
            pool.submit(moe.forward_task(count.data_ptr(), topk, ids.data_ptr(), routing.data_ptr(),
                                         x.data_ptr(), output.data_ptr(), False))
            pool.sync()
            if not torch.isfinite(output.float()).all():
                raise AssertionError(f"non-finite output: {rows}/{pattern}")
            if pattern in ("no_cpu", "zero") and torch.count_nonzero(output):
                raise AssertionError(f"expected zero output: {rows}/{pattern}")
            outputs[f"{rows}/{pattern}"] = output.clone()
    torch.save(outputs, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.output is not None:
        write_outputs(args.output)
        return
    import torch
    with tempfile.TemporaryDirectory(prefix="kt-mxfp4-reuse-") as directory:
        for mode in ("q8", "match"):
            paths = []
            for reuse in ("0", "1"):
                path = Path(directory) / f"{mode}-{reuse}.pt"
                env = dict(os.environ, KT_MOE_MXFP4_TOKEN_REUSE=reuse)
                env.pop("KT_MOE_MATCH_NPU_MXFP8", None)
                if mode == "match":
                    env["KT_MOE_MATCH_NPU_MXFP8"] = "1"
                subprocess.run([sys.executable, str(Path(__file__).resolve()), "--output", str(path)],
                               env=env, check=True)
                paths.append(path)
            off, on = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
            for key in off:
                if not torch.equal(off[key], on[key]):
                    raise AssertionError(f"{mode} OFF/ON BF16 mismatch: {key}")
            print(f"PASS: {mode}, {len(off)} routed MoE cases, exact BF16 OFF/ON agreement")


if __name__ == "__main__":
    main()

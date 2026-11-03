"""Build and run dot/matrix regressions without torch, CANN or model weights."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    parser.add_argument("--arch-flags", default="-march=native")
    args = parser.parse_args()
    kernel = Path(__file__).resolve().parents[1]
    repo = kernel.parent
    common = ["-O3", *shlex.split(args.arch_flags), "-ffunction-sections", "-fdata-sections"]
    with tempfile.TemporaryDirectory(prefix="kt-mxfp4-dot-") as directory:
        obj = Path(directory) / "quants.o"
        exe = Path(directory) / ("reuse.exe" if os.name == "nt" else "reuse")
        subprocess.run([*shlex.split(args.cc), *common, "-c",
                        str(repo / "third_party/llama.cpp/ggml-quants.c"), "-o", str(obj)], check=True)
        subprocess.run([*shlex.split(args.cxx), *common, "-ffast-math", "-std=c++17",
                        "-I", str(kernel), "-I", str(repo / "third_party"),
                        str(kernel / "tests/mxfp4_token_reuse_test.cpp"), str(obj), "-o", str(exe)], check=True)
        for value in ("0", "1"):
            env = dict(os.environ, KT_MOE_MXFP4_TOKEN_REUSE=value, KT_MOE_MATCH_NPU_MXFP8="1")
            subprocess.run([str(exe)], env=env, check=True)


if __name__ == "__main__":
    main()

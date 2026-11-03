# Ascend CPU MoE 后端

配套说明见 [CPU MoE 使用指南](https://github.com/yxk-fdu/vllm-ascend/blob/share/cpu-moe-9334-20261020/docs/cpu_moe/README.zh.md)。

本分支包含 Ascend 适配、原生 MXFP4、GGUF、异步提交/完成及 CPU 计算日志源码。运行包 Python 文件来自 9334 的 kt_async_runtime；C++ 使用当前源码树，包含空 CPU 路由跳过计算的实现，不引入实验目录中的诊断消融版本。

先准备匹配的 torch/torch_npu/CANN 环境，以及 hwloc-devel（Debian 为 libhwloc-dev）、pkg-config、CMake、GCC/G++、pybind11、numpy、safetensors、gguf。

```bash
git submodule update --init third_party/llama.cpp third_party/pybind11
git -C third_party/llama.cpp apply ../../patches/llama-cpp-mxfp4.patch
git -C third_party/llama.cpp apply ../../patches/llama-cpp-mxfp4-token-reuse.patch
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash kt-kernel/build_ascend_cpu_moe.sh /your/work/kt-build
export PYTHONPATH=/your/work/kt-build/package:${PYTHONPATH:-}
```

补丁对应 llama.cpp `a94e6ff8774b7c9f950d9545baf0ce35e8d1ed2f`。不要重复应用。转换器 `tools/convert_mxfp4_layer_to_gguf.py` 使用该依赖的 gguf-py。产物不提交到仓库；目标机器必须按自身 CPU、Python ABI 和 Ascend 环境重新编译。

9334 使用 Python 3.11.10、torch 2.10.0+cpu、torch_npu 2.10.0.post2。旧 torch<2.9 上限已放宽到 <2.11；安装时仍需显式准备匹配版本。独立构建脚本不运行 pip 依赖解析，不会自动替换现有 torch。纯 Ascend 构建不需要 CUDA Triton。

9334 GGUF 可复用 `/mnt/sdc/DeepSeek-V4-Flash-0731-GGUF`；原生模型为 `/mnt/data/DeepSeek-V4-Flash-0731`。其他服务器先问作者是否已有权重。

## GGUF MXFP4 多激活权重复用

同一专家的激活按最多 4 行分块，共享 MXFP4 权重加载和解码。
支持 MXFP4 × Q8_0（NEON / AVX2 / scalar）和
`KT_MOE_MATCH_NPU_MXFP8=1` 的数值匹配路径（AVX512-BF16 复用；无该指令时保持原标量内核）。
权重格式、MXFP8 模拟量化、BF16 舍入、路由和 vLLM 后端接口保持原有语义；单行和余下单行使用原点积。
原生 `native_mxfp4` 权重不使用本次多激活内核，默认分组阈值仍为 10。

已应用原 MXFP4 补丁的源码只需应用新增补丁，不要重复应用原补丁：

```bash
git -C third_party/llama.cpp apply --check ../../patches/llama-cpp-mxfp4-token-reuse.patch
git -C third_party/llama.cpp apply ../../patches/llama-cpp-mxfp4-token-reuse.patch
```

之后用上文独立构建脚本编译到一个新的目录，将启动脚本中的 `KT_BUILD` 和
`KT_MOE_POLL_LIBRARY` 同时指向新产物，再重启服务。仅更新 Python 文件不会更新 C++ 内核。
vLLM 的 `cpu_moe` 配置、提交/完成事件和 mapped-host combine 无需修改。

| 变量 | 默认 | 含义 |
| --- | --- | --- |
| `KT_MOE_MXFP4_TOKEN_REUSE` | `1` | `1` 开启，`0` 回退原单激活点积；进程内缓存，修改后重启 |
| `KT_GROUP_MIN_LEN` | 见说明 | GGUF 三组投影均为 MXFP4 且开启复用时为 `2`；其余为 `10`；可显式设正整数 |

保留原启动脚本的 `KT_MOE_MATCH_NPU_MXFP8=1`。它控制激活数值模式，与复用开关独立。
完整回退：设置 `KT_MOE_MXFP4_TOKEN_REUSE=0`、取消 `KT_GROUP_MIN_LEN` 后重启。
分离测量时固定 `KT_GROUP_MIN_LEN=2`，只切换复用开关。

无需模型的内核和配置回归（仓库根目录执行）：

```bash
python -m unittest discover -s kt-kernel/tests -p test_mxfp4_reuse_config.py
python kt-kernel/tests/run_mxfp4_reuse_kernel_test.py
# ARM 按服务器实际编译目标传入 --arch-flags，例如 -march=armv8.2-a+fp16+dotprod
# 已重编译的 Linux kt-kernel 环境：检查共享/不同/混合/空 CPU 路由及零输入
python kt-kernel/tests/check_mxfp4_reuse_moe.py
```

矩阵回归覆盖两条路径、复用 OFF/ON、1..64 行的指定档位、奇偶 K 块、非紧密步长及零激活，
并按同一 ISA 的原单激活实现逐位比较。整层回归在独立进程中比较两条路径的 OFF/ON BF16 输出。
服务器还需验证 DSpark 接受计数、输出和多次图重放，再测稳态 TPOT 与 CPU 层耗时；不预设性能收益。

本次不调整 `cpu_moe.max_num_tokens`。例如容量 32 的配置，超过 32 行仍按既有逻辑退出 CPU hybrid；
扩容到 64 时要另行核对 capture sizes、静态缓冲区和迁移算子容量。

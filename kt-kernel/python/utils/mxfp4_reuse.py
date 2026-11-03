"""Initialization-time options for GGUF MXFP4 token reuse (no torch dependency)."""

import os


def token_reuse_enabled() -> bool:
    value = os.environ.get("KT_MOE_MXFP4_TOKEN_REUSE", "1")
    if value not in ("0", "1"):
        raise ValueError("KT_MOE_MXFP4_TOKEN_REUSE must be 0 or 1")
    return value == "1"


def group_min_len(weight_types, *, native=False) -> int:
    reuse = token_reuse_enabled() and not native and all(int(t) == 39 for t in weight_types)
    try:
        threshold = int(os.environ.get("KT_GROUP_MIN_LEN", "2" if reuse else "10"))
    except ValueError as exc:
        raise ValueError("KT_GROUP_MIN_LEN must be an integer") from exc
    if threshold < 1:
        raise ValueError("KT_GROUP_MIN_LEN must be at least 1")
    return threshold

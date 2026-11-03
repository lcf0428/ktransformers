#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "llama.cpp/ggml-quants.h"
#include "mxfp8_match.h"

namespace kt_mxfp4_reuse {

inline bool enabled() {
  static const bool value = [] {
    const char* option = std::getenv("KT_MOE_MXFP4_TOKEN_REUSE");
    if (!option || std::strcmp(option, "1") == 0) return true;
    if (std::strcmp(option, "0") == 0) return false;
    throw std::invalid_argument("KT_MOE_MXFP4_TOKEN_REUSE must be 0 or 1");
  }();
  return value;
}

// k, lda and ldb are quantization-block counts; ldc is a float count.
inline void q8_matmul(long m, long n, long k, const block_mxfp4* a, long lda,
                      const block_q8_0* b, long ldb, float* c, long ldc) {
  if (!enabled() || n < 2) {
    // Keep the original token-major traversal for a meaningful OFF baseline.
    for (long j = 0; j < n; ++j) {
      for (long i = 0; i < m; ++i) {
#if defined(__GNUC__)
        if (i + 1 < m) {
          const char* next = reinterpret_cast<const char*>(a + (i + 1) * lda);
          __builtin_prefetch(next, 0, 1);
          __builtin_prefetch(next + 64, 0, 1);
        }
#endif
        float sum;
        ggml_vec_dot_mxfp4_q8_0(k * 32, &sum, 0, a + i * lda, 0, b + j * ldb, 0, 1);
        c[j * ldc + i] = std::isfinite(sum) ? sum : 0.f;
      }
    }
    return;
  }
  for (long i = 0; i < m; ++i) {
#if defined(__GNUC__)
    if (i + 1 < m) {
      const char* next = reinterpret_cast<const char*>(a + (i + 1) * lda);
      __builtin_prefetch(next, 0, 1);
      __builtin_prefetch(next + 64, 0, 1);
    }
#endif
    for (long j = 0; j < n;) {
      const int rows = static_cast<int>(std::min<long>(4, n - j));
      float sums[4] = {};
      if (rows == 1) {
        ggml_vec_dot_mxfp4_q8_0(k * 32, sums, 0, a + i * lda, 0, b + j * ldb, 0, 1);
      } else {
        ggml_vec_dot_mxfp4_q8_0_many(k * 32, sums, 0, a + i * lda, 0,
                                    b + j * ldb, ldb * sizeof(block_q8_0), rows);
      }
      for (int r = 0; r < rows; ++r) {
        // Preserve the existing Q8_0 branch's non-finite handling.
        c[(j + r) * ldc + i] = std::isfinite(sums[r]) ? sums[r] : 0.f;
      }
      j += rows;
    }
  }
}

// k/lda count MXFP4 blocks; ldb/ldc count float elements. Activations have
// already passed through the existing MXFP8 quantize/dequantize operation.
inline void match_matmul(long m, long n, long k, const uint8_t* a, long lda,
                         const float* b, long ldb, float* c, long ldc) {
#if defined(__AVX512BF16__) && defined(__AVX512F__)
  const bool vector_reuse = enabled();
#else
  const bool vector_reuse = false;
#endif
  if (!vector_reuse || n < 2) {
    for (long j = 0; j < n; ++j) {
      for (long i = 0; i < m; ++i) {
        c[j * ldc + i] = kt_match_npu::dot_gguf_mxfp4_mxfp8(a + i * lda * 17, b + j * ldb, k);
      }
    }
    return;
  }
  for (long i = 0; i < m; ++i) {
    const uint8_t* weights = a + i * lda * 17;
    for (long j = 0; j < n;) {
      const int rows = static_cast<int>(std::min<long>(4, n - j));
      const float* input = b + j * ldb;
      float* output = c + j * ldc + i;
      switch (rows) {
        case 4: kt_match_npu::dot_gguf_mxfp4_mxfp8_many<4>(weights, input, ldb, k, output, ldc); break;
        case 3: kt_match_npu::dot_gguf_mxfp4_mxfp8_many<3>(weights, input, ldb, k, output, ldc); break;
        case 2: kt_match_npu::dot_gguf_mxfp4_mxfp8_many<2>(weights, input, ldb, k, output, ldc); break;
        default: *output = kt_match_npu::dot_gguf_mxfp4_mxfp8(weights, input, k); break;
      }
      j += rows;
    }
  }
}

}  // namespace kt_mxfp4_reuse

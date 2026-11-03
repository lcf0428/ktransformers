// Standalone regression: compile with patched llama.cpp/ggml-quants.c.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

#include "operators/llamafile/mxfp4_reuse.h"

// The dot kernels use ggml's FP16 lookup on builds without native FP16.
// Initialize it here without loading a model or linking the tensor runtime.
extern "C" { float ggml_table_f32_f16[1 << 16]; }

// Unused tensor-runtime references in ggml-quants.c must still resolve on COFF
// linkers. Fail immediately if the dot-only test ever reaches one of them.
extern "C" size_t ggml_row_size(enum ggml_type, int64_t) { std::abort(); }
extern "C" size_t ggml_type_size(enum ggml_type) { std::abort(); }
extern "C" void ggml_print_backtrace(void) { std::abort(); }

static void check_equal(float actual, float expected) {
  if (std::memcmp(&actual, &expected, sizeof(float)) != 0) {
    std::fprintf(stderr, "different results: %.9g != %.9g\n", actual, expected);
    throw std::runtime_error("token reuse changed a dot result");
  }
}

int main() {
#if defined(__AVX512BF16__) && defined(__AVX512F__)
  std::puts("MXFP8 match: AVX512-BF16");
#else
  std::puts("MXFP8 match: scalar");
#endif
#if defined(__ARM_NEON)
  std::puts("Q8_0: NEON");
#elif defined(__AVX2__)
  std::puts("Q8_0: AVX2");
#else
  std::puts("Q8_0: scalar");
#endif
  for (int h = 0; h < 65536; ++h) {
    const int e = (h >> 10) & 31, f = h & 1023;
    float value = e == 0 ? std::ldexp(float(f), -24) : std::ldexp(float(1024 + f), e - 25);
    if (e == 31) {
      const uint32_t special = f ? 0x7fc00000u : 0x7f800000u;
      std::memcpy(&value, &special, sizeof(value));
    }
    ggml_table_f32_f16[h] = h & 32768 ? -value : value;
  }
  std::mt19937 random(1024);
  int cases = 0;
  for (long n : {1, 2, 3, 4, 5, 8, 17, 32, 48, 64}) {
    for (long blocks : {1, 2, 3, 7, 64, 128}) {
      for (bool zero : {false, true}) {
        constexpr long m = 3;
        const long lda = blocks + 2, ldb = blocks + 3, ldc = m + 5;
        const long float_stride = blocks * 32 + 11;
        std::vector<block_mxfp4> weights(m * lda);
        std::vector<block_q8_0> q8(n * ldb);
        std::vector<float> input(n * float_stride);
        for (auto& w : weights) {
          // Include subnormal/low E8M0 scales and both FP4 signed zero codes.
          const uint8_t exponents[] = {0, 1, 2, 120, 127, 130};
          w.e = exponents[random() % 6];
          for (auto& q : w.qs) q = uint8_t(random());
        }
        for (auto& b : q8) {
          // Q8_0 quantization emits [-127,127]; include FP16 subnormal scales.
          const uint16_t scales[] = {0, 1, 0x3000, 0x3800, 0x3c00};
          const uint16_t scale = scales[random() % 5];
          std::memcpy(&b.d, &scale, sizeof(scale));
          for (auto& q : b.qs) q = zero ? 0 : int(random() % 255) - 127;
        }
        for (auto& x : input) x = zero ? 0.f : float(int(random() % 1025) - 512) / 512.f;
        for (long r = 0; r < n; ++r) kt_match_npu::quant_dequant(input.data() + r * float_stride, blocks * 32);
        const auto original_input = input;
        constexpr float sentinel = 12345.f;
        std::vector<float> output(n * ldc, sentinel);
        kt_mxfp4_reuse::q8_matmul(m, n, blocks, weights.data(), lda, q8.data(), ldb, output.data(), ldc);
        for (long r = 0; r < n; ++r) {
          for (long i = 0; i < m; ++i) {
            float expected;
            ggml_vec_dot_mxfp4_q8_0(blocks * 32, &expected, 0, &weights[i * lda], 0, &q8[r * ldb], 0, 1);
            check_equal(output[r * ldc + i], std::isfinite(expected) ? expected : 0.f);
          }
          for (long i = m; i < ldc; ++i) check_equal(output[r * ldc + i], sentinel);
        }
        std::fill(output.begin(), output.end(), sentinel);
        kt_mxfp4_reuse::match_matmul(m, n, blocks, reinterpret_cast<const uint8_t*>(weights.data()),
                                    lda, input.data(), float_stride, output.data(), ldc);
        for (long r = 0; r < n; ++r) {
          for (long i = 0; i < m; ++i) {
            float expected = kt_match_npu::dot_gguf_mxfp4_mxfp8(
                reinterpret_cast<const uint8_t*>(&weights[i * lda]), input.data() + r * float_stride, blocks);
            check_equal(output[r * ldc + i], expected);
          }
          for (long i = m; i < ldc; ++i) check_equal(output[r * ldc + i], sentinel);
        }
        if (input != original_input) throw std::runtime_error("input was modified");
        ++cases;
      }
    }
  }
  std::printf("PASS: %d matrix cases, Q8_0 and MXFP8, reuse=%d\n", cases, int(kt_mxfp4_reuse::enabled()));
}

#pragma once

/// TurboQuant: Near-optimal scalar quantization for vector search
///
/// Implementation of the MSE and inner-product quantizers from
/// "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
/// (Zandieh et al., ICLR 2026).
///
/// Uses randomised Hadamard transform for the rotation (O(d log d) vs O(d²)
/// for a dense Haar-random matrix). NEON-optimised for ARM.

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define TQ_NEON 1
#else
#define TQ_NEON 0
#endif

namespace turbo_quant {

// ── Constants ───────────────────────────────────────────────────────────────

static constexpr uint32_t kMaxBitWidth = 4;
static constexpr uint32_t kMaxCentroids = 1 << kMaxBitWidth; // 16

// ── Codebook ────────────────────────────────────────────────────────────────

/// Lloyd-Max centroids for each supported bit-width.
/// Centroids are for the *unit-variance* case (i.e. optimal for N(0,1)
/// in the high-d Gaussian limit). The quantizer scales by 1/√d_padded
/// internally after rotation.
///
/// Call set_centroids() with your own tables, or use
/// Codebook::gaussian_approx() for the standard N(0,1) Lloyd-Max values.
struct Codebook {
    /// Sorted ascending centroid values for each bit-width (1..4).
    /// centroids[b] has 2^b entries.
    std::array<std::array<float, kMaxCentroids>, kMaxBitWidth + 1> centroids{};

    /// Decision boundaries: boundaries[b] has 2^b − 1 entries.
    /// boundary[i] is the midpoint between centroid[i] and centroid[i+1].
    std::array<std::array<float, kMaxCentroids>, kMaxBitWidth + 1> boundaries{};

    /// Number of centroids for each bit-width.
    std::array<uint32_t, kMaxBitWidth + 1> num_centroids{};

    /// Set centroids for a given bit-width. Computes boundaries automatically.
    void set_centroids(uint32_t bit_width, const float* c, uint32_t count);

    /// Default codebook: Lloyd-Max for N(0,1) (high-d Gaussian approximation).
    /// Accurate for d ≥ ~100.  Replace with exact Beta-distribution solutions
    /// for lower d.
    static Codebook gaussian_approx();
};

// ── Quantized vector storage ────────────────────────────────────────────────

struct QuantizedVector {
    std::vector<uint8_t> mse_indices; // bit-packed MSE quantisation indices
    std::vector<uint8_t> qjl_signs;   // bit-packed QJL signs (prod variant)
    float norm{0.0f};                  // ‖x‖₂
    float residual_norm{0.0f};         // ‖r‖₂  (prod variant only)
    uint32_t dim{0};                   // original dimension
    uint32_t dim_padded{0};            // padded to next power of 2
    uint32_t bit_width{0};

    uint32_t mse_bytes() const { return (dim_padded * bit_width + 7) / 8; }
    uint32_t qjl_bytes() const { return (dim_padded + 7) / 8; }
    uint32_t total_bytes() const {
        return mse_bytes() + qjl_bytes() + 2 * sizeof(float);
    }
};

// ── Pre-computed query state ────────────────────────────────────────────────

struct QueryState {
    std::vector<float> rotated;     // Π · q  (length dim_padded)
    std::vector<float> qjl_rotated; // S · q  (length dim_padded, prod only)

    // Precomputed ADC tables: lut[i * n_centroids + j] = rotated[i] * centroid[j].
    // Computed once in prepare_query, eliminates per-doc FMA.
    std::vector<float> lut;
    uint32_t lut_centroids{0};      // 2^b centroids stored per coordinate
};

// ── Configuration ───────────────────────────────────────────────────────────

struct Config {
    uint32_t dim;
    uint32_t bit_width;  // 1, 2, 3, or 4
    bool use_prod;       // inner-product variant with QJL correction
    uint64_t seed{42};
};

// ── TurboQuant quantiser ────────────────────────────────────────────────────

class TurboQuant {
public:
    TurboQuant(const Config& cfg, const Codebook& cb);

    /// Quantise a single vector (length = cfg.dim).
    QuantizedVector quantize(const float* x) const;

    /// Dequantise to float vector (length = cfg.dim).
    void dequantize(const QuantizedVector& qv, float* out) const;

    /// Prepare rotated query for fast scoring.
    QueryState prepare_query(const float* query) const;

    /// Asymmetric dot product: float query × quantised document.
    float dot_asymmetric(const QueryState& qs,
                         const QuantizedVector& doc) const;

    /// Batch asymmetric dot products.
    void dot_asymmetric_batch(const QueryState& qs,
                              const QuantizedVector* docs,
                              uint32_t n,
                              float* out) const;

    /// Batch asymmetric dot products using precomputed ADC tables.
    void dot_asymmetric_lut_batch(const QueryState& qs,
                                  const QuantizedVector* docs,
                                  uint32_t n,
                                  float* out) const;

    uint32_t dim() const { return cfg_.dim; }
    uint32_t dim_padded() const { return dim_padded_; }
    uint32_t bit_width() const { return cfg_.bit_width; }

private:
    Config cfg_;
    Codebook cb_;
    uint32_t dim_padded_;

    // For the MSE rotation  Π = (1/√n) H · diag(rot_signs_)
    std::vector<float> rot_signs_;     // ±1, length dim_padded_
    // For the QJL rotation  S = (1/√n) H · diag(qjl_signs_)
    std::vector<float> qjl_rot_signs_; // ±1, length dim_padded_ (prod only)

    // Scaled centroids: centroids[b][i] / √dim_padded_
    std::array<std::array<float, kMaxCentroids>, kMaxBitWidth + 1> scaled_centroids_{};
    std::array<std::array<float, kMaxCentroids>, kMaxBitWidth + 1> scaled_boundaries_{};

    // ── internal helpers ────────────────────────────────────────────────
    void rotate_forward(const float* in, float* out) const;
    void rotate_inverse(const float* in, float* out) const;
    void qjl_forward(const float* in, float* out) const;

    void quantize_coords(const float* y, uint8_t* packed) const;
    void dequantize_coords(const uint8_t* packed, float* y) const;

    /// Core NEON-optimised dot: Σ rotated_query[i] × centroid[idx[i]]
    float dot_mse_core(const float* rotated_query,
                       const uint8_t* packed_indices) const;

    /// QJL dot: Σ qjl_rotated_query[i] × sign[i]
    float dot_qjl_core(const float* qjl_rotated,
                        const uint8_t* packed_signs,
                        uint32_t n) const;
};

// ── Low-level utilities (exposed for testing) ───────────────────────────────

namespace detail {

inline uint32_t next_pow2(uint32_t v) {
    v--;
    v |= v >> 1; v |= v >> 2; v |= v >> 4;
    v |= v >> 8; v |= v >> 16;
    return v + 1;
}

/// In-place Fast Walsh–Hadamard Transform.  NEON-optimised.
void fwht(float* data, uint32_t n);

/// Pack b-bit indices into bytes (little-endian bit order).
void pack_bits(const uint8_t* values, uint8_t* packed,
               uint32_t count, uint32_t bits);

/// Unpack b-bit indices from bytes.
void unpack_bits(const uint8_t* packed, uint8_t* values,
                 uint32_t count, uint32_t bits);

/// Generate random ±1 signs from a seed.
void generate_signs(float* signs, uint32_t n, uint64_t seed);

} // namespace detail
} // namespace turbo_quant

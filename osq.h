#pragma once

/// Optimised Scalar Quantization (OSQ)
///
/// Implementation of Elastic's OSQ for vector search.  Key property:
/// uniform grid spacing within a per-vector interval [a, b] enables
/// decomposition of dot products into integer arithmetic + scalar
/// corrections.  This is dramatically faster than lookup-table-based
/// approaches (like TurboQuant) on CPU SIMD.
///
/// Pipeline:
///   1. Compute corpus centroid c = mean of all vectors in the segment.
///   2. Centre each vector: x' = x − c  (quantise the residual).
///   3. Per-vector: fit interval [a, b] via the anisotropic objective.
///
/// If cosine similarity is desired, normalise vectors to unit length
/// *before* passing them to quantize().
///
/// Dot product decomposition (centred residual x'):
///
///   x̃'_i = a + idx_i · Δ      where Δ = (b − a) / (2^bits − 1)
///
///   Asymmetric:  ⟨q, x⟩ = ⟨q, c⟩ + (⟨c, x⟩ − ‖c‖²)
///                         + norm · (a · Σq'_i + Δ · Σ(q'_i · idx_i))
///                 where q' = q − c is the centred query
///
///   Symmetric:   ⟨x, y⟩ = ⟨x, c⟩ + ⟨c, y⟩ − ‖c‖²
///                         + norm_x · norm_y ·
///                           (d·a_x·a_y + a_x·Δ_y·Σidy
///                            + a_y·Δ_x·Σidx + Δ_x·Δ_y·Σ(idx·idy))
///
/// The Σ(idx_i · idy_i) term is a pure integer dot product — the
/// payoff for uniform grids.
///
/// NEON-optimised for ARM.

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define OSQ_NEON 1
#else
#define OSQ_NEON 0
#endif

namespace osq {

static constexpr uint32_t kMaxBitWidth = 4;

// ── Configuration ───────────────────────────────────────────────────────

struct Config {
    uint32_t dim;
    uint32_t bit_width;         // 1, 2, 3, or 4
    uint32_t refine_iters{5};   // coordinate-descent iterations (0 = init only)
    float lambda{0.1f};         // anisotropic blending: (1-λ)(x·e)²/‖x‖² + λ‖e‖²
};

// ── Quantised vector ────────────────────────────────────────────────────

struct OSQVector {
    std::vector<uint8_t> packed;  // bit-packed indices in [0, 2^b − 1]
    float a{0};                    // interval lower bound (of centred residual)
    float b{0};                    // interval upper bound
    float delta{0};                // (b − a) / (2^bits − 1)
    float norm{1.0f};              // reserved (always 1.0; kept for ABI compat)
    float sum_indices{0};          // Σ idx_i  (precomputed for symmetric dot)
    float cx{0};                   // ⟨c, x⟩ = dot(centroid, original_vector)
    uint32_t dim{0};
    uint32_t bit_width{0};

    // Precomputed bit-planes for 2-bit and 4-bit vectors.
    // Each plane is in 1-bit packed format: (dim+7)/8 bytes.
    // plane[k][i] has bit j set iff bit k of index (8i+j) is set.
    // 2-bit vectors use planes [0..1]; 4-bit use [0..3].
    // Enables decomposing integer dot products into AND+popcount.
    std::array<std::vector<uint8_t>, 4> bit_planes;

    uint32_t packed_bytes() const { return (dim * bit_width + 7) / 8; }
};

// ── Query state ─────────────────────────────────────────────────────────

struct QueryState {
    std::vector<float> query;  // centred query: q' = q − c
    float query_sum{0};         // Σ q'_i  (centred query sum)
    float qc{0};                // ⟨q, c⟩ = dot(original query, centroid)
};

// ── OSQ quantiser ───────────────────────────────────────────────────────

class OSQ {
public:
    explicit OSQ(const Config& cfg);

    /// Set the corpus centroid (call once, before quantize / prepare_query).
    /// If never called, centroid defaults to zero (no centering).
    void set_centroid(const float* centroid);

    /// Quantise a single vector.
    OSQVector quantize(const float* x) const;

    /// Dequantise to float vector.
    void dequantize(const OSQVector& qv, float* out) const;

    /// Prepare query for scoring.
    QueryState prepare_query(const float* query) const;

    /// Asymmetric dot product: float query × quantised doc.
    ///   = qc + doc.norm × (a · query_sum + Δ · Σ(q_i · idx_i))
    /// where qc = ⟨q, centroid⟩.
    float dot_asymmetric(const QueryState& qs,
                         const OSQVector& doc) const;

    /// Symmetric dot product: quantised × quantised.
    ///   Uses integer dot product core + centroid corrections.
    float dot_symmetric(const OSQVector& x,
                        const OSQVector& y) const;

    /// Batch asymmetric dot products.
    void dot_asymmetric_batch(const QueryState& qs,
                              const OSQVector* docs, uint32_t n,
                              float* out) const;

    uint32_t dim() const { return cfg_.dim; }
    uint32_t bit_width() const { return cfg_.bit_width; }

private:
    Config cfg_;
    uint32_t max_idx_;  // 2^bits − 1

    // Corpus centroid (length dim).  Defaults to zero.
    std::vector<float> centroid_;
    float centroid_sq_{0};  // ‖c‖²

    // Optimal interval half-widths for N(0,1) per bit-width.
    static constexpr float kInitScale[] = {
        0.0f,     // unused (index 0)
        0.7979f,  // 1-bit
        1.4935f,  // 2-bit
        2.0000f,  // 3-bit (approximate)
        2.5140f   // 4-bit
    };

    // ── Interval refinement ─────────────────────────────────────────────
    void init_interval(const float* x, uint32_t n,
                       float& a, float& b) const;
    void refine_interval(const float* x, uint32_t n,
                         float& a, float& b) const;
    float quantization_error(const float* x, uint32_t n,
                             float xx, float a, float b) const;

    // ── Core dot product kernels ────────────────────────────────────────

    /// Σ(q_i · idx_i) with float query and packed integer indices.
    float float_int_dot(const float* q,
                        const uint8_t* packed, uint32_t n,
                        uint32_t bits) const;

    /// Σ(idx_x_i · idx_y_i)  —  pure integer dot product.
    /// Supports mixed bit widths (e.g. 4-bit doc × 1-bit query).
    uint64_t int_int_dot(const uint8_t* packed_x, uint32_t bits_x,
                         const uint8_t* packed_y, uint32_t bits_y,
                         uint32_t n) const;
};

// ── Bit packing (shared with turbo_quant) ───────────────────────────────

namespace detail {

void pack_bits(const uint8_t* vals, uint8_t* out,
               uint32_t count, uint32_t bits);
void unpack_bits(const uint8_t* packed, uint8_t* out,
                 uint32_t count, uint32_t bits);

} // namespace detail
} // namespace osq

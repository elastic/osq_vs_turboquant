#include "turbo_quant.h"

#include <algorithm>
#include <numeric>

namespace turbo_quant {

// ═══════════════════════════════════════════════════════════════════════════
//  Codebook
// ═══════════════════════════════════════════════════════════════════════════

void Codebook::set_centroids(uint32_t b, const float* c, uint32_t count) {
    assert(b >= 1 && b <= kMaxBitWidth);
    assert(count == (1u << b));
    num_centroids[b] = count;
    for (uint32_t i = 0; i < count; ++i) centroids[b][i] = c[i];

    // Decision boundaries = midpoints between adjacent centroids.
    for (uint32_t i = 0; i + 1 < count; ++i)
        boundaries[b][i] = 0.5f * (c[i] + c[i + 1]);
}

Codebook Codebook::gaussian_approx() {
    Codebook cb{};

    // Lloyd-Max centroids for N(0,1), sorted ascending.
    // These are the high-d limit of the exact Beta-distribution solutions.
    // Replace with exact values for your target dimension if desired.

    // 1-bit  (2 centroids)
    static const float c1[] = {-0.7978845608f, 0.7978845608f};
    cb.set_centroids(1, c1, 2);

    // 2-bit  (4 centroids)
    static const float c2[] = {-1.5104176088f, -0.4527800398f,
                                0.4527800398f,  1.5104176088f};
    cb.set_centroids(2, c2, 4);

    // 3-bit  (8 centroids)
    static const float c3[] = {-2.1519680965f, -1.3439675635f,
                               -0.7560052255f, -0.2451135846f,
                                0.2451135846f,  0.7560052255f,
                                1.3439675635f,  2.1519680965f};
    cb.set_centroids(3, c3, 8);

    // 4-bit  (16 centroids)
    static const float c4[] = {
        -2.7325521685f, -2.0690079824f, -1.6180334840f, -1.2562297404f,
        -0.9423520927f, -0.6567567030f, -0.3880823458f, -0.1284195765f,
         0.1284195765f,  0.3880823458f,  0.6567567030f,  0.9423520927f,
         1.2562297404f,  1.6180334840f,  2.0690079824f,  2.7325521685f};
    cb.set_centroids(4, c4, 16);

    return cb;
}

// ═══════════════════════════════════════════════════════════════════════════
//  detail:: utilities
// ═══════════════════════════════════════════════════════════════════════════

namespace detail {

// ── Fast Walsh–Hadamard Transform (in-place) ────────────────────────────

void fwht(float* data, uint32_t n) {
    for (uint32_t len = 1; len < n; len <<= 1) {
        for (uint32_t blk = 0; blk < n; blk += 2 * len) {
            uint32_t j = 0;
#if TQ_NEON
            for (; j + 4 <= len; j += 4) {
                float32x4_t a = vld1q_f32(data + blk + j);
                float32x4_t b = vld1q_f32(data + blk + j + len);
                vst1q_f32(data + blk + j,       vaddq_f32(a, b));
                vst1q_f32(data + blk + j + len,  vsubq_f32(a, b));
            }
#endif
            for (; j < len; ++j) {
                float a = data[blk + j];
                float b = data[blk + j + len];
                data[blk + j]       = a + b;
                data[blk + j + len] = a - b;
            }
        }
    }
}

// ── Bit packing / unpacking ─────────────────────────────────────────────

// Specialised packers for common bit widths avoid the general bit-shift loop.

static void pack_1bit(const uint8_t* vals, uint8_t* out, uint32_t n) {
    uint32_t full = n / 8;
    for (uint32_t i = 0; i < full; ++i) {
        const uint8_t* v = vals + i * 8;
        out[i] = uint8_t(v[0] | (v[1]<<1) | (v[2]<<2) | (v[3]<<3) |
                         (v[4]<<4) | (v[5]<<5) | (v[6]<<6) | (v[7]<<7));
    }
    if (n % 8) {
        uint8_t byte = 0;
        for (uint32_t j = 0; j < n % 8; ++j)
            byte |= (vals[full * 8 + j] & 1) << j;
        out[full] = byte;
    }
}

static void pack_2bit(const uint8_t* vals, uint8_t* out, uint32_t n) {
    uint32_t full = n / 4;
    for (uint32_t i = 0; i < full; ++i) {
        const uint8_t* v = vals + i * 4;
        out[i] = uint8_t(v[0] | (v[1]<<2) | (v[2]<<4) | (v[3]<<6));
    }
    if (n % 4) {
        uint8_t byte = 0;
        for (uint32_t j = 0; j < n % 4; ++j)
            byte |= (vals[full * 4 + j] & 3) << (j * 2);
        out[full] = byte;
    }
}

static void pack_4bit(const uint8_t* vals, uint8_t* out, uint32_t n) {
    uint32_t full = n / 2;
    for (uint32_t i = 0; i < full; ++i)
        out[i] = uint8_t((vals[2*i] & 0xF) | (vals[2*i+1] << 4));
    if (n & 1)
        out[full] = vals[n - 1] & 0xF;
}

static void pack_general(const uint8_t* vals, uint8_t* out,
                          uint32_t n, uint32_t bits) {
    uint32_t total_bytes = (n * bits + 7) / 8;
    std::memset(out, 0, total_bytes);
    uint32_t bit_pos = 0;
    uint8_t mask = (1u << bits) - 1;
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t byte_idx  = bit_pos / 8;
        uint32_t bit_off   = bit_pos % 8;
        out[byte_idx] |= (vals[i] & mask) << bit_off;
        if (bit_off + bits > 8)
            out[byte_idx + 1] |= vals[i] >> (8 - bit_off);
        bit_pos += bits;
    }
}

void pack_bits(const uint8_t* vals, uint8_t* out,
               uint32_t n, uint32_t bits) {
    switch (bits) {
        case 1: pack_1bit(vals, out, n); break;
        case 2: pack_2bit(vals, out, n); break;
        case 4: pack_4bit(vals, out, n); break;
        default: pack_general(vals, out, n, bits); break;
    }
}

// ── Unpacking ───────────────────────────────────────────────────────────

static void unpack_1bit(const uint8_t* packed, uint8_t* out, uint32_t n) {
    uint32_t full = n / 8;
    for (uint32_t i = 0; i < full; ++i) {
        uint8_t b = packed[i];
        uint8_t* o = out + i * 8;
        o[0] = b & 1; o[1] = (b>>1)&1; o[2] = (b>>2)&1; o[3] = (b>>3)&1;
        o[4] = (b>>4)&1; o[5] = (b>>5)&1; o[6] = (b>>6)&1; o[7] = (b>>7)&1;
    }
    if (n % 8) {
        uint8_t b = packed[full];
        for (uint32_t j = 0; j < n % 8; ++j)
            out[full * 8 + j] = (b >> j) & 1;
    }
}

static void unpack_2bit(const uint8_t* packed, uint8_t* out, uint32_t n) {
    uint32_t full = n / 4;
    for (uint32_t i = 0; i < full; ++i) {
        uint8_t b = packed[i];
        uint8_t* o = out + i * 4;
        o[0] = b & 3; o[1] = (b>>2)&3; o[2] = (b>>4)&3; o[3] = (b>>6)&3;
    }
    if (n % 4) {
        uint8_t b = packed[full];
        for (uint32_t j = 0; j < n % 4; ++j)
            out[full * 4 + j] = (b >> (j*2)) & 3;
    }
}

static void unpack_4bit(const uint8_t* packed, uint8_t* out, uint32_t n) {
    uint32_t full = n / 2;
    for (uint32_t i = 0; i < full; ++i) {
        out[2*i]     = packed[i] & 0xF;
        out[2*i + 1] = packed[i] >> 4;
    }
    if (n & 1)
        out[n - 1] = packed[full] & 0xF;
}

static void unpack_general(const uint8_t* packed, uint8_t* out,
                            uint32_t n, uint32_t bits) {
    uint8_t mask = (1u << bits) - 1;
    uint32_t bit_pos = 0;
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t byte_idx = bit_pos / 8;
        uint32_t bit_off  = bit_pos % 8;
        uint8_t val = (packed[byte_idx] >> bit_off);
        if (bit_off + bits > 8)
            val |= packed[byte_idx + 1] << (8 - bit_off);
        out[i] = val & mask;
        bit_pos += bits;
    }
}

void unpack_bits(const uint8_t* packed, uint8_t* out,
                 uint32_t n, uint32_t bits) {
    switch (bits) {
        case 1: unpack_1bit(packed, out, n); break;
        case 2: unpack_2bit(packed, out, n); break;
        case 4: unpack_4bit(packed, out, n); break;
        default: unpack_general(packed, out, n, bits); break;
    }
}

// ── Random sign generation ──────────────────────────────────────────────

void generate_signs(float* signs, uint32_t n, uint64_t seed) {
    // SplitMix64 for fast, deterministic ±1 generation.
    auto splitmix = [](uint64_t& s) -> uint64_t {
        s += 0x9E3779B97F4A7C15ULL;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    };
    uint64_t state = seed;
    for (uint32_t i = 0; i < n; ++i)
        signs[i] = (splitmix(state) & 1) ? 1.0f : -1.0f;
}

} // namespace detail

// ═══════════════════════════════════════════════════════════════════════════
//  TurboQuant
// ═══════════════════════════════════════════════════════════════════════════

TurboQuant::TurboQuant(const Config& cfg, const Codebook& cb)
    : cfg_(cfg), cb_(cb) {
    assert(cfg.bit_width >= 1 && cfg.bit_width <= kMaxBitWidth);
    assert(cfg.dim > 0);

    dim_padded_ = detail::next_pow2(cfg.dim);

    // Generate random signs for the MSE rotation.
    rot_signs_.resize(dim_padded_);
    detail::generate_signs(rot_signs_.data(), dim_padded_, cfg.seed);

    // Scale centroids by 1/√dim_padded.
    float inv_sqrt_n = 1.0f / std::sqrt(static_cast<float>(dim_padded_));
    uint32_t b = cfg.bit_width;
    uint32_t nc = cb.num_centroids[b];
    for (uint32_t i = 0; i < nc; ++i)
        scaled_centroids_[b][i] = cb.centroids[b][i] * inv_sqrt_n;
    for (uint32_t i = 0; i + 1 < nc; ++i)
        scaled_boundaries_[b][i] = cb.boundaries[b][i] * inv_sqrt_n;

    // For the prod variant, also prepare codebook at bit_width − 1
    // (MSE stage uses b−1 bits; QJL uses the remaining 1 bit).
    if (cfg.use_prod && b > 1) {
        uint32_t bm = b - 1;
        uint32_t ncm = cb.num_centroids[bm];
        for (uint32_t i = 0; i < ncm; ++i)
            scaled_centroids_[bm][i] = cb.centroids[bm][i] * inv_sqrt_n;
        for (uint32_t i = 0; i + 1 < ncm; ++i)
            scaled_boundaries_[bm][i] = cb.boundaries[bm][i] * inv_sqrt_n;
    }

    // Generate QJL rotation signs (different seed).
    if (cfg.use_prod) {
        qjl_rot_signs_.resize(dim_padded_);
        detail::generate_signs(qjl_rot_signs_.data(), dim_padded_,
                               cfg.seed ^ 0xCAFEBABEDEADBEEFULL);
    }
}

// ── Rotation ────────────────────────────────────────────────────────────

void TurboQuant::rotate_forward(const float* in, float* out) const {
    uint32_t n = dim_padded_;
    // Zero-pad.
    std::memset(out, 0, n * sizeof(float));
    std::memcpy(out, in, cfg_.dim * sizeof(float));

    // Apply random sign flips.
    uint32_t i = 0;
#if TQ_NEON
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(out + i);
        float32x4_t s = vld1q_f32(rot_signs_.data() + i);
        vst1q_f32(out + i, vmulq_f32(v, s));
    }
#endif
    for (; i < n; ++i)
        out[i] *= rot_signs_[i];

    // FWHT.
    detail::fwht(out, n);

    // Normalise: multiply by 1/√n.
    float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(n));
    i = 0;
#if TQ_NEON
    float32x4_t scale = vdupq_n_f32(inv_sqrt);
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(out + i);
        vst1q_f32(out + i, vmulq_f32(v, scale));
    }
#endif
    for (; i < n; ++i)
        out[i] *= inv_sqrt;
}

void TurboQuant::rotate_inverse(const float* in, float* out) const {
    uint32_t n = dim_padded_;
    // Π⁻¹ = (1/√n) D H   (FWHT first, then sign flips, then scale).
    std::memcpy(out, in, n * sizeof(float));

    detail::fwht(out, n);

    float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(n));
    uint32_t i = 0;
#if TQ_NEON
    float32x4_t scale = vdupq_n_f32(inv_sqrt);
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(out + i);
        float32x4_t s = vld1q_f32(rot_signs_.data() + i);
        vst1q_f32(out + i, vmulq_f32(vmulq_f32(v, s), scale));
    }
#endif
    for (; i < n; ++i)
        out[i] *= rot_signs_[i] * inv_sqrt;
}

void TurboQuant::qjl_forward(const float* in, float* out) const {
    uint32_t n = dim_padded_;
    std::memset(out, 0, n * sizeof(float));
    std::memcpy(out, in, cfg_.dim * sizeof(float));

    uint32_t i = 0;
#if TQ_NEON
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(out + i);
        float32x4_t s = vld1q_f32(qjl_rot_signs_.data() + i);
        vst1q_f32(out + i, vmulq_f32(v, s));
    }
#endif
    for (; i < n; ++i)
        out[i] *= qjl_rot_signs_[i];

    detail::fwht(out, n);

    // Normalise.
    float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(n));
    i = 0;
#if TQ_NEON
    float32x4_t sc = vdupq_n_f32(inv_sqrt);
    for (; i + 4 <= n; i += 4)
        vst1q_f32(out + i, vmulq_f32(vld1q_f32(out + i), sc));
#endif
    for (; i < n; ++i)
        out[i] *= inv_sqrt;
}

// ── Coordinate quantisation / dequantisation ────────────────────────────

void TurboQuant::quantize_coords(const float* y, uint8_t* packed) const {
    uint32_t n  = dim_padded_;
    uint32_t b  = cfg_.use_prod ? cfg_.bit_width - 1 : cfg_.bit_width;
    uint32_t nb = cb_.num_centroids[b] - 1; // number of boundaries

    const float* bnds = scaled_boundaries_[b].data();

    // Allocate unpacked index buffer (could stack-allocate for small n).
    std::vector<uint8_t> indices(n);

    // For each coordinate: count how many boundaries the value exceeds.
    // This is a branchless linear scan — 'nb' is small (≤ 15).
    uint32_t i = 0;
#if TQ_NEON
    // NEON: process 4 coordinates at a time.
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(y + i);
        int32x4_t idx = vdupq_n_s32(0);
        for (uint32_t k = 0; k < nb; ++k) {
            float32x4_t bnd = vdupq_n_f32(bnds[k]);
            // vcgtq_f32 returns 0xFFFFFFFF (= -1 as int32) when v > bnd.
            uint32x4_t gt = vcgtq_f32(v, bnd);
            idx = vsubq_s32(idx, vreinterpretq_s32_u32(gt));
        }
        // Store indices.
        indices[i]     = static_cast<uint8_t>(vgetq_lane_s32(idx, 0));
        indices[i + 1] = static_cast<uint8_t>(vgetq_lane_s32(idx, 1));
        indices[i + 2] = static_cast<uint8_t>(vgetq_lane_s32(idx, 2));
        indices[i + 3] = static_cast<uint8_t>(vgetq_lane_s32(idx, 3));
    }
#endif
    // Scalar tail.
    for (; i < n; ++i) {
        uint8_t idx = 0;
        for (uint32_t k = 0; k < nb; ++k)
            idx += (y[i] > bnds[k]) ? 1 : 0;
        indices[i] = idx;
    }

    detail::pack_bits(indices.data(), packed, n, b);
}

void TurboQuant::dequantize_coords(const uint8_t* packed, float* y) const {
    uint32_t n = dim_padded_;
    uint32_t b = cfg_.use_prod ? cfg_.bit_width - 1 : cfg_.bit_width;

    const float* cents = scaled_centroids_[b].data();

    std::vector<uint8_t> indices(n);
    detail::unpack_bits(packed, indices.data(), n, b);

    uint32_t i = 0;
    // Centroid lookup — the hot gather loop.
    // With non-uniform centroids this is inherently a table lookup;
    // NEON can help a little but the gather is the bottleneck.
#if TQ_NEON
    // For 1-bit: only 2 centroids.  Use vbsl to select.
    if (b == 1) {
        float32x4_t c0 = vdupq_n_f32(cents[0]);
        float32x4_t c1 = vdupq_n_f32(cents[1]);
        for (; i + 8 <= n; i += 8) {
            // 8 indices from 1 byte.
            uint8_t byte = packed[i / 8];
            for (uint32_t j = 0; j < 8 && i + j < n; ++j)
                y[i + j] = (byte >> j) & 1 ? c1[0] : c0[0];
            // (For 1-bit, a fully branchless NEON expansion is possible
            //  but the byte→lane expansion is the bottleneck regardless.)
        }
    }
#endif
    for (; i < n; ++i)
        y[i] = cents[indices[i]];
}

// ── Public: quantize ────────────────────────────────────────────────────

QuantizedVector TurboQuant::quantize(const float* x) const {
    QuantizedVector qv;
    qv.dim        = cfg_.dim;
    qv.dim_padded = dim_padded_;
    qv.bit_width  = cfg_.bit_width;

    // Compute and store the norm.
    float norm_sq = 0;
    uint32_t i = 0;
#if TQ_NEON
    float32x4_t acc = vdupq_n_f32(0);
    for (; i + 4 <= cfg_.dim; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        acc = vmlaq_f32(acc, v, v);
    }
    norm_sq = vaddvq_f32(acc);
#endif
    for (; i < cfg_.dim; ++i)
        norm_sq += x[i] * x[i];
    qv.norm = std::sqrt(norm_sq);

    if (qv.norm < 1e-30f) {
        // Zero vector: all-zero quantised output.
        uint32_t b = cfg_.use_prod ? cfg_.bit_width - 1 : cfg_.bit_width;
        qv.mse_indices.resize((dim_padded_ * b + 7) / 8, 0);
        if (cfg_.use_prod)
            qv.qjl_signs.resize((dim_padded_ + 7) / 8, 0);
        return qv;
    }

    // Normalise to unit sphere, then rotate.
    std::vector<float> x_unit(cfg_.dim);
    float inv_norm = 1.0f / qv.norm;
    for (uint32_t j = 0; j < cfg_.dim; ++j)
        x_unit[j] = x[j] * inv_norm;

    std::vector<float> y(dim_padded_);
    rotate_forward(x_unit.data(), y.data());

    // Quantise coordinates.
    uint32_t b = cfg_.use_prod ? cfg_.bit_width - 1 : cfg_.bit_width;
    qv.mse_indices.resize((dim_padded_ * b + 7) / 8);
    quantize_coords(y.data(), qv.mse_indices.data());

    // ── QJL correction (prod variant) ───────────────────────────────────
    if (cfg_.use_prod) {
        // Dequantise MSE to get x̃_mse in rotated space.
        std::vector<float> y_hat(dim_padded_);
        dequantize_coords(qv.mse_indices.data(), y_hat.data());

        // Residual in rotated space: r_rot = y - y_hat
        // (Residual norm is preserved under rotation.)
        std::vector<float> r_rot(dim_padded_);
        float res_sq = 0;
        uint32_t j = 0;
#if TQ_NEON
        float32x4_t racc = vdupq_n_f32(0);
        for (; j + 4 <= dim_padded_; j += 4) {
            float32x4_t a = vld1q_f32(y.data() + j);
            float32x4_t b = vld1q_f32(y_hat.data() + j);
            float32x4_t r = vsubq_f32(a, b);
            vst1q_f32(r_rot.data() + j, r);
            racc = vmlaq_f32(racc, r, r);
        }
        res_sq = vaddvq_f32(racc);
#endif
        for (; j < dim_padded_; ++j) {
            float r = y[j] - y_hat[j];
            r_rot[j] = r;
            res_sq += r * r;
        }
        qv.residual_norm = std::sqrt(res_sq);

        // Apply QJL rotation to the residual (in the original space).
        // First inverse-rotate residual back, then QJL-rotate it.
        // Optimisation: since both are Hadamard-based we could compose them,
        // but for clarity we do two passes.
        std::vector<float> r_orig(dim_padded_);
        rotate_inverse(r_rot.data(), r_orig.data());

        std::vector<float> s(dim_padded_);
        qjl_forward(r_orig.data(), s.data());

        // 1-bit sign quantisation.
        std::vector<uint8_t> sign_vals(dim_padded_);
        for (uint32_t j = 0; j < dim_padded_; ++j)
            sign_vals[j] = s[j] >= 0 ? 1 : 0;
        qv.qjl_signs.resize((dim_padded_ + 7) / 8);
        detail::pack_bits(sign_vals.data(), qv.qjl_signs.data(),
                          dim_padded_, 1);
    }

    return qv;
}

// ── Public: dequantize ──────────────────────────────────────────────────

void TurboQuant::dequantize(const QuantizedVector& qv, float* out) const {
    std::vector<float> y(dim_padded_);
    dequantize_coords(qv.mse_indices.data(), y.data());

    // Rescale by original norm.
    uint32_t i = 0;
#if TQ_NEON
    float32x4_t nv = vdupq_n_f32(qv.norm);
    for (; i + 4 <= dim_padded_; i += 4)
        vst1q_f32(y.data() + i,
                  vmulq_f32(vld1q_f32(y.data() + i), nv));
#endif
    for (; i < dim_padded_; ++i)
        y[i] *= qv.norm;

    // Inverse rotation.
    std::vector<float> x_hat(dim_padded_);
    rotate_inverse(y.data(), x_hat.data());

    // TODO: add QJL reconstruction for prod variant.
    // x̃_qjl = √(π/2) × (residual_norm / √n) × Π_qjl⁻¹ · qjl_float
    // This is left for the caller to add if needed; the asymmetric dot
    // product below handles it without full dequantization.

    std::memcpy(out, x_hat.data(), qv.dim * sizeof(float));
}

// ── Public: prepare_query ───────────────────────────────────────────────

QueryState TurboQuant::prepare_query(const float* query) const {
    QueryState qs;
    qs.rotated.resize(dim_padded_);
    rotate_forward(query, qs.rotated.data());

    if (cfg_.use_prod) {
        qs.qjl_rotated.resize(dim_padded_);
        qjl_forward(query, qs.qjl_rotated.data());
    }

    // Precompute ADC lookup tables: lut[i * nc + j] = rotated[i] * centroid[j].
    uint32_t b = cfg_.use_prod ? cfg_.bit_width - 1 : cfg_.bit_width;
    uint32_t nc = 1u << b;
    qs.lut_centroids = nc;
    qs.lut.resize(dim_padded_ * nc);
    const float* cents = scaled_centroids_[b].data();
    for (uint32_t i = 0; i < dim_padded_; ++i) {
        float qi = qs.rotated[i];
        for (uint32_t j = 0; j < nc; ++j)
            qs.lut[i * nc + j] = qi * cents[j];
    }

    return qs;
}

// ── Core: MSE dot product ───────────────────────────────────────────────
//
// Computes  Σᵢ rotated_query[i] × centroid[idx[i]]
//
// This is the inner loop for asymmetric distance and is the main
// performance-critical path.  The gather over the centroid table is
// inherently serial for non-uniform centroids.  We optimise by
// specialising per bit-width.

float TurboQuant::dot_mse_core(const float* rq,
                                const uint8_t* packed) const {
    uint32_t n = dim_padded_;
    uint32_t b = cfg_.use_prod ? cfg_.bit_width - 1 : cfg_.bit_width;
    const float* cents = scaled_centroids_[b].data();

    float sum = 0;

    // ── 4-bit fast path ────────────────────────────────────────────────
    if (b == 4) {
        uint32_t i = 0;
#if TQ_NEON
        float32x4_t acc0 = vdupq_n_f32(0);
        float32x4_t acc1 = vdupq_n_f32(0);
        // Process 8 coordinates per iteration (4 packed bytes).
        for (; i + 8 <= n; i += 8) {
            // Unpack 4 bytes → 8 nibble indices.
            uint8_t b0 = packed[i/2],   b1 = packed[i/2+1],
                    b2 = packed[i/2+2], b3 = packed[i/2+3];

            // Gather centroids via scalar lookup (NEON lacks float gather).
            float32x4_t c0 = {cents[b0 & 0xF], cents[b0 >> 4],
                              cents[b1 & 0xF], cents[b1 >> 4]};
            float32x4_t c1 = {cents[b2 & 0xF], cents[b2 >> 4],
                              cents[b3 & 0xF], cents[b3 >> 4]};

            float32x4_t q0 = vld1q_f32(rq + i);
            float32x4_t q1 = vld1q_f32(rq + i + 4);
            acc0 = vmlaq_f32(acc0, q0, c0);
            acc1 = vmlaq_f32(acc1, q1, c1);
        }
        sum = vaddvq_f32(vaddq_f32(acc0, acc1));
#endif
        for (; i < n; i += 2) {
            uint8_t byte = packed[i / 2];
            sum += rq[i]     * cents[byte & 0xF];
            if (i + 1 < n)
                sum += rq[i+1] * cents[byte >> 4];
        }
        return sum;
    }

    // ── 2-bit fast path ────────────────────────────────────────────────
    if (b == 2) {
        uint32_t i = 0;
#if TQ_NEON
        float32x4_t acc = vdupq_n_f32(0);
        for (; i + 4 <= n; i += 4) {
            uint8_t byte = packed[i / 4];
            float32x4_t c = {cents[byte & 3], cents[(byte>>2) & 3],
                             cents[(byte>>4) & 3], cents[(byte>>6) & 3]};
            acc = vmlaq_f32(acc, vld1q_f32(rq + i), c);
        }
        sum = vaddvq_f32(acc);
#endif
        for (; i < n; ++i) {
            uint32_t byte_idx = i / 4;
            uint32_t shift = (i % 4) * 2;
            uint8_t idx = (packed[byte_idx] >> shift) & 3;
            sum += rq[i] * cents[idx];
        }
        return sum;
    }

    // ── 1-bit fast path ────────────────────────────────────────────────
    if (b == 1) {
        float c0 = cents[0], c1 = cents[1];
        uint32_t i = 0;
#if TQ_NEON
        // Centroid values are just c0 and c1.
        // sign bit selects between them: sum += rq[i] * (bit ? c1 : c0)
        //  = rq[i] * c0 + rq[i] * bit * (c1 - c0)
        // Rewrite: sum = c0 * Σrq[i] + (c1-c0) * Σ(rq[i] where bit=1)
        float sum_all = 0, sum_ones = 0;
        float32x4_t sa = vdupq_n_f32(0), so = vdupq_n_f32(0);
        for (; i + 8 <= n; i += 8) {
            uint8_t byte = packed[i / 8];
            float32x4_t q0 = vld1q_f32(rq + i);
            float32x4_t q1 = vld1q_f32(rq + i + 4);
            sa = vaddq_f32(sa, vaddq_f32(q0, q1));

            // Expand bits to float masks.
            float m0 = (byte & 1) ? 1.0f : 0.0f;
            float m1 = (byte & 2) ? 1.0f : 0.0f;
            float m2 = (byte & 4) ? 1.0f : 0.0f;
            float m3 = (byte & 8) ? 1.0f : 0.0f;
            float32x4_t mask0 = {m0, m1, m2, m3};
            so = vmlaq_f32(so, q0, mask0);

            float m4 = (byte & 16) ? 1.0f : 0.0f;
            float m5 = (byte & 32) ? 1.0f : 0.0f;
            float m6 = (byte & 64) ? 1.0f : 0.0f;
            float m7 = (byte & 128) ? 1.0f : 0.0f;
            float32x4_t mask1 = {m4, m5, m6, m7};
            so = vmlaq_f32(so, q1, mask1);
        }
        sum_all  = vaddvq_f32(sa);
        sum_ones = vaddvq_f32(so);
#else
        float sum_all = 0, sum_ones = 0;
#endif
        for (; i < n; ++i) {
            uint8_t bit = (packed[i/8] >> (i%8)) & 1;
            sum_all += rq[i];
            if (bit) sum_ones += rq[i];
        }
        return c0 * sum_all + (c1 - c0) * sum_ones;
    }

    // ── General path (3-bit) ───────────────────────────────────────────
    {
        std::vector<uint8_t> indices(n);
        detail::unpack_bits(packed, indices.data(), n, b);
        for (uint32_t i = 0; i < n; ++i)
            sum += rq[i] * cents[indices[i]];
        return sum;
    }
}

// ── Core: QJL sign dot product ──────────────────────────────────────────
//
// Computes  Σᵢ qjl_rotated[i] × sign[i]   where sign ∈ {-1, +1}
// encoded as packed bits (1 = +1, 0 = -1).

float TurboQuant::dot_qjl_core(const float* sq,
                                 const uint8_t* packed_signs,
                                 uint32_t n) const {
    float sum = 0;
    uint32_t i = 0;

#if TQ_NEON
    float32x4_t acc0 = vdupq_n_f32(0);
    float32x4_t acc1 = vdupq_n_f32(0);
    float32x4_t pos = vdupq_n_f32(1.0f);
    float32x4_t neg = vdupq_n_f32(-1.0f);

    for (; i + 8 <= n; i += 8) {
        uint8_t byte = packed_signs[i / 8];

        // Expand 8 sign bits to two float32x4 vectors of ±1.
        // Use NEON bit-test: broadcast byte, AND with per-lane masks,
        // compare to zero.
        uint8x8_t bv = vdup_n_u8(byte);
        static const uint8_t bit_masks_arr[8] = {1,2,4,8,16,32,64,128};
        uint8x8_t masks = vld1_u8(bit_masks_arr);
        uint8x8_t bits = vtst_u8(bv, masks); // 0xFF if set, 0x00 if not

        // Expand lower 4 bits → 32-bit mask.
        uint16x8_t w = vmovl_u8(bits);
        uint32x4_t lo32 = vmovl_u16(vget_low_u16(w));
        uint32x4_t hi32 = vmovl_u16(vget_high_u16(w));

        // lo32/hi32 are 0x000000FF or 0x00000000; make proper masks.
        uint32x4_t mlo = vcgtq_u32(lo32, vdupq_n_u32(0));
        uint32x4_t mhi = vcgtq_u32(hi32, vdupq_n_u32(0));

        float32x4_t s0 = vbslq_f32(mlo, pos, neg);
        float32x4_t s1 = vbslq_f32(mhi, pos, neg);

        float32x4_t q0 = vld1q_f32(sq + i);
        float32x4_t q1 = vld1q_f32(sq + i + 4);
        acc0 = vmlaq_f32(acc0, q0, s0);
        acc1 = vmlaq_f32(acc1, q1, s1);
    }
    sum = vaddvq_f32(vaddq_f32(acc0, acc1));
#endif

    for (; i < n; ++i) {
        uint8_t bit = (packed_signs[i/8] >> (i%8)) & 1;
        float sign = bit ? 1.0f : -1.0f;
        sum += sq[i] * sign;
    }
    return sum;
}

// ── Public: asymmetric dot product ──────────────────────────────────────

float TurboQuant::dot_asymmetric(const QueryState& qs,
                                  const QuantizedVector& doc) const {
    // MSE contribution:
    //   ⟨q, x̃_mse⟩ = norm × Σᵢ (Πq)ᵢ × centroid[idx_i]
    float mse_dot = doc.norm *
        dot_mse_core(qs.rotated.data(), doc.mse_indices.data());

    if (!cfg_.use_prod)
        return mse_dot;

    // QJL contribution:
    //   ⟨q, x̃_qjl⟩ = α × residual_norm × Σᵢ (Sq)ᵢ × sign_i
    //
    // Scaling factor α: for a Hadamard-based QJL projection with
    // orthonormal rows, α = √(π/2).  This differs from the paper's
    // √(π/2)/d which assumes Gaussian S with rows of norm ≈ √d.
    // The Hadamard substitute is standard for JL projections in
    // practice; adjust α empirically if needed.
    static constexpr float kAlpha = 1.2533141373f; // √(π/2)

    float qjl_dot = kAlpha * doc.residual_norm *
        dot_qjl_core(qs.qjl_rotated.data(),
                      doc.qjl_signs.data(),
                      dim_padded_);

    return mse_dot + qjl_dot;
}

void TurboQuant::dot_asymmetric_batch(const QueryState& qs,
                                       const QuantizedVector* docs,
                                       uint32_t n,
                                       float* out) const {
    for (uint32_t i = 0; i < n; ++i)
        out[i] = dot_asymmetric(qs, docs[i]);
}

// ── ADC LUT scoring ────────────────────────────────────────────────────
//
// Uses precomputed tables lut[i * nc + j] = rotated_query[i] * centroid[j].
// The per-doc loop is pure gather + accumulate with no multiply.

static float dot_lut_4bit(const float* lut, const uint8_t* packed,
                          uint32_t n) {
    float sum = 0;
    uint32_t i = 0;
#if TQ_NEON
    float32x4_t acc0 = vdupq_n_f32(0);
    float32x4_t acc1 = vdupq_n_f32(0);
    for (; i + 8 <= n; i += 8) {
        uint8_t b0 = packed[i/2],   b1 = packed[i/2+1],
                b2 = packed[i/2+2], b3 = packed[i/2+3];
        // Each coordinate has its own 16-entry table at lut + i*16.
        float32x4_t v0 = {lut[(i+0)*16 + (b0 & 0xF)],
                          lut[(i+1)*16 + (b0 >> 4)],
                          lut[(i+2)*16 + (b1 & 0xF)],
                          lut[(i+3)*16 + (b1 >> 4)]};
        float32x4_t v1 = {lut[(i+4)*16 + (b2 & 0xF)],
                          lut[(i+5)*16 + (b2 >> 4)],
                          lut[(i+6)*16 + (b3 & 0xF)],
                          lut[(i+7)*16 + (b3 >> 4)]};
        acc0 = vaddq_f32(acc0, v0);
        acc1 = vaddq_f32(acc1, v1);
    }
    sum = vaddvq_f32(vaddq_f32(acc0, acc1));
#endif
    for (; i < n; i += 2) {
        uint8_t byte = packed[i / 2];
        sum += lut[i * 16 + (byte & 0xF)];
        if (i + 1 < n)
            sum += lut[(i+1) * 16 + (byte >> 4)];
    }
    return sum;
}

static float dot_lut_2bit(const float* lut, const uint8_t* packed,
                          uint32_t n) {
    float sum = 0;
    uint32_t i = 0;
#if TQ_NEON
    float32x4_t acc = vdupq_n_f32(0);
    for (; i + 4 <= n; i += 4) {
        uint8_t byte = packed[i / 4];
        float32x4_t v = {lut[(i+0)*4 + (byte & 3)],
                         lut[(i+1)*4 + ((byte>>2) & 3)],
                         lut[(i+2)*4 + ((byte>>4) & 3)],
                         lut[(i+3)*4 + ((byte>>6) & 3)]};
        acc = vaddq_f32(acc, v);
    }
    sum = vaddvq_f32(acc);
#endif
    for (; i < n; ++i) {
        uint32_t byte_idx = i / 4;
        uint32_t shift = (i % 4) * 2;
        uint8_t idx = (packed[byte_idx] >> shift) & 3;
        sum += lut[i * 4 + idx];
    }
    return sum;
}

static float dot_lut_1bit(const float* lut, const uint8_t* packed,
                          uint32_t n) {
    float sum = 0;
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t byte_idx = i / 8;
        uint32_t bit_pos = i % 8;
        uint8_t idx = (packed[byte_idx] >> bit_pos) & 1;
        sum += lut[i * 2 + idx];
    }
    return sum;
}

void TurboQuant::dot_asymmetric_lut_batch(const QueryState& qs,
                                           const QuantizedVector* docs,
                                           uint32_t n,
                                           float* out) const {
    uint32_t b = cfg_.use_prod ? cfg_.bit_width - 1 : cfg_.bit_width;
    uint32_t dp = dim_padded_;
    for (uint32_t d = 0; d < n; ++d) {
        float mse_dot;
        if (b == 4)
            mse_dot = dot_lut_4bit(qs.lut.data(), docs[d].mse_indices.data(), dp);
        else if (b == 2)
            mse_dot = dot_lut_2bit(qs.lut.data(), docs[d].mse_indices.data(), dp);
        else
            mse_dot = dot_lut_1bit(qs.lut.data(), docs[d].mse_indices.data(), dp);
        out[d] = docs[d].norm * mse_dot;
    }
}

} // namespace turbo_quant

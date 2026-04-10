#include "osq.h"

#include <algorithm>
#include <cstring>
#include <numeric>

namespace osq {

// ═══════════════════════════════════════════════════════════════════════════
//  Bit packing  (identical logic to turbo_quant — could be shared)
// ═══════════════════════════════════════════════════════════════════════════

namespace detail {

static void pack_1bit(const uint8_t* v, uint8_t* o, uint32_t n) {
    uint32_t full = n / 8;
    for (uint32_t i = 0; i < full; ++i) {
        const uint8_t* s = v + i * 8;
        o[i] = uint8_t(s[0]|(s[1]<<1)|(s[2]<<2)|(s[3]<<3)|
                       (s[4]<<4)|(s[5]<<5)|(s[6]<<6)|(s[7]<<7));
    }
    if (n % 8) {
        uint8_t byte = 0;
        for (uint32_t j = 0; j < n % 8; ++j)
            byte |= (v[full*8+j] & 1) << j;
        o[full] = byte;
    }
}

static void pack_2bit(const uint8_t* v, uint8_t* o, uint32_t n) {
    uint32_t full = n / 4;
    for (uint32_t i = 0; i < full; ++i) {
        const uint8_t* s = v + i * 4;
        o[i] = uint8_t(s[0]|(s[1]<<2)|(s[2]<<4)|(s[3]<<6));
    }
    if (n % 4) {
        uint8_t byte = 0;
        for (uint32_t j = 0; j < n % 4; ++j)
            byte |= (v[full*4+j] & 3) << (j*2);
        o[full] = byte;
    }
}

static void pack_4bit(const uint8_t* v, uint8_t* o, uint32_t n) {
    uint32_t full = n / 2;
    for (uint32_t i = 0; i < full; ++i)
        o[i] = uint8_t((v[2*i] & 0xF) | (v[2*i+1] << 4));
    if (n & 1) o[full] = v[n-1] & 0xF;
}

static void pack_general(const uint8_t* v, uint8_t* o,
                          uint32_t n, uint32_t bits) {
    std::memset(o, 0, (n * bits + 7) / 8);
    uint8_t mask = (1u << bits) - 1;
    uint32_t bp = 0;
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t bi = bp / 8, bo = bp % 8;
        o[bi] |= (v[i] & mask) << bo;
        if (bo + bits > 8) o[bi+1] |= v[i] >> (8 - bo);
        bp += bits;
    }
}

void pack_bits(const uint8_t* v, uint8_t* o, uint32_t n, uint32_t bits) {
    switch (bits) {
        case 1: pack_1bit(v, o, n); break;
        case 2: pack_2bit(v, o, n); break;
        case 4: pack_4bit(v, o, n); break;
        default: pack_general(v, o, n, bits); break;
    }
}

static void unpack_1bit(const uint8_t* p, uint8_t* o, uint32_t n) {
    for (uint32_t i = 0; i < n / 8; ++i) {
        uint8_t b = p[i]; uint8_t* d = o + i*8;
        d[0]=b&1; d[1]=(b>>1)&1; d[2]=(b>>2)&1; d[3]=(b>>3)&1;
        d[4]=(b>>4)&1; d[5]=(b>>5)&1; d[6]=(b>>6)&1; d[7]=(b>>7)&1;
    }
    if (n%8) { uint8_t b=p[n/8]; for(uint32_t j=0;j<n%8;++j) o[n/8*8+j]=(b>>j)&1; }
}

static void unpack_2bit(const uint8_t* p, uint8_t* o, uint32_t n) {
    for (uint32_t i = 0; i < n / 4; ++i) {
        uint8_t b = p[i]; uint8_t* d = o + i*4;
        d[0]=b&3; d[1]=(b>>2)&3; d[2]=(b>>4)&3; d[3]=(b>>6)&3;
    }
    if (n%4) { uint8_t b=p[n/4]; for(uint32_t j=0;j<n%4;++j) o[n/4*4+j]=(b>>(j*2))&3; }
}

static void unpack_4bit(const uint8_t* p, uint8_t* o, uint32_t n) {
    for (uint32_t i = 0; i < n / 2; ++i) {
        o[2*i] = p[i] & 0xF; o[2*i+1] = p[i] >> 4;
    }
    if (n & 1) o[n-1] = p[n/2] & 0xF;
}

static void unpack_general(const uint8_t* p, uint8_t* o,
                            uint32_t n, uint32_t bits) {
    uint8_t mask = (1u << bits) - 1;
    uint32_t bp = 0;
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t bi = bp/8, bo = bp%8;
        uint8_t v = p[bi] >> bo;
        if (bo + bits > 8) v |= p[bi+1] << (8-bo);
        o[i] = v & mask;
        bp += bits;
    }
}

void unpack_bits(const uint8_t* p, uint8_t* o, uint32_t n, uint32_t bits) {
    switch (bits) {
        case 1: unpack_1bit(p, o, n); break;
        case 2: unpack_2bit(p, o, n); break;
        case 4: unpack_4bit(p, o, n); break;
        default: unpack_general(p, o, n, bits); break;
    }
}

} // namespace detail

// ═══════════════════════════════════════════════════════════════════════════
//  OSQ construction
// ═══════════════════════════════════════════════════════════════════════════

constexpr float OSQ::kInitScale[];

OSQ::OSQ(const Config& cfg) : cfg_(cfg) {
    assert(cfg.bit_width >= 1 && cfg.bit_width <= kMaxBitWidth);
    assert(cfg.dim > 0);
    max_idx_ = (1u << cfg.bit_width) - 1;
    centroid_.assign(cfg.dim, 0.0f);
    centroid_sq_ = 0.0f;
}

void OSQ::set_centroid(const float* centroid) {
    uint32_t n = cfg_.dim;
    centroid_.assign(centroid, centroid + n);
    centroid_sq_ = 0.0f;
#if OSQ_NEON
    uint32_t i = 0;
    float32x4_t acc = vdupq_n_f32(0);
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(centroid + i);
        acc = vmlaq_f32(acc, v, v);
    }
    centroid_sq_ = vaddvq_f32(acc);
    for (; i < n; ++i) centroid_sq_ += centroid[i] * centroid[i];
#else
    for (uint32_t i = 0; i < n; ++i) centroid_sq_ += centroid[i] * centroid[i];
#endif
}

// ═══════════════════════════════════════════════════════════════════════════
//  Interval initialisation and refinement
// ═══════════════════════════════════════════════════════════════════════════

void OSQ::init_interval(const float* x, uint32_t n,
                        float& a, float& b) const {
    // Compute per-vector mean and stddev.
    float sum = 0, sum2 = 0;
#if OSQ_NEON
    uint32_t i = 0;
    float32x4_t vs = vdupq_n_f32(0), vs2 = vdupq_n_f32(0);
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        vs  = vaddq_f32(vs, v);
        vs2 = vmlaq_f32(vs2, v, v);
    }
    sum  = vaddvq_f32(vs);
    sum2 = vaddvq_f32(vs2);
    for (; i < n; ++i) { sum += x[i]; sum2 += x[i] * x[i]; }
#else
    for (uint32_t i = 0; i < n; ++i) { sum += x[i]; sum2 += x[i] * x[i]; }
#endif

    float mean = sum / n;
    float var  = sum2 / n - mean * mean;
    float sigma = std::sqrt(std::max(var, 1e-12f));

    float k = kInitScale[cfg_.bit_width];
    a = mean - k * sigma;
    b = mean + k * sigma;
}

float OSQ::quantization_error(const float* x, uint32_t n,
                               float xx, float a, float b) const {
    // Compute L = (1-λ)(x·e)²/‖x‖² + λ‖e‖²
    // where e_i = x_i − x̃_i  and  x̃_i = a + idx_i·Δ
    float delta = (b - a) / static_cast<float>(max_idx_);
    if (std::abs(delta) < 1e-15f) return 0.0f;
    float inv_delta = 1.0f / delta;

    float xe = 0.0f;  // x · e
    float ee = 0.0f;  // e · e
    for (uint32_t i = 0; i < n; ++i) {
        float tf = (x[i] - a) * inv_delta;
        int idx = static_cast<int>(tf + 0.5f);
        idx = idx < 0 ? 0 : (idx > static_cast<int>(max_idx_)
                              ? static_cast<int>(max_idx_) : idx);
        float xiq = a + static_cast<float>(idx) * delta;
        float ei = x[i] - xiq;
        xe += x[i] * ei;
        ee += ei * ei;
    }

    float lambda = cfg_.lambda;
    return (1.0f - lambda) * xe * xe / xx + lambda * ee;
}

void OSQ::refine_interval(const float* x, uint32_t n,
                          float& a, float& b) const {
    // Iterative optimisation of the anisotropic error:
    //
    //   L(a,b) = (1−λ) · (x·e)² / ‖x‖²  +  λ · ‖e‖²
    //
    // where e = x − x̃ is the reconstruction error.  The first term
    // measures the squared dot-product error projected onto the document
    // vector — the quantity that actually degrades ranking.  The λ term
    // regularises with MSE to prevent overfitting to one direction.
    //
    // Each iteration:
    //   1. Quantise components to the nearest grid point of [a, b],
    //   2. Solve for (a, b) that minimise L with fixed assignments.
    //
    // Unlike k-means the error is not guaranteed to decrease at each
    // step because rounding to the nearest grid point is not the
    // snapping which minimises the anisotropic error.  Once it starts
    // increasing it nearly always continues to do so, so we stop as
    // soon as we see an increase.

    float inv_M = 1.0f / static_cast<float>(max_idx_);
    float lambda = cfg_.lambda;

    // Precompute ‖x‖².
    float xx = 0.0f;
    for (uint32_t i = 0; i < n; ++i) xx += x[i] * x[i];
    if (xx < 1e-30f) return;

    float scale = (1.0f - lambda) / xx;

    float error = quantization_error(x, n, xx, a, b);

    for (uint32_t iter = 0; iter < cfg_.refine_iters; ++iter) {
        float delta = (b - a) * inv_M;
        if (std::abs(delta) < 1e-15f) break;
        float inv_delta = 1.0f / delta;

        // ── Quantise and accumulate sufficient statistics ────────────────
        //
        // Reconstruction: x̃_i = a·(1−s_i) + b·s_i  where s_i = idx_i/max_idx.
        // The sufficient statistics for the 2×2 system are:
        //   dax = Σ x_i·(1−s_i),  dbx = Σ x_i·s_i
        //   daa = Σ (1−s_i)²,     dab = Σ (1−s_i)·s_i,  dbb = Σ s_i²

        double daa = 0, dab = 0, dbb = 0;
        double dax = 0, dbx = 0;

        for (uint32_t i = 0; i < n; ++i) {
            float tf = (std::clamp(x[i], a, b) - a) * inv_delta;
            int idx = static_cast<int>(tf + 0.5f);
            float s = static_cast<float>(idx) * inv_M;
            float u = 1.0f - s;  // coefficient of a

            daa += u * u;
            dab += u * s;
            dbb += s * s;
            dax += x[i] * u;
            dbx += x[i] * s;
        }

        // ── Solve the stationary equations of the anisotropic loss ──────
        //
        // Setting ∂L/∂a = 0 and ∂L/∂b = 0 gives:
        //
        //   ⎡ scale·dax² + λ·daa    scale·dax·dbx + λ·dab ⎤ ⎡a⎤   ⎡dax⎤
        //   ⎣ scale·dax·dbx + λ·dab  scale·dbx² + λ·dbb   ⎦ ⎣b⎦ = ⎣dbx⎦

        double M00 = scale * dax * dax + lambda * daa;
        double M01 = scale * dax * dbx + lambda * dab;
        double M11 = scale * dbx * dbx + lambda * dbb;

        double det = M00 * M11 - M01 * M01;
        if (std::abs(det) < 1e-30) break;
        double inv_det = 1.0 / det;

        float new_a = static_cast<float>((M11 * dax - M01 * dbx) * inv_det);
        float new_b = static_cast<float>((M00 * dbx - M01 * dax) * inv_det);

        if (new_a >= new_b) break;

        // ── Monotonicity check ──────────────────────────────────────────
        float new_error = quantization_error(x, n, xx, new_a, new_b);
        if (new_error > error) break;  // revert — error is increasing

        // Converged?
        float change = std::abs(new_a - a) + std::abs(new_b - b);

        a = new_a;
        b = new_b;
        error = new_error;

        if (change < 1e-8f) break;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Quantise / dequantise
// ═══════════════════════════════════════════════════════════════════════════

OSQVector OSQ::quantize(const float* x) const {
    uint32_t n = cfg_.dim;
    OSQVector qv;
    qv.dim = n;
    qv.bit_width = cfg_.bit_width;

    // ── Step 1: compute ⟨c, x⟩ and centre: x' = x − c ─────────────────
    std::vector<float> xc(n);
    float cx = 0;
    for (uint32_t i = 0; i < n; ++i) {
        cx += centroid_[i] * x[i];
        xc[i] = x[i] - centroid_[i];
    }
    qv.cx = cx;

    const float* xp = xc.data();
    qv.norm = 1.0f;

    // ── Step 2: initialise and refine [a, b] on centred data ────────────
    init_interval(xp, n, qv.a, qv.b);
    if (cfg_.refine_iters > 0)
        refine_interval(xp, n, qv.a, qv.b);

    qv.delta = (qv.b - qv.a) / static_cast<float>(max_idx_);

    // ── Step 3: final quantisation pass ─────────────────────────────────
    float inv_delta = (std::abs(qv.delta) > 1e-15f)
                      ? 1.0f / qv.delta : 0.0f;
    std::vector<uint8_t> indices(n);
    float sum_idx = 0;
    uint32_t mi = max_idx_;

    for (uint32_t i = 0; i < n; ++i) {
        float tf = (xp[i] - qv.a) * inv_delta;
        int idx = static_cast<int>(tf + 0.5f);
        idx = idx < 0 ? 0 : (idx > static_cast<int>(mi) ? static_cast<int>(mi) : idx);
        indices[i] = static_cast<uint8_t>(idx);
        sum_idx += static_cast<float>(idx);
    }

    qv.sum_indices = sum_idx;
    qv.packed.resize(qv.packed_bytes());
    detail::pack_bits(indices.data(), qv.packed.data(), n, cfg_.bit_width);

    // ── Step 4: precompute bit-planes ────────────────────────────────────
    // Decompose b-bit indices into b separate 1-bit packed planes.
    // Enables integer dot products via AND+popcount (RaBitQ style).
    //
    // 2-bit: 2 planes → Σ x_i·y_i = 4·pc(x1&y1) + 2·pc(x1&y0 + x0&y1) + pc(x0&y0)
    // 4-bit: 4 planes → used in mixed 4×1 kernel.
    if (cfg_.bit_width == 2) {
        uint32_t n1_bytes = (n + 7) / 8;
        uint32_t n2_bytes = (n + 3) / 4;  // bytes of 2-bit packed data
        qv.bit_planes[0].assign(n1_bytes, 0);
        qv.bit_planes[1].assign(n1_bytes, 0);

        // 2-bit packing: byte B holds 4 crumbs at bit positions
        // [0:1], [2:3], [4:5], [6:7] for indices 4j, 4j+1, 4j+2, 4j+3.
        // Group of 8 indices [8g..8g+7] spans 2 packed bytes [2g, 2g+1].
        const uint8_t* p2 = qv.packed.data();
        for (uint32_t g = 0; g < n1_bytes; ++g) {
            uint8_t B0 = (2*g     < n2_bytes) ? p2[2*g]     : 0;
            uint8_t B1 = (2*g + 1 < n2_bytes) ? p2[2*g + 1] : 0;
            // Extract even bits (bit 0 of each crumb) → plane 0.
            // Extract odd bits (bit 1 of each crumb) → plane 1.
            auto extract_even = [](uint8_t b) -> uint8_t {
                return ((b >> 0) & 1) | ((b >> 1) & 2) |
                       ((b >> 2) & 4) | ((b >> 3) & 8);
            };
            qv.bit_planes[0][g] = extract_even(B0) | (extract_even(B1) << 4);
            qv.bit_planes[1][g] = extract_even(B0 >> 1) | (extract_even(B1 >> 1) << 4);
        }
    } else if (cfg_.bit_width == 4) {
        uint32_t n1_bytes = (n + 7) / 8;
        uint32_t n4_bytes = (n + 1) / 2;
        for (uint32_t k = 0; k < 4; ++k)
            qv.bit_planes[k].assign(n1_bytes, 0);

        const uint8_t* p4 = qv.packed.data();
        for (uint32_t g = 0; g < n1_bytes; ++g) {
            uint32_t remain = std::min(4u, n4_bytes - g * 4);
            uint8_t B[4] = {0, 0, 0, 0};
            for (uint32_t j = 0; j < remain; ++j)
                B[j] = p4[g * 4 + j];
            uint8_t p0 = 0, p1 = 0, p2 = 0, p3 = 0;
            for (uint32_t j = 0; j < 4; ++j) {
                uint8_t lo = B[j];
                uint8_t hi = B[j] >> 4;
                p0 |= ((lo     ) & 1) << (2*j) | ((hi     ) & 1) << (2*j+1);
                p1 |= ((lo >> 1) & 1) << (2*j) | ((hi >> 1) & 1) << (2*j+1);
                p2 |= ((lo >> 2) & 1) << (2*j) | ((hi >> 2) & 1) << (2*j+1);
                p3 |= ((lo >> 3) & 1) << (2*j) | ((hi >> 3) & 1) << (2*j+1);
            }
            qv.bit_planes[0][g] = p0;
            qv.bit_planes[1][g] = p1;
            qv.bit_planes[2][g] = p2;
            qv.bit_planes[3][g] = p3;
        }
    }

    return qv;
}

void OSQ::dequantize(const OSQVector& qv, float* out) const {
    uint32_t n = qv.dim;
    std::vector<uint8_t> indices(n);
    detail::unpack_bits(qv.packed.data(), indices.data(), n, qv.bit_width);

    // Reconstruct: out = centroid + norm * (a + idx * Δ)
    for (uint32_t i = 0; i < n; ++i)
        out[i] = centroid_[i] +
                 qv.norm * (qv.a + static_cast<float>(indices[i]) * qv.delta);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Query preparation
// ═══════════════════════════════════════════════════════════════════════════

QueryState OSQ::prepare_query(const float* q) const {
    QueryState qs;
    uint32_t n = cfg_.dim;

    // Store the CENTRED query q' = q − c.  The asymmetric dot product
    // decomposition is:
    //
    //   ⟨q, x⟩ = ⟨q, c⟩ + (⟨c, x⟩ − ‖c‖²) + ⟨q − c, x̃'⟩
    //
    // where x̃' is the quantised centred residual.  The middle term uses
    // the exact precomputed ⟨c, x⟩ (no quantization error).  Centering
    // q means the quantization error e in x̃' is multiplied by (q_i − c_i)
    // rather than q_i, which dramatically reduces noise when the data
    // has a non-zero mean.
    qs.query.resize(n);

    float qc = 0, sum = 0;
#if OSQ_NEON
    uint32_t i = 0;
    float32x4_t vqc = vdupq_n_f32(0), vs = vdupq_n_f32(0);
    for (; i + 4 <= n; i += 4) {
        float32x4_t vq = vld1q_f32(q + i);
        float32x4_t vc = vld1q_f32(centroid_.data() + i);
        vqc = vmlaq_f32(vqc, vq, vc);
        float32x4_t vqc_diff = vsubq_f32(vq, vc);
        vst1q_f32(qs.query.data() + i, vqc_diff);
        vs = vaddq_f32(vs, vqc_diff);
    }
    qc = vaddvq_f32(vqc);
    sum = vaddvq_f32(vs);
    for (; i < n; ++i) {
        qc += q[i] * centroid_[i];
        qs.query[i] = q[i] - centroid_[i];
        sum += qs.query[i];
    }
#else
    for (uint32_t i = 0; i < n; ++i) {
        qc += q[i] * centroid_[i];
        qs.query[i] = q[i] - centroid_[i];
        sum += qs.query[i];
    }
#endif
    qs.qc = qc;
    qs.query_sum = sum;

    return qs;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Core kernels
// ═══════════════════════════════════════════════════════════════════════════

// ── float × int dot:  Σ q_i · idx_i ────────────────────────────────────
//
// This is the hot loop for asymmetric distance.  We unpack integer
// indices, widen to float32, and FMA with the query.  No lookup table
// needed — the uniform grid means the index IS the value (up to
// linear scaling).

float OSQ::float_int_dot(const float* q,
                          const uint8_t* packed, uint32_t n,
                          uint32_t bits) const {
    float sum = 0;

    // ── 4-bit ───────────────────────────────────────────────────────────
    if (bits == 4) {
#if OSQ_NEON
        uint32_t i = 0;
        float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
        uint8x8_t mask_0f = vdup_n_u8(0x0F);

        // Process 8 indices (4 packed bytes) per iteration.
        for (; i + 8 <= n; i += 8) {
            uint8x8_t raw = vld1_u8(packed + i / 2);

            // Unpack nibbles and interleave to original order.
            uint8x8_t lo = vand_u8(raw, mask_0f);
            uint8x8_t hi = vshr_n_u8(raw, 4);
            uint8x8x2_t z = vzip_u8(lo, hi);
            uint8x8_t indices = z.val[0];

            // Widen to float32: u8 → u16 → u32 → f32.
            uint16x8_t w16 = vmovl_u8(indices);
            uint32x4_t w32_lo = vmovl_u16(vget_low_u16(w16));
            uint32x4_t w32_hi = vmovl_u16(vget_high_u16(w16));
            float32x4_t f_lo = vcvtq_f32_u32(w32_lo);
            float32x4_t f_hi = vcvtq_f32_u32(w32_hi);

            acc0 = vmlaq_f32(acc0, vld1q_f32(q + i),     f_lo);
            acc1 = vmlaq_f32(acc1, vld1q_f32(q + i + 4), f_hi);
        }
        sum = vaddvq_f32(vaddq_f32(acc0, acc1));
        for (; i < n; i += 2) {
            uint8_t byte = packed[i / 2];
            sum += q[i] * static_cast<float>(byte & 0xF);
            if (i + 1 < n)
                sum += q[i + 1] * static_cast<float>(byte >> 4);
        }
#else
        for (uint32_t i = 0; i < n; i += 2) {
            uint8_t byte = packed[i / 2];
            sum += q[i] * static_cast<float>(byte & 0xF);
            if (i + 1 < n)
                sum += q[i + 1] * static_cast<float>(byte >> 4);
        }
#endif
        return sum;
    }

    // ── 2-bit ───────────────────────────────────────────────────────────
    if (bits == 2) {
#if OSQ_NEON
        uint32_t i = 0;
        float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
        float32x4_t acc2 = vdupq_n_f32(0), acc3 = vdupq_n_f32(0);

        // Process 16 indices (4 packed bytes) per iteration.
        for (; i + 16 <= n; i += 16) {
            // Load 4 bytes, each containing 4 crumbs.  Unpack to
            // 4 × uint32x4 via scalar extraction.
            uint8_t bytes[4];
            std::memcpy(bytes, packed + i / 4, 4);
            uint32x4_t i0 = {bytes[0]&3u, (bytes[0]>>2)&3u,
                             (bytes[0]>>4)&3u, (uint32_t)(bytes[0]>>6)};
            uint32x4_t i1 = {bytes[1]&3u, (bytes[1]>>2)&3u,
                             (bytes[1]>>4)&3u, (uint32_t)(bytes[1]>>6)};
            uint32x4_t i2 = {bytes[2]&3u, (bytes[2]>>2)&3u,
                             (bytes[2]>>4)&3u, (uint32_t)(bytes[2]>>6)};
            uint32x4_t i3 = {bytes[3]&3u, (bytes[3]>>2)&3u,
                             (bytes[3]>>4)&3u, (uint32_t)(bytes[3]>>6)};
            acc0 = vmlaq_f32(acc0, vld1q_f32(q+i),    vcvtq_f32_u32(i0));
            acc1 = vmlaq_f32(acc1, vld1q_f32(q+i+4),  vcvtq_f32_u32(i1));
            acc2 = vmlaq_f32(acc2, vld1q_f32(q+i+8),  vcvtq_f32_u32(i2));
            acc3 = vmlaq_f32(acc3, vld1q_f32(q+i+12), vcvtq_f32_u32(i3));
        }
        sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0,acc1),vaddq_f32(acc2,acc3)));
        for (; i < n; ++i) {
            uint8_t idx = (packed[i/4] >> ((i%4)*2)) & 3;
            sum += q[i] * static_cast<float>(idx);
        }
#else
        for (uint32_t i = 0; i < n; ++i) {
            uint8_t idx = (packed[i/4] >> ((i%4)*2)) & 3;
            sum += q[i] * static_cast<float>(idx);
        }
#endif
        return sum;
    }

    // ── 1-bit ───────────────────────────────────────────────────────────
    //  Σ(q_i · idx_i) where idx ∈ {0, 1}  =  Σ q_i where idx == 1
    //
    // Expand each bit to a 32-bit mask via integer compare, then AND
    // with the float query lane.  Quad accumulators (2 bytes = 16
    // elements per iteration) for pipeline parallelism.
    if (bits == 1) {
#if OSQ_NEON
        uint32_t i = 0;
        float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
        float32x4_t acc2 = vdupq_n_f32(0), acc3 = vdupq_n_f32(0);
        uint32x4_t zero = vdupq_n_u32(0);

        for (; i + 16 <= n; i += 16) {
            uint8_t b0 = packed[i / 8];
            uint8_t b1 = packed[i / 8 + 1];

            // First byte → 8 elements via 2 × float32x4.
            uint32x4_t m0 = vcgtq_u32(
                (uint32x4_t){b0&1u, b0&2u, b0&4u, b0&8u}, zero);
            uint32x4_t m1 = vcgtq_u32(
                (uint32x4_t){b0&16u, b0&32u, b0&64u, b0&128u}, zero);

            // Second byte → 8 elements.
            uint32x4_t m2 = vcgtq_u32(
                (uint32x4_t){b1&1u, b1&2u, b1&4u, b1&8u}, zero);
            uint32x4_t m3 = vcgtq_u32(
                (uint32x4_t){b1&16u, b1&32u, b1&64u, b1&128u}, zero);

            acc0 = vaddq_f32(acc0, vreinterpretq_f32_u32(vandq_u32(
                vreinterpretq_u32_f32(vld1q_f32(q+i)),    m0)));
            acc1 = vaddq_f32(acc1, vreinterpretq_f32_u32(vandq_u32(
                vreinterpretq_u32_f32(vld1q_f32(q+i+4)),  m1)));
            acc2 = vaddq_f32(acc2, vreinterpretq_f32_u32(vandq_u32(
                vreinterpretq_u32_f32(vld1q_f32(q+i+8)),  m2)));
            acc3 = vaddq_f32(acc3, vreinterpretq_f32_u32(vandq_u32(
                vreinterpretq_u32_f32(vld1q_f32(q+i+12)), m3)));
        }
        // 8-element tail.
        for (; i + 8 <= n; i += 8) {
            uint8_t byte = packed[i / 8];
            uint32x4_t ml = vcgtq_u32(
                (uint32x4_t){byte&1u, byte&2u, byte&4u, byte&8u}, zero);
            uint32x4_t mh = vcgtq_u32(
                (uint32x4_t){byte&16u, byte&32u, byte&64u, byte&128u}, zero);
            acc0 = vaddq_f32(acc0, vreinterpretq_f32_u32(vandq_u32(
                vreinterpretq_u32_f32(vld1q_f32(q+i)),   ml)));
            acc1 = vaddq_f32(acc1, vreinterpretq_f32_u32(vandq_u32(
                vreinterpretq_u32_f32(vld1q_f32(q+i+4)), mh)));
        }
        sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0,acc1),
                                    vaddq_f32(acc2,acc3)));
        for (; i < n; ++i)
            if ((packed[i/8] >> (i%8)) & 1) sum += q[i];
#else
        for (uint32_t i = 0; i < n; ++i)
            if ((packed[i/8] >> (i%8)) & 1) sum += q[i];
#endif
        return sum;
    }

    // ── 3-bit (general) ────────────────────────────────────────────────
    {
        std::vector<uint8_t> indices(n);
        detail::unpack_bits(packed, indices.data(), n, bits);
        for (uint32_t i = 0; i < n; ++i)
            sum += q[i] * static_cast<float>(indices[i]);
        return sum;
    }
}

// ── int × int dot:  Σ idx_x · idx_y ────────────────────────────────────
//
// THE payoff for uniform grids.  This is a pure integer dot product
// on small (1–4 bit) values.  On NEON, 4-bit indices are multiplied
// as uint8 nibbles with pairwise accumulation into uint32 — no float
// conversion, no lookup table, no gather.
//
// Supports mixed bit widths: when bits_x != bits_y, falls through to
// a general unpack-and-multiply path.

uint64_t OSQ::int_int_dot(const uint8_t* px, uint32_t bits_x,
                           const uint8_t* py, uint32_t bits_y,
                           uint32_t n) const {
    // ── Mixed bit widths ────────────────────────────────────────────────
    //
    // Specialised path for 4-bit × 1-bit (the production pattern).
    //
    // Key insight (à la RaBitQ): since idx1 ∈ {0,1}, the dot product is
    // just "sum the 4-bit values wherever the 1-bit flag is set".  We
    // decompose the 4-bit nibbles into 4 bit-planes and reduce each to
    // a popcount:
    //
    //   Σ idx4_i × idx1_i = 8·popcount(plane3 & bits1)
    //                     + 4·popcount(plane2 & bits1)
    //                     + 2·popcount(plane1 & bits1)
    //                     +   popcount(plane0 & bits1)
    //
    // The bit-plane extraction requires re-interleaving nibble bits to
    // match the 1-bit packing layout.  On NEON we use vcntq_u8 for the
    // popcount.
    if (bits_x != bits_y) {
        // Canonicalise: p_hi is the higher-bit-width side.
        const uint8_t* p_hi = (bits_x >= bits_y) ? px : py;
        const uint8_t* p_lo = (bits_x >= bits_y) ? py : px;
        uint32_t b_hi = std::max(bits_x, bits_y);
        uint32_t b_lo = std::min(bits_x, bits_y);

        if (b_hi == 4 && b_lo == 1) {
            // p_hi: 4-bit packed (2 nibbles/byte), p_lo: 1-bit packed (8 bits/byte).
            // For dim=768: p_hi=384 bytes, p_lo=96 bytes.
            //
            // Bit-plane decomposition (à la RaBitQ):
            //
            //   Σ idx4_i × idx1_i = 8·popcount(plane3 & bits1)
            //                     + 4·popcount(plane2 & bits1)
            //                     + 2·popcount(plane1 & bits1)
            //                     +   popcount(plane0 & bits1)
            //
            // We precompute 4 bit-planes from the 4-bit data, each in
            // the same 1-bit packed layout as p_lo.  Then each plane
            // reduces to the same AND+popcount as the 1-1 kernel (~7ns).
            //
            // Bit-plane extraction: for 4-bit packed byte B containing
            // nibbles [lo=B&0xF, hi=B>>4] (indices 2j, 2j+1), plane k
            // contributes bit k of each nibble.  For a group of 4 bytes
            // (8 indices), plane k output byte =
            //   ((B0>>k)&1) | (((B0>>(k+4))&1)<<1) |
            //   ((B1>>k)&1)<<2 | (((B1>>(k+4))&1)<<3) | ...

            uint32_t n1_bytes = (n + 7) / 8;  // bytes of 1-bit data
            uint32_t n4_bytes = (n + 1) / 2;  // bytes of 4-bit data

            // Use precomputed bit-planes from the 4-bit OSQVector.
            // The caller (dot_symmetric) passes the raw packed pointers;
            // we look up the planes from the OSQVector via context.
            // However, int_int_dot only receives raw pointers, so we
            // use an alternative: pass planes through a member or
            // restructure.  For now, compute planes on the fly but
            // using stack allocation (96 bytes each for dim=768).
            //
            // NOTE: In production, planes would be precomputed at
            // quantize time (stored in OSQVector::bit_planes) and
            // the kernel restructured to accept them.  Here we check
            // if p_hi matches a known vector's packed data, but the
            // simplest correct approach is to use stack buffers and
            // avoid heap allocation.

            uint8_t plane_buf[4][128];  // max 1024 dims → 128 bytes each
            assert(n1_bytes <= 128);
            std::memset(plane_buf, 0, sizeof(plane_buf));

            for (uint32_t g = 0; g < n1_bytes; ++g) {
                uint32_t remain = std::min(4u, n4_bytes - g * 4);
                uint8_t B[4] = {0, 0, 0, 0};
                for (uint32_t j = 0; j < remain; ++j)
                    B[j] = p_hi[g * 4 + j];
                uint8_t p0 = 0, p1 = 0, p2 = 0, p3 = 0;
                for (uint32_t j = 0; j < 4; ++j) {
                    uint8_t lo = B[j];
                    uint8_t hi = B[j] >> 4;
                    p0 |= ((lo     ) & 1) << (2*j) | ((hi     ) & 1) << (2*j+1);
                    p1 |= ((lo >> 1) & 1) << (2*j) | ((hi >> 1) & 1) << (2*j+1);
                    p2 |= ((lo >> 2) & 1) << (2*j) | ((hi >> 2) & 1) << (2*j+1);
                    p3 |= ((lo >> 3) & 1) << (2*j) | ((hi >> 3) & 1) << (2*j+1);
                }
                plane_buf[0][g] = p0;
                plane_buf[1][g] = p1;
                plane_buf[2][g] = p2;
                plane_buf[3][g] = p3;
            }

            // total = 1·popcount(plane0 & p_lo) + 2·popcount(plane1 & p_lo)
            //       + 4·popcount(plane2 & p_lo) + 8·popcount(plane3 & p_lo)

            uint64_t total = 0;

#if OSQ_NEON
            auto popcount_and = [](const uint8_t* pa, const uint8_t* pb,
                                   uint32_t nbytes) -> uint64_t {
                uint32_t i = 0;
                uint32x4_t acc0 = vdupq_n_u32(0);
                uint32x4_t acc1 = vdupq_n_u32(0);
                for (; i + 32 <= nbytes; i += 32) {
                    uint8x16_t a0 = vld1q_u8(pa + i);
                    uint8x16_t b0 = vld1q_u8(pb + i);
                    uint8x16_t pc0 = vcntq_u8(vandq_u8(a0, b0));
                    uint16x8_t w0 = vpaddlq_u8(pc0);
                    acc0 = vpadalq_u16(acc0, w0);
                    uint8x16_t a1 = vld1q_u8(pa + i + 16);
                    uint8x16_t b1 = vld1q_u8(pb + i + 16);
                    uint8x16_t pc1 = vcntq_u8(vandq_u8(a1, b1));
                    uint16x8_t w1 = vpaddlq_u8(pc1);
                    acc1 = vpadalq_u16(acc1, w1);
                }
                for (; i + 16 <= nbytes; i += 16) {
                    uint8x16_t a0 = vld1q_u8(pa + i);
                    uint8x16_t b0 = vld1q_u8(pb + i);
                    uint8x16_t pc0 = vcntq_u8(vandq_u8(a0, b0));
                    uint16x8_t w0 = vpaddlq_u8(pc0);
                    acc0 = vpadalq_u16(acc0, w0);
                }
                uint64_t s = vaddvq_u32(vaddq_u32(acc0, acc1));
                for (; i < nbytes; ++i)
                    s += __builtin_popcount(pa[i] & pb[i]);
                return s;
            };

            total = 1 * popcount_and(plane_buf[0], p_lo, n1_bytes)
                  + 2 * popcount_and(plane_buf[1], p_lo, n1_bytes)
                  + 4 * popcount_and(plane_buf[2], p_lo, n1_bytes)
                  + 8 * popcount_and(plane_buf[3], p_lo, n1_bytes);
#else
            // Scalar: iterate 1-bit bytes, gather matching 4-bit nibbles.
            for (uint32_t g = 0; g < n1_bytes; ++g) {
                uint8_t mask_byte = p_lo[g];
                if (mask_byte == 0) continue;
                const uint8_t* p4g = p_hi + g * 4;
                uint8_t B0 = p4g[0], B1 = p4g[1], B2 = p4g[2], B3 = p4g[3];
                uint8_t idx[8] = {
                    uint8_t(B0 & 0xF), uint8_t(B0 >> 4),
                    uint8_t(B1 & 0xF), uint8_t(B1 >> 4),
                    uint8_t(B2 & 0xF), uint8_t(B2 >> 4),
                    uint8_t(B3 & 0xF), uint8_t(B3 >> 4)
                };
                for (uint32_t b = 0; b < 8 && (g * 8 + b) < n; ++b)
                    if ((mask_byte >> b) & 1)
                        total += idx[b];
            }
#endif
            return total;
        }

        // General mixed-width fallback: unpack and scalar multiply.
        std::vector<uint8_t> ax(n), bx(n);
        detail::unpack_bits(px, ax.data(), n, bits_x);
        detail::unpack_bits(py, bx.data(), n, bits_y);
        uint64_t sum = 0;
        for (uint32_t i = 0; i < n; ++i)
            sum += static_cast<uint64_t>(ax[i]) * bx[i];
        return sum;
    }

    uint64_t sum = 0;

    // ── 4-bit: nibble multiply with NEON ────────────────────────────────
    if (bits_x == 4) {
        uint32_t nbytes = (n + 1) / 2;
#if OSQ_NEON
        uint32_t i = 0;
        // Dual accumulators for pipeline parallelism.
        // Process 32 packed bytes (64 indices) per iteration.
        uint32x4_t acc0 = vdupq_n_u32(0);
        uint32x4_t acc1 = vdupq_n_u32(0);
        uint8x16_t mask = vdupq_n_u8(0x0F);

        for (; i + 32 <= nbytes; i += 32) {
            // First 16 bytes.
            uint8x16_t a0_raw = vld1q_u8(px + i);
            uint8x16_t b0_raw = vld1q_u8(py + i);
            uint8x16_t prod0_lo = vmulq_u8(vandq_u8(a0_raw, mask),
                                            vandq_u8(b0_raw, mask));
            uint8x16_t prod0_hi = vmulq_u8(vshrq_n_u8(a0_raw, 4),
                                            vshrq_n_u8(b0_raw, 4));
            uint16x8_t s0 = vpaddlq_u8(prod0_lo);
            s0 = vpadalq_u8(s0, prod0_hi);
            acc0 = vpadalq_u16(acc0, s0);

            // Second 16 bytes — independent chain.
            uint8x16_t a1_raw = vld1q_u8(px + i + 16);
            uint8x16_t b1_raw = vld1q_u8(py + i + 16);
            uint8x16_t prod1_lo = vmulq_u8(vandq_u8(a1_raw, mask),
                                            vandq_u8(b1_raw, mask));
            uint8x16_t prod1_hi = vmulq_u8(vshrq_n_u8(a1_raw, 4),
                                            vshrq_n_u8(b1_raw, 4));
            uint16x8_t s1 = vpaddlq_u8(prod1_lo);
            s1 = vpadalq_u8(s1, prod1_hi);
            acc1 = vpadalq_u16(acc1, s1);
        }
        // Single-register tail.
        for (; i + 16 <= nbytes; i += 16) {
            uint8x16_t a_raw = vld1q_u8(px + i);
            uint8x16_t b_raw = vld1q_u8(py + i);
            uint8x16_t prod_lo = vmulq_u8(vandq_u8(a_raw, mask),
                                           vandq_u8(b_raw, mask));
            uint8x16_t prod_hi = vmulq_u8(vshrq_n_u8(a_raw, 4),
                                           vshrq_n_u8(b_raw, 4));
            uint16x8_t s = vpaddlq_u8(prod_lo);
            s = vpadalq_u8(s, prod_hi);
            acc0 = vpadalq_u16(acc0, s);
        }
        sum = vaddvq_u32(vaddq_u32(acc0, acc1));
        // Scalar tail.
        for (; i < nbytes; ++i) {
            uint8_t a_lo = px[i] & 0xF, a_hi = px[i] >> 4;
            uint8_t b_lo = py[i] & 0xF, b_hi = py[i] >> 4;
            sum += static_cast<uint64_t>(a_lo) * b_lo;
            if (2 * i + 1 < n)
                sum += static_cast<uint64_t>(a_hi) * b_hi;
        }
#else
        for (uint32_t i = 0; i < nbytes; ++i) {
            uint8_t a_lo = px[i] & 0xF, a_hi = px[i] >> 4;
            uint8_t b_lo = py[i] & 0xF, b_hi = py[i] >> 4;
            sum += static_cast<uint64_t>(a_lo) * b_lo;
            if (2 * i + 1 < n)
                sum += static_cast<uint64_t>(a_hi) * b_hi;
        }
#endif
        return sum;
    }

    // ── 2-bit ───────────────────────────────────────────────────────────
    if (bits_x == 2) {
        uint32_t nbytes = (n + 3) / 4;
#if OSQ_NEON
        uint32_t i = 0;
        uint32x4_t acc0 = vdupq_n_u32(0);
        uint32x4_t acc1 = vdupq_n_u32(0);
        uint8x16_t mask2 = vdupq_n_u8(0x03);

        for (; i + 32 <= nbytes; i += 32) {
            // First 16 bytes — 64 crumbs.
            uint8x16_t a_raw = vld1q_u8(px + i);
            uint8x16_t b_raw = vld1q_u8(py + i);
            uint8x16_t p0 = vmulq_u8(vandq_u8(a_raw, mask2),
                                      vandq_u8(b_raw, mask2));
            uint8x16_t p1 = vmulq_u8(vandq_u8(vshrq_n_u8(a_raw, 2), mask2),
                                      vandq_u8(vshrq_n_u8(b_raw, 2), mask2));
            uint8x16_t p2 = vmulq_u8(vandq_u8(vshrq_n_u8(a_raw, 4), mask2),
                                      vandq_u8(vshrq_n_u8(b_raw, 4), mask2));
            uint8x16_t p3 = vmulq_u8(vshrq_n_u8(a_raw, 6),
                                      vshrq_n_u8(b_raw, 6));
            uint16x8_t s0 = vpaddlq_u8(p0);
            s0 = vpadalq_u8(s0, p1);
            s0 = vpadalq_u8(s0, p2);
            s0 = vpadalq_u8(s0, p3);
            acc0 = vpadalq_u16(acc0, s0);

            // Second 16 bytes — independent chain.
            a_raw = vld1q_u8(px + i + 16);
            b_raw = vld1q_u8(py + i + 16);
            p0 = vmulq_u8(vandq_u8(a_raw, mask2),
                           vandq_u8(b_raw, mask2));
            p1 = vmulq_u8(vandq_u8(vshrq_n_u8(a_raw, 2), mask2),
                           vandq_u8(vshrq_n_u8(b_raw, 2), mask2));
            p2 = vmulq_u8(vandq_u8(vshrq_n_u8(a_raw, 4), mask2),
                           vandq_u8(vshrq_n_u8(b_raw, 4), mask2));
            p3 = vmulq_u8(vshrq_n_u8(a_raw, 6),
                           vshrq_n_u8(b_raw, 6));
            uint16x8_t s1 = vpaddlq_u8(p0);
            s1 = vpadalq_u8(s1, p1);
            s1 = vpadalq_u8(s1, p2);
            s1 = vpadalq_u8(s1, p3);
            acc1 = vpadalq_u16(acc1, s1);
        }
        for (; i + 16 <= nbytes; i += 16) {
            uint8x16_t a_raw = vld1q_u8(px + i);
            uint8x16_t b_raw = vld1q_u8(py + i);
            uint8x16_t p0 = vmulq_u8(vandq_u8(a_raw, mask2),
                                      vandq_u8(b_raw, mask2));
            uint8x16_t p1 = vmulq_u8(vandq_u8(vshrq_n_u8(a_raw, 2), mask2),
                                      vandq_u8(vshrq_n_u8(b_raw, 2), mask2));
            uint8x16_t p2 = vmulq_u8(vandq_u8(vshrq_n_u8(a_raw, 4), mask2),
                                      vandq_u8(vshrq_n_u8(b_raw, 4), mask2));
            uint8x16_t p3 = vmulq_u8(vshrq_n_u8(a_raw, 6),
                                      vshrq_n_u8(b_raw, 6));
            uint16x8_t s = vpaddlq_u8(p0);
            s = vpadalq_u8(s, p1);
            s = vpadalq_u8(s, p2);
            s = vpadalq_u8(s, p3);
            acc0 = vpadalq_u16(acc0, s);
        }
        sum = vaddvq_u32(vaddq_u32(acc0, acc1));
        for (; i < nbytes; ++i) {
            uint8_t ab = px[i], bb = py[i];
            for (uint32_t s = 0; s < 8; s += 2) {
                uint32_t idx = 2 * i + s / 2;
                if (idx < n)
                    sum += ((ab >> s) & 3) * ((bb >> s) & 3);
            }
        }
#else
        for (uint32_t i = 0; i < nbytes; ++i) {
            uint8_t ab = px[i], bb = py[i];
            for (uint32_t s = 0; s < 8; s += 2) {
                uint32_t idx = 2 * i + s / 2;
                if (idx < n)
                    sum += ((ab >> s) & 3) * ((bb >> s) & 3);
            }
        }
#endif
        return sum;
    }

    // ── 1-bit:  Σ(a_i AND b_i) = popcount(a & b) ───────────────────────
    //
    // NEON: vcntq_u8 gives per-byte popcount.  Process 32 bytes per
    // iteration with dual accumulators for pipeline parallelism.
    if (bits_x == 1) {
        uint32_t nbytes = (n + 7) / 8;
#if OSQ_NEON
        uint32_t i = 0;
        uint32x4_t acc0 = vdupq_n_u32(0);
        uint32x4_t acc1 = vdupq_n_u32(0);

        for (; i + 32 <= nbytes; i += 32) {
            uint8x16_t a0 = vld1q_u8(px + i);
            uint8x16_t b0 = vld1q_u8(py + i);
            uint8x16_t pc0 = vcntq_u8(vandq_u8(a0, b0));
            uint16x8_t w0 = vpaddlq_u8(pc0);
            acc0 = vpadalq_u16(acc0, w0);

            uint8x16_t a1 = vld1q_u8(px + i + 16);
            uint8x16_t b1 = vld1q_u8(py + i + 16);
            uint8x16_t pc1 = vcntq_u8(vandq_u8(a1, b1));
            uint16x8_t w1 = vpaddlq_u8(pc1);
            acc1 = vpadalq_u16(acc1, w1);
        }
        for (; i + 16 <= nbytes; i += 16) {
            uint8x16_t a = vld1q_u8(px + i);
            uint8x16_t b = vld1q_u8(py + i);
            uint8x16_t pc = vcntq_u8(vandq_u8(a, b));
            uint16x8_t w = vpaddlq_u8(pc);
            acc0 = vpadalq_u16(acc0, w);
        }
        sum = vaddvq_u32(vaddq_u32(acc0, acc1));
        // Scalar tail.
        for (; i < nbytes; ++i)
            sum += __builtin_popcount(px[i] & py[i]);
#else
        for (uint32_t i = 0; i < nbytes; ++i)
            sum += __builtin_popcount(px[i] & py[i]);
#endif
        return sum;
    }

    // ── General (3-bit) ────────────────────────────────────────────────
    {
        std::vector<uint8_t> ax(n), bx(n);
        detail::unpack_bits(px, ax.data(), n, bits_x);
        detail::unpack_bits(py, bx.data(), n, bits_x);
        for (uint32_t i = 0; i < n; ++i)
            sum += static_cast<uint64_t>(ax[i]) * bx[i];
        return sum;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Public dot products
// ═══════════════════════════════════════════════════════════════════════════

float OSQ::dot_asymmetric(const QueryState& qs,
                           const OSQVector& doc) const {
    // ⟨q, x⟩ = ⟨q, c⟩ + (⟨c, x⟩ − ‖c‖²) + ⟨q', x̃'⟩
    //         = qc + (cx − cc) + norm · (a · Σq'_i + Δ · Σ(q'_i · idx_i))
    //
    // where q' = q − c is the centred query (stored in qs.query)
    // and x̃' is the quantised centred residual.
    // The (cx − cc) term uses the exact precomputed ⟨c, x⟩, introducing
    // no quantization error.
    float fi_dot = float_int_dot(qs.query.data(),
                                  doc.packed.data(), doc.dim,
                                  doc.bit_width);
    return qs.qc + (doc.cx - centroid_sq_)
         + doc.norm * (doc.a * qs.query_sum + doc.delta * fi_dot);
}

float OSQ::dot_symmetric(const OSQVector& x,
                          const OSQVector& y) const {
    // ⟨x, y⟩ = ⟨c + x', c + y'⟩ = ‖c‖² + ⟨c, x'⟩ + ⟨c, y'⟩ + ⟨x', y'⟩
    //
    // But ⟨c, x'⟩ = ⟨c, x − c⟩ = ⟨c, x⟩ − ‖c‖² = cx − ‖c‖²
    // So:  ⟨x, y⟩ = cx + cy − ‖c‖² + ⟨x̃', ỹ'⟩
    //
    // The quantised residual dot ⟨x̃', ỹ'⟩ uses the integer kernel:
    //   = norm_x·norm_y ×
    //     (d·a_x·a_y + a_x·Δ_y·Σidy + a_y·Δ_x·Σidx + Δ_x·Δ_y·Σ(idx·idy))
    assert(x.dim == y.dim);
    uint32_t d = x.dim;

    uint64_t ii_dot;

    // ── Bit-plane fast paths ───────────────────────────────────────────
    // When both vectors have precomputed bit-planes, decompose the
    // integer dot product into AND+popcount operations.  This avoids
    // the per-element multiply and runs close to the 1-1 popcount speed.
    //
    //   b-bit × 1-bit (mixed):  Σ w_k · popcount(plane_k & p1)
    //   b-bit × b-bit (same):   ΣΣ w_jk · popcount(x_plane_j & y_plane_k)
    //
    // The helper popcount_and is the same AND+popcount kernel as 1-1.

#if OSQ_NEON
    auto popcount_and = [](const uint8_t* pa, const uint8_t* pb,
                           uint32_t nbytes) -> uint64_t {
        uint32_t i = 0;
        uint32x4_t acc0 = vdupq_n_u32(0);
        uint32x4_t acc1 = vdupq_n_u32(0);
        for (; i + 32 <= nbytes; i += 32) {
            uint8x16_t a0 = vld1q_u8(pa + i);
            uint8x16_t b0 = vld1q_u8(pb + i);
            uint8x16_t pc0 = vcntq_u8(vandq_u8(a0, b0));
            uint16x8_t w0 = vpaddlq_u8(pc0);
            acc0 = vpadalq_u16(acc0, w0);
            uint8x16_t a1 = vld1q_u8(pa + i + 16);
            uint8x16_t b1 = vld1q_u8(pb + i + 16);
            uint8x16_t pc1 = vcntq_u8(vandq_u8(a1, b1));
            uint16x8_t w1 = vpaddlq_u8(pc1);
            acc1 = vpadalq_u16(acc1, w1);
        }
        for (; i + 16 <= nbytes; i += 16) {
            uint8x16_t a0 = vld1q_u8(pa + i);
            uint8x16_t b0 = vld1q_u8(pb + i);
            uint8x16_t pc0 = vcntq_u8(vandq_u8(a0, b0));
            uint16x8_t w0 = vpaddlq_u8(pc0);
            acc0 = vpadalq_u16(acc0, w0);
        }
        uint64_t s = vaddvq_u32(vaddq_u32(acc0, acc1));
        for (; i < nbytes; ++i)
            s += __builtin_popcount(pa[i] & pb[i]);
        return s;
    };
#else
    auto popcount_and = [](const uint8_t* pa, const uint8_t* pb,
                           uint32_t nbytes) -> uint64_t {
        uint64_t s = 0;
        for (uint32_t i = 0; i < nbytes; ++i)
            s += __builtin_popcount(pa[i] & pb[i]);
        return s;
    };
#endif

    bool have_planes_x = !x.bit_planes[0].empty();
    bool have_planes_y = !y.bit_planes[0].empty();

    if (x.bit_width != y.bit_width) {
        // ── Mixed width: 4-bit × 1-bit using precomputed planes ────────
        const OSQVector& vhi = (x.bit_width > y.bit_width) ? x : y;
        const OSQVector& vlo = (x.bit_width > y.bit_width) ? y : x;

        if (vhi.bit_width == 4 && vlo.bit_width == 1
            && !vhi.bit_planes[0].empty()) {
            uint32_t nb = (d + 7) / 8;
            const uint8_t* p1 = vlo.packed.data();
            ii_dot = 1 * popcount_and(vhi.bit_planes[0].data(), p1, nb)
                   + 2 * popcount_and(vhi.bit_planes[1].data(), p1, nb)
                   + 4 * popcount_and(vhi.bit_planes[2].data(), p1, nb)
                   + 8 * popcount_and(vhi.bit_planes[3].data(), p1, nb);
        } else {
            ii_dot = int_int_dot(x.packed.data(), x.bit_width,
                                 y.packed.data(), y.bit_width, d);
        }
    } else if (x.bit_width == 2 && have_planes_x && have_planes_y) {
        // ── 2-bit same width: 4 popcount passes ────────────────────────
        // Σ(2x1+x0)(2y1+y0) = 4·pc(x1&y1) + 2·(pc(x1&y0) + pc(x0&y1))
        //                    + pc(x0&y0)
        uint32_t nb = (d + 7) / 8;
        const uint8_t* x0 = x.bit_planes[0].data();
        const uint8_t* x1 = x.bit_planes[1].data();
        const uint8_t* y0 = y.bit_planes[0].data();
        const uint8_t* y1 = y.bit_planes[1].data();
        ii_dot = 4 * popcount_and(x1, y1, nb)
               + 2 * popcount_and(x1, y0, nb)
               + 2 * popcount_and(x0, y1, nb)
               +     popcount_and(x0, y0, nb);
    } else {
        // ── 1-bit, 4-bit same-width, and fallback ──────────────────────
        // 4-bit same-width: 16 popcount passes is slower than direct
        // nibble multiply, so we use the standard int_int_dot kernel.
        ii_dot = int_int_dot(x.packed.data(), x.bit_width,
                             y.packed.data(), y.bit_width, d);
    }

    float residual_dot = static_cast<float>(d) * x.a * y.a
                       + x.a * y.delta * y.sum_indices
                       + y.a * x.delta * x.sum_indices
                       + x.delta * y.delta * static_cast<float>(ii_dot);

    return x.cx + y.cx - centroid_sq_
         + x.norm * y.norm * residual_dot;
}

void OSQ::dot_asymmetric_batch(const QueryState& qs,
                                const OSQVector* docs, uint32_t n,
                                float* out) const {
    for (uint32_t i = 0; i < n; ++i)
        out[i] = dot_asymmetric(qs, docs[i]);
}

} // namespace osq

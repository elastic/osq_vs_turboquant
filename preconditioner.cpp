#include "preconditioner.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define PRECOND_NEON 1
#else
#define PRECOND_NEON 0
#endif

// ═══════════════════════════════════════════════════════════════════════════
//  NEON-optimised matrix-vector multiply
// ═══════════════════════════════════════════════════════════════════════════

void matrix_vector_multiply(std::size_t dim,
                            const float* __restrict__ M,
                            const float* __restrict__ x,
                            float* __restrict__ y) {
#if PRECOND_NEON
    std::size_t i = 0;

    // Process 4 output rows at a time — keeps 4 independent FMA chains
    // in flight for good NEON pipeline utilisation.
    for (; i + 4 <= dim; i += 4) {
        const float* r0 = M + (i + 0) * dim;
        const float* r1 = M + (i + 1) * dim;
        const float* r2 = M + (i + 2) * dim;
        const float* r3 = M + (i + 3) * dim;

        float32x4_t a0 = vdupq_n_f32(0.0f);
        float32x4_t a1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f);
        float32x4_t a3 = vdupq_n_f32(0.0f);

        std::size_t j = 0;
        for (; j + 4 <= dim; j += 4) {
            float32x4_t v = vld1q_f32(x + j);
            a0 = vfmaq_f32(a0, vld1q_f32(r0 + j), v);
            a1 = vfmaq_f32(a1, vld1q_f32(r1 + j), v);
            a2 = vfmaq_f32(a2, vld1q_f32(r2 + j), v);
            a3 = vfmaq_f32(a3, vld1q_f32(r3 + j), v);
        }

        float s0 = vaddvq_f32(a0);
        float s1 = vaddvq_f32(a1);
        float s2 = vaddvq_f32(a2);
        float s3 = vaddvq_f32(a3);

        // Tail (dim not a multiple of 4).
        for (; j < dim; ++j) {
            float xj = x[j];
            s0 += r0[j] * xj;
            s1 += r1[j] * xj;
            s2 += r2[j] * xj;
            s3 += r3[j] * xj;
        }

        y[i + 0] = s0;
        y[i + 1] = s1;
        y[i + 2] = s2;
        y[i + 3] = s3;
    }

    // Remaining rows (0-3).
    for (; i < dim; ++i) {
        const float* row = M + i * dim;
        float32x4_t acc = vdupq_n_f32(0.0f);
        std::size_t j = 0;
        for (; j + 4 <= dim; j += 4) {
            acc = vfmaq_f32(acc, vld1q_f32(row + j), vld1q_f32(x + j));
        }
        float s = vaddvq_f32(acc);
        for (; j < dim; ++j) s += row[j] * x[j];
        y[i] = s;
    }
#else
    // Scalar fallback.
    for (std::size_t i = 0; i < dim; ++i) {
        const float* row = M + i * dim;
        float s = 0.0f;
        for (std::size_t j = 0; j < dim; ++j) {
            s += row[j] * x[j];
        }
        y[i] = s;
    }
#endif
}

// ═══════════════════════════════════════════════════════════════════════════
//  Modified Gram-Schmidt  (double precision for numerical stability)
// ═══════════════════════════════════════════════════════════════════════════

static void modified_gram_schmidt(std::size_t dim, std::vector<double>& m) {
    for (std::size_t i = 0; i < dim; ++i) {
        auto* mi = m.data() + i * dim;

        double norm = 0.0;
        for (std::size_t j = 0; j < dim; ++j) {
            norm += mi[j] * mi[j];
        }
        norm = std::sqrt(norm);
        if (norm == 0.0) continue;

        double inv_norm = 1.0 / norm;
        for (std::size_t j = 0; j < dim; ++j) {
            mi[j] *= inv_norm;
        }

        for (std::size_t k = i + 1; k < dim; ++k) {
            auto* mk = m.data() + k * dim;
            double dot = 0.0;
            for (std::size_t j = 0; j < dim; ++j) {
                dot += mi[j] * mk[j];
            }
            for (std::size_t j = 0; j < dim; ++j) {
                mk[j] -= dot * mi[j];
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Random orthogonal block generation
// ═══════════════════════════════════════════════════════════════════════════

std::pair<std::vector<std::vector<float>>, std::vector<std::size_t>>
random_orthogonal(std::size_t dim, std::size_t block_dim) {
    block_dim = std::min(dim, block_dim);
    std::size_t nblocks = dim / block_dim;
    std::size_t rem = dim % block_dim;

    std::size_t total_blocks = nblocks + (rem > 0 ? 1 : 0);
    std::vector<std::vector<float>> blocks(total_blocks);
    std::vector<std::size_t> dim_blocks(total_blocks);

    std::mt19937 gen(215873873);
    std::normal_distribution<double> norm(0.0, 1.0);

    std::vector<double> m(block_dim * block_dim);
    for (std::size_t i = 0; i < nblocks; ++i) {
        std::generate_n(m.begin(), block_dim * block_dim,
                        [&] { return norm(gen); });
        modified_gram_schmidt(block_dim, m);
        blocks[i].assign(m.begin(), m.begin() + block_dim * block_dim);
        dim_blocks[i] = block_dim;
    }

    if (rem > 0) {
        m.resize(rem * rem);
        std::generate_n(m.begin(), rem * rem, [&] { return norm(gen); });
        modified_gram_schmidt(rem, m);
        blocks[nblocks].assign(m.begin(), m.begin() + rem * rem);
        dim_blocks[nblocks] = rem;
    }

    return {std::move(blocks), std::move(dim_blocks)};
}

// ═══════════════════════════════════════════════════════════════════════════
//  Dimension-to-block assignment strategies
// ═══════════════════════════════════════════════════════════════════════════

static std::vector<std::vector<std::size_t>>
equal_variance_permutation(std::size_t dim,
                           const std::vector<float>& corpus,
                           const std::vector<std::size_t>& dim_blocks) {
    // Compute per-coordinate variance.
    std::vector<OnlineMeanAndVariance> moments(dim);
    for (std::size_t i = 0; i < corpus.size(); i += dim) {
        for (std::size_t j = 0; j < dim; ++j) {
            moments[j].add(corpus[i + j]);
        }
    }

    // Sort coordinates by descending variance.
    std::vector<std::size_t> indices(dim);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](auto a, auto b) {
        return moments[a].var() > moments[b].var();
    });

    // Greedy assignment: always assign to the block with lowest total variance.
    std::vector<double> variances(dim_blocks.size(), 0.0);
    std::vector<std::vector<std::size_t>> assignment(dim_blocks.size());
    for (std::size_t i : indices) {
        auto j = std::min_element(variances.begin(), variances.end())
                 - variances.begin();
        assignment[j].push_back(i);
        variances[j] = (assignment[j].size() == dim_blocks[j]
                        ? std::numeric_limits<double>::max()
                        : variances[j] + moments[i].var());
    }

    // Sort within each block for cache-friendly access.
    for (auto& block : assignment) {
        std::sort(block.begin(), block.end());
    }
    return assignment;
}

static std::vector<std::vector<std::size_t>>
random_permutation(std::size_t dim,
                   const std::vector<std::size_t>& dim_blocks) {
    std::vector<std::size_t> indices(dim);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 gen(215873873);
    std::shuffle(indices.begin(), indices.end(), gen);

    std::vector<std::vector<std::size_t>> assignment(dim_blocks.size());
    std::size_t pos = 0;
    for (std::size_t j = 0; j < dim_blocks.size(); ++j) {
        for (std::size_t k = 0; k < dim_blocks[j]; ++k) {
            assignment[j].push_back(indices[pos++]);
        }
        std::sort(assignment[j].begin(), assignment[j].end());
    }
    return assignment;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Apply block-diagonal orthogonal transform
// ═══════════════════════════════════════════════════════════════════════════

void random_orthogonal_transform(std::size_t dim,
                                 std::vector<float>& queries,
                                 std::vector<float>& corpus,
                                 std::size_t block_dim,
                                 PermutationMethod method) {
    auto [blocks, dim_blocks] = random_orthogonal(dim, block_dim);

    // Single-block case: no permutation needed.
    if (blocks.size() == 1) {
        std::vector<float> y(dim_blocks[0]);
        for (std::size_t i = 0; i < queries.size(); i += dim) {
            matrix_vector_multiply(dim_blocks[0], blocks[0].data(),
                                   queries.data() + i, y.data());
            std::copy(y.begin(), y.end(), queries.begin() + i);
        }
        for (std::size_t i = 0; i < corpus.size(); i += dim) {
            matrix_vector_multiply(dim_blocks[0], blocks[0].data(),
                                   corpus.data() + i, y.data());
            std::copy(y.begin(), y.end(), corpus.begin() + i);
        }
        return;
    }

    auto assignment = [&]() {
        switch (method) {
        case PermutationMethod::EqualVariance:
            return equal_variance_permutation(dim, corpus, dim_blocks);
        case PermutationMethod::Random:
            return random_permutation(dim, dim_blocks);
        }
        return random_permutation(dim, dim_blocks); // unreachable
    }();

    // Apply: gather block coordinates, multiply, scatter to permuted layout.
    // Output vectors are in permuted order — dot products are invariant
    // since both sides get the same permutation.

    auto transform_vecs = [&](std::vector<float>& vecs) {
        std::vector<float> orig(dim);
        std::vector<float> x, y;
        for (std::size_t i = 0; i < vecs.size(); /**/) {
            std::copy(vecs.begin() + i, vecs.begin() + i + dim, orig.begin());
            for (std::size_t j = 0; j < blocks.size(); ++j) {
                auto bd = dim_blocks[j];
                x.resize(bd);
                y.resize(bd);
                std::transform(assignment[j].begin(), assignment[j].end(),
                               x.begin(), [&](std::size_t k) {
                                   return orig[k];
                               });
                matrix_vector_multiply(bd, blocks[j].data(),
                                       x.data(), y.data());
                std::copy(y.begin(), y.end(), vecs.begin() + i);
                i += bd;
            }
        }
    };

    transform_vecs(queries);
    transform_vecs(corpus);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Hadamard rotation
// ═══════════════════════════════════════════════════════════════════════════

static uint32_t next_pow2(uint32_t v) {
    v--;
    v |= v >> 1;  v |= v >> 2;  v |= v >> 4;
    v |= v >> 8;  v |= v >> 16;
    return v + 1;
}

// In-place Fast Walsh-Hadamard Transform (unnormalised).
static void fwht(float* data, uint32_t n) {
#if PRECOND_NEON
    // NEON-accelerated butterfly for strides >= 4.
    uint32_t len = 1;
    for (; len < 4 && len < n; len <<= 1) {
        for (uint32_t i = 0; i < n; i += len << 1) {
            for (uint32_t j = 0; j < len; ++j) {
                float u = data[i + j];
                float v = data[i + j + len];
                data[i + j]       = u + v;
                data[i + j + len] = u - v;
            }
        }
    }
    for (; len < n; len <<= 1) {
        for (uint32_t i = 0; i < n; i += len << 1) {
            for (uint32_t j = 0; j < len; j += 4) {
                float32x4_t u = vld1q_f32(data + i + j);
                float32x4_t v = vld1q_f32(data + i + j + len);
                vst1q_f32(data + i + j,       vaddq_f32(u, v));
                vst1q_f32(data + i + j + len,  vsubq_f32(u, v));
            }
        }
    }
#else
    for (uint32_t len = 1; len < n; len <<= 1) {
        for (uint32_t i = 0; i < n; i += len << 1) {
            for (uint32_t j = 0; j < len; ++j) {
                float u = data[i + j];
                float v = data[i + j + len];
                data[i + j]       = u + v;
                data[i + j + len] = u - v;
            }
        }
    }
#endif
}

uint32_t hadamard_transform(std::size_t dim,
                            std::vector<float>& queries,
                            std::vector<float>& corpus,
                            uint64_t seed) {
    uint32_t d_pad = next_pow2(static_cast<uint32_t>(dim));

    // Generate random sign flips.
    std::mt19937_64 rng(seed);
    std::vector<float> signs(d_pad);
    for (uint32_t i = 0; i < d_pad; ++i) {
        signs[i] = (rng() & 1) ? 1.0f : -1.0f;
    }

    float scale = 1.0f / std::sqrt(static_cast<float>(d_pad));

    auto transform_vec = [&](std::vector<float>& vecs) {
        std::size_t n_vecs = vecs.size() / dim;
        std::vector<float> padded(n_vecs * d_pad, 0.0f);

        // Zero-pad and apply sign flips.
        for (std::size_t v = 0; v < n_vecs; ++v) {
            float* dst = padded.data() + v * d_pad;
            const float* src = vecs.data() + v * dim;
            for (std::size_t i = 0; i < dim; ++i) {
                dst[i] = src[i] * signs[i];
            }
            // Padding dimensions are already zero.
        }

        // In-place WHT + normalise.
        for (std::size_t v = 0; v < n_vecs; ++v) {
            float* p = padded.data() + v * d_pad;
            fwht(p, d_pad);
#if PRECOND_NEON
            float32x4_t s = vdupq_n_f32(scale);
            for (uint32_t i = 0; i < d_pad; i += 4) {
                vst1q_f32(p + i, vmulq_f32(vld1q_f32(p + i), s));
            }
#else
            for (uint32_t i = 0; i < d_pad; ++i) {
                p[i] *= scale;
            }
#endif
        }

        vecs = std::move(padded);
    };

    transform_vec(queries);
    transform_vec(corpus);

    return d_pad;
}

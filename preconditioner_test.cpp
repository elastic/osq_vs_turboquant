/// Preconditioner bake-off: block-diagonal orthogonal vs Hadamard rotation.
///
/// Compares transform throughput, MSE, and dot-product accuracy across
/// different preconditioning strategies applied before OSQ quantization.
///
/// Methods tested:
///   1. No transform (baseline)
///   2. Block-diagonal 32×32 random permutation
///   3. Block-diagonal 64×64 random permutation
///   4. Block-diagonal 32×32 variance-matching permutation
///   5. Block-diagonal 64×64 variance-matching permutation
///   6. Full dense random orthogonal (block_dim = dim)
///   7. Hadamard rotation (768 → 1024 with padding)
///
/// Data: anisotropic Gaussian (linear variance ramp across coordinates)
///       so the preconditioner has something meaningful to equalise.

#include "osq.h"
#include "preconditioner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

// ── Helpers ────────────────────────────────────────────────────────────────

static float fdot(const float* a, const float* b, uint32_t n) {
    double s = 0;
    for (uint32_t i = 0; i < n; ++i) s += double(a[i]) * b[i];
    return static_cast<float>(s);
}

// ── Data generation ────────────────────────────────────────────────────────

/// Anisotropic Gaussian: coordinate i has std dev sigma_i = 1 + scale * i / dim.
/// This creates a "ramp" where later coordinates have up to (1+scale)× more
/// variance, a challenging case for uniform quantization.
static void generate_anisotropic(uint32_t dim, uint32_t n_vecs,
                                float scale,
                                std::vector<float>& vecs,
                                uint64_t seed) {
    vecs.resize(std::size_t(dim) * n_vecs);
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (uint32_t v = 0; v < n_vecs; ++v) {
        float* p = vecs.data() + std::size_t(v) * dim;
        for (uint32_t i = 0; i < dim; ++i) {
            float sigma = 1.0f + scale * float(i) / float(dim);
            p[i] = dist(rng) * sigma;
        }
    }
}

/// Isotropic Gaussian (control).
static void generate_isotropic(uint32_t dim, uint32_t n_vecs,
                               std::vector<float>& vecs,
                               uint64_t seed) {
    vecs.resize(std::size_t(dim) * n_vecs);
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (uint32_t v = 0; v < n_vecs; ++v) {
        float* p = vecs.data() + std::size_t(v) * dim;
        for (uint32_t i = 0; i < dim; ++i) {
            p[i] = dist(rng);
        }
    }
}

// ── MSE measurement ────────────────────────────────────────────────────────

/// Measure relative MSE: Σ||x - dequant(quant(x))||² / Σ||x||²
/// Operates in the (possibly transformed) coordinate system.
static double measure_mse(uint32_t dim, uint32_t bits,
                         const std::vector<float>& vecs,
                         uint32_t n_vecs) {
    osq::Config cfg{dim, bits, 5, 1.0f}; // lambda=1 (pure MSE)
    osq::OSQ q(cfg);

    double total_mse = 0, total_nsq = 0;
    for (uint32_t v = 0; v < n_vecs; ++v) {
        const float* x = vecs.data() + std::size_t(v) * dim;
        osq::OSQVector qv = q.quantize(x);
        std::vector<float> xh(dim);
        q.dequantize(qv, xh.data());
        for (uint32_t i = 0; i < dim; ++i) {
            double d = x[i] - xh[i];
            total_mse += d * d;
        }
        total_nsq += fdot(x, x, dim);
    }
    return total_mse / total_nsq;
}

// ── Dot-product accuracy measurement ───────────────────────────────────────

/// Measure relative dot-product error in the transformed space.
/// Uses symmetric 1-bit doc / 4-bit query (production config).
/// Since both sides are in the same transformed coordinate system,
/// the true dot product in the transformed space equals the true
/// dot product in the original space (orthogonal invariance).
static double measure_dot_error(uint32_t dim, uint32_t doc_bits,
                              uint32_t query_bits,
                              const std::vector<float>& corpus,
                              const std::vector<float>& queries,
                              uint32_t n_docs, uint32_t n_queries) {
    osq::Config dcfg{dim, doc_bits, 5, 0.1f};
    osq::Config qcfg{dim, query_bits, 5, 0.1f};
    osq::OSQ osq_d(dcfg);
    osq::OSQ osq_q(qcfg);

    // Compute and set centroid from corpus.
    std::vector<float> centroid(dim, 0.0f);
    for (uint32_t v = 0; v < n_docs; ++v) {
        const float* x = corpus.data() + std::size_t(v) * dim;
        for (uint32_t i = 0; i < dim; ++i) centroid[i] += x[i];
    }
    float inv_n = 1.0f / float(n_docs);
    for (uint32_t i = 0; i < dim; ++i) centroid[i] *= inv_n;
    osq_d.set_centroid(centroid.data());
    osq_q.set_centroid(centroid.data());

    // Quantize docs.
    std::vector<osq::OSQVector> qdocs(n_docs);
    for (uint32_t v = 0; v < n_docs; ++v) {
        qdocs[v] = osq_d.quantize(corpus.data() + std::size_t(v) * dim);
    }

    double err_sq = 0, dot_sq = 0;
    for (uint32_t qi = 0; qi < n_queries; ++qi) {
        const float* q = queries.data() + std::size_t(qi) * dim;
        osq::OSQVector qq = osq_q.quantize(q);
        for (uint32_t di = 0; di < n_docs; ++di) {
            const float* x = corpus.data() + std::size_t(di) * dim;
            float exact = fdot(q, x, dim);
            float approx = osq_d.dot_symmetric(qq, qdocs[di]);
            double e = approx - exact;
            err_sq += e * e;
            dot_sq += exact * double(exact);
        }
    }
    return std::sqrt(err_sq / std::max(dot_sq, 1e-30));
}

// ── Transform throughput ───────────────────────────────────────────────────

struct transform_timing {
    double ns_per_vec;
    uint32_t effective_dim;
};

template <typename F>
static transform_timing bench_transform(
    uint32_t dim, uint32_t n_vecs,
    const std::vector<float>& corpus_orig,
    const std::vector<float>& queries_orig,
    const char* name,
    F apply_fn) {

    constexpr int k_reps = 10;
    double best_ms = 1e30;
    uint32_t eff_dim = dim;

    for (int r = 0; r < k_reps; ++r) {
        std::vector<float> corpus = corpus_orig;
        std::vector<float> queries = queries_orig;
        auto t0 = std::chrono::high_resolution_clock::now();
        eff_dim = apply_fn(corpus, queries);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        best_ms = std::min(best_ms, ms);
    }

    uint32_t total_vecs = n_vecs + static_cast<uint32_t>(queries_orig.size() / dim);
    double ns_per = (best_ms * 1e6) / total_vecs;
    std::printf("  %-28s  %7.0f ns/vec  (dim_eff=%u)\n",
                name, ns_per, eff_dim);
    return {ns_per, eff_dim};
}

// ── Main bake-off ──────────────────────────────────────────────────────────

struct Method {
    const char* name;
    // Returns effective dimension after transform.
    // Transforms corpus and queries in-place.
    uint32_t (*apply)(uint32_t dim,
                      std::vector<float>& corpus,
                      std::vector<float>& queries);
};

// Method implementations.

static uint32_t apply_none(uint32_t dim,
                          std::vector<float>&,
                          std::vector<float>&) {
    return dim;
}

static uint32_t apply_block32_random(uint32_t dim,
                                   std::vector<float>& corpus,
                                   std::vector<float>& queries) {
    random_orthogonal_transform(dim, queries, corpus, 32,
                              PermutationMethod::Random);
    return dim;
}

static uint32_t apply_block64_random(uint32_t dim,
                                   std::vector<float>& corpus,
                                   std::vector<float>& queries) {
    random_orthogonal_transform(dim, queries, corpus, 64,
                              PermutationMethod::Random);
    return dim;
}

static uint32_t apply_block32_var_match(uint32_t dim,
                                     std::vector<float>& corpus,
                                     std::vector<float>& queries) {
    random_orthogonal_transform(dim, queries, corpus, 32,
                              PermutationMethod::EqualVariance);
    return dim;
}

static uint32_t apply_block64_var_match(uint32_t dim,
                                     std::vector<float>& corpus,
                                     std::vector<float>& queries) {
    random_orthogonal_transform(dim, queries, corpus, 64,
                              PermutationMethod::EqualVariance);
    return dim;
}

static uint32_t apply_full_dense(uint32_t dim,
                               std::vector<float>& corpus,
                               std::vector<float>& queries) {
    random_orthogonal_transform(dim, queries, corpus, dim,
                              PermutationMethod::Random);
    return dim;
}

static uint32_t apply_hadamard(uint32_t dim,
                              std::vector<float>& corpus,
                              std::vector<float>& queries) {
    return hadamard_transform(dim, queries, corpus);
}

int main() {
    constexpr uint32_t dim = 768;
    constexpr uint32_t n_docs = 1000;
    constexpr uint32_t n_queries = 100;
    constexpr float aniso_scale = 4.0f; // max σ = 1 + 4 = 5

    Method methods[] = {
        {"No transform",          apply_none},
        {"Block 32 random",       apply_block32_random},
        {"Block 64 random",       apply_block64_random},
        {"Block 32 var-match",    apply_block32_var_match},
        {"Block 64 var-match",    apply_block64_var_match},
        {"Full dense orthogonal", apply_full_dense},
        {"Hadamard (768->1024)",  apply_hadamard},
    };
    constexpr int n_methods = sizeof(methods) / sizeof(methods[0]);

    // ── Generate data ──────────────────────────────────────────────────────

    std::vector<float> corpus_aniso, queries_aniso;
    generate_anisotropic(dim, n_docs, aniso_scale, corpus_aniso, 42);
    generate_anisotropic(dim, n_queries, aniso_scale, queries_aniso, 99);

    std::vector<float> corpus_iso, queries_iso;
    generate_isotropic(dim, n_docs, corpus_iso, 42);
    generate_isotropic(dim, n_queries, queries_iso, 99);

    // ── Transform throughput ───────────────────────────────────────────────

    std::printf("=== Transform throughput (d=%u, %u+%u vectors) ===\n",
                dim, n_docs, n_queries);
    for (int m = 0; m < n_methods; ++m) {
        bench_transform(dim, n_docs, corpus_aniso, queries_aniso,
                       methods[m].name,
                       [&](std::vector<float>& c, std::vector<float>& q) {
                           return methods[m].apply(dim, c, q);
                       });
    }

    // ── MSE comparison: anisotropic data ───────────────────────────────────

    std::printf("\n=== MSE comparison: anisotropic data (sigma ramp 1..%.0f) ===\n",
                1.0f + aniso_scale);
    std::printf("  %-28s  %10s  %10s  %10s\n",
                "Method", "1-bit", "2-bit", "4-bit");
    std::printf("  %-28s  %10s  %10s  %10s\n",
                "----------------------------", "----------", "----------", "----------");

    for (int m = 0; m < n_methods; ++m) {
        std::vector<float> corpus = corpus_aniso;
        std::vector<float> queries = queries_aniso;
        uint32_t eff_dim = methods[m].apply(dim, corpus, queries);

        double mse1 = measure_mse(eff_dim, 1, corpus, n_docs);
        double mse2 = measure_mse(eff_dim, 2, corpus, n_docs);
        double mse4 = measure_mse(eff_dim, 4, corpus, n_docs);

        std::printf("  %-28s  %10.6f  %10.6f  %10.6f\n",
                    methods[m].name, mse1, mse2, mse4);
    }

    // ── MSE comparison: isotropic data (control) ───────────────────────────

    std::printf("\n=== MSE comparison: isotropic data (control) ===\n");
    std::printf("  %-28s  %10s  %10s  %10s\n",
                "Method", "1-bit", "2-bit", "4-bit");
    std::printf("  %-28s  %10s  %10s  %10s\n",
                "----------------------------", "----------", "----------", "----------");

    for (int m = 0; m < n_methods; ++m) {
        std::vector<float> corpus = corpus_iso;
        std::vector<float> queries = queries_iso;
        uint32_t eff_dim = methods[m].apply(dim, corpus, queries);

        double mse1 = measure_mse(eff_dim, 1, corpus, n_docs);
        double mse2 = measure_mse(eff_dim, 2, corpus, n_docs);
        double mse4 = measure_mse(eff_dim, 4, corpus, n_docs);

        std::printf("  %-28s  %10.6f  %10.6f  %10.6f\n",
                    methods[m].name, mse1, mse2, mse4);
    }

    // ── Dot-product accuracy: anisotropic data ─────────────────────────────

    std::printf("\n=== Dot-product accuracy: anisotropic data, "
                "sym 1-4 centred ===\n");
    std::printf("  %-28s  %12s\n", "Method", "rel_dot_err");
    std::printf("  %-28s  %12s\n",
                "----------------------------", "------------");

    for (int m = 0; m < n_methods; ++m) {
        std::vector<float> corpus = corpus_aniso;
        std::vector<float> queries = queries_aniso;
        uint32_t eff_dim = methods[m].apply(dim, corpus, queries);

        double err = measure_dot_error(eff_dim, 1, 4,
                                     corpus, queries,
                                     n_docs, n_queries);
        std::printf("  %-28s  %12.6f\n", methods[m].name, err);
    }

    // ── Dot-product accuracy: isotropic data ───────────────────────────────

    std::printf("\n=== Dot-product accuracy: isotropic data, "
                "sym 1-4 centred ===\n");
    std::printf("  %-28s  %12s\n", "Method", "rel_dot_err");
    std::printf("  %-28s  %12s\n",
                "----------------------------", "------------");

    for (int m = 0; m < n_methods; ++m) {
        std::vector<float> corpus = corpus_iso;
        std::vector<float> queries = queries_iso;
        uint32_t eff_dim = methods[m].apply(dim, corpus, queries);

        double err = measure_dot_error(eff_dim, 1, 4,
                                     corpus, queries,
                                     n_docs, n_queries);
        std::printf("  %-28s  %12.6f\n", methods[m].name, err);
    }

    std::printf("\nDone.\n");
    return 0;
}

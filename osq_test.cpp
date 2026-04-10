#include "osq.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>

using namespace osq;

static float dot(const float* a, const float* b, uint32_t n) {
    float s = 0;
    for (uint32_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

// ── MSE distortion ──────────────────────────────────────────────────────

static void test_mse(uint32_t dim, uint32_t bits, uint32_t n_vecs) {
    Config cfg{dim, bits};
    OSQ osq(cfg);

    std::mt19937_64 rng(42);
    std::normal_distribution<float> dist(0, 1);

    double total_mse = 0, total_nsq = 0;
    for (uint32_t v = 0; v < n_vecs; ++v) {
        std::vector<float> x(dim);
        for (auto& xi : x) xi = dist(rng);

        OSQVector qv = osq.quantize(x.data());
        std::vector<float> xh(dim);
        osq.dequantize(qv, xh.data());

        for (uint32_t i = 0; i < dim; ++i) {
            double d = x[i] - xh[i];
            total_mse += d * d;
        }
        total_nsq += dot(x.data(), x.data(), dim);
    }
    std::printf("  dim=%4u  bits=%u  relative_mse=%.6f\n",
                dim, bits, total_mse / total_nsq);
}

// ── Dot product accuracy ────────────────────────────────────────────────

static void test_dot(uint32_t dim, uint32_t bits, uint32_t n_vecs) {
    Config cfg{dim, bits};
    OSQ osq(cfg);

    std::mt19937_64 rng(99);
    std::normal_distribution<float> dist(0, 1);

    std::vector<float> q(dim);
    for (auto& qi : q) qi = dist(rng);
    QueryState qs = osq.prepare_query(q.data());

    double err_sq = 0, dot_sq = 0;
    for (uint32_t v = 0; v < n_vecs; ++v) {
        std::vector<float> x(dim);
        for (auto& xi : x) xi = dist(rng);

        OSQVector qv = osq.quantize(x.data());
        float approx = osq.dot_asymmetric(qs, qv);
        float exact  = dot(q.data(), x.data(), dim);

        double e = approx - exact;
        err_sq += e * e;
        dot_sq += exact * (double)exact;
    }
    std::printf("  dim=%4u  bits=%u  dot_relative_err=%.6f\n",
                dim, bits, std::sqrt(err_sq / dot_sq));
}

// ── Symmetric dot accuracy ──────────────────────────────────────────────

static void test_symmetric(uint32_t dim, uint32_t bits, uint32_t n_vecs) {
    Config cfg{dim, bits};
    OSQ osq(cfg);

    std::mt19937_64 rng(77);
    std::normal_distribution<float> dist(0, 1);

    double err_sq = 0, dot_sq = 0;
    for (uint32_t v = 0; v < n_vecs; ++v) {
        std::vector<float> x(dim), y(dim);
        for (auto& xi : x) xi = dist(rng);
        for (auto& yi : y) yi = dist(rng);

        OSQVector qx = osq.quantize(x.data());
        OSQVector qy = osq.quantize(y.data());

        float approx = osq.dot_symmetric(qx, qy);
        float exact  = dot(x.data(), y.data(), dim);

        double e = approx - exact;
        err_sq += e * e;
        dot_sq += exact * (double)exact;
    }
    std::printf("  dim=%4u  bits=%u  sym_dot_rel_err=%.6f\n",
                dim, bits, std::sqrt(err_sq / dot_sq));
}

// ── Throughput benchmarks ───────────────────────────────────────────────

static void bench_asymmetric(uint32_t dim, uint32_t bits, uint32_t n_docs) {
    Config cfg{dim, bits};
    OSQ osq(cfg);

    std::mt19937_64 rng(0);
    std::normal_distribution<float> dist(0, 1);

    std::vector<OSQVector> docs(n_docs);
    for (auto& d : docs) {
        std::vector<float> x(dim);
        for (auto& xi : x) xi = dist(rng);
        d = osq.quantize(x.data());
    }

    std::vector<float> q(dim);
    for (auto& qi : q) qi = dist(rng);
    QueryState qs = osq.prepare_query(q.data());

    std::vector<float> scores(n_docs);
    osq.dot_asymmetric_batch(qs, docs.data(), n_docs, scores.data());

    constexpr int kReps = 100;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < kReps; ++r)
        osq.dot_asymmetric_batch(qs, docs.data(), n_docs, scores.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ns_per = (ms * 1e6) / (kReps * n_docs);
    std::printf("  [asym]  dim=%4u  bits=%u  docs=%u  %.1f ns/doc  (%.2f ms)\n",
                dim, bits, n_docs, ns_per, ms / kReps);
}

static void bench_symmetric(uint32_t dim, uint32_t bits, uint32_t n_docs) {
    Config cfg{dim, bits};
    OSQ osq(cfg);

    std::mt19937_64 rng(0);
    std::normal_distribution<float> dist(0, 1);

    std::vector<OSQVector> docs(n_docs);
    for (auto& d : docs) {
        std::vector<float> x(dim);
        for (auto& xi : x) xi = dist(rng);
        d = osq.quantize(x.data());
    }
    // "Query" is also quantised for symmetric case.
    std::vector<float> q(dim);
    for (auto& qi : q) qi = dist(rng);
    OSQVector qq = osq.quantize(q.data());

    std::vector<float> scores(n_docs);
    for (uint32_t i = 0; i < n_docs; ++i)
        scores[i] = osq.dot_symmetric(qq, docs[i]);

    constexpr int kReps = 100;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < kReps; ++r)
        for (uint32_t i = 0; i < n_docs; ++i)
            scores[i] = osq.dot_symmetric(qq, docs[i]);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ns_per = (ms * 1e6) / (kReps * n_docs);
    std::printf("  [sym]   dim=%4u  bits=%u  docs=%u  %.1f ns/doc  (%.2f ms)\n",
                dim, bits, n_docs, ns_per, ms / kReps);
}

// ── Randomised Hadamard transform ───────────────────────────────────────
// Fast Walsh-Hadamard butterfly, in-place, unnormalized.
// n must be a power of 2.
static void fwht_inplace(float* data, uint32_t n) {
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
}

// Apply randomised Hadamard: D·H·x / sqrt(d'), where D is a random
// diagonal ±1 matrix.  sign_flips[i] ∈ {+1, -1}.
static void rand_hadamard(float* out, const float* x, uint32_t d,
                           uint32_t d_pad, const float* sign_flips) {
    // Zero-pad and apply sign flips.
    for (uint32_t i = 0; i < d; ++i)
        out[i] = x[i] * sign_flips[i];
    for (uint32_t i = d; i < d_pad; ++i)
        out[i] = 0.0f;
    // In-place WHT.
    fwht_inplace(out, d_pad);
    // Normalise.
    float scale = 1.0f / std::sqrt(static_cast<float>(d_pad));
    for (uint32_t i = 0; i < d_pad; ++i)
        out[i] *= scale;
}

// Inverse: H^{-1} D^{-1} y = D · H · y / sqrt(d')  (H is self-inverse
// up to scaling, D is its own inverse).
static void inv_rand_hadamard(float* out, const float* y, uint32_t d,
                               uint32_t d_pad, const float* sign_flips) {
    // Copy y, apply WHT, normalise, then apply sign flips and truncate.
    std::vector<float> tmp(y, y + d_pad);
    fwht_inplace(tmp.data(), d_pad);
    float scale = 1.0f / std::sqrt(static_cast<float>(d_pad));
    for (uint32_t i = 0; i < d; ++i)
        out[i] = tmp[i] * scale * sign_flips[i];
}

static uint32_t next_pow2(uint32_t v) {
    v--;
    v |= v >> 1; v |= v >> 2; v |= v >> 4;
    v |= v >> 8; v |= v >> 16;
    return v + 1;
}

// ── Hadamard MSE comparison ────────────────────────────────────────────
// Compare MSE with and without Hadamard rotation to verify the
// theoretical d/d' improvement factor.
static void test_hadamard_mse(uint32_t dim, uint32_t bits, uint32_t n_vecs) {
    uint32_t d_pad = next_pow2(dim);
    Config cfg_plain{dim, bits, 5, 1.0f};     // lambda=1 (pure MSE)
    Config cfg_had{d_pad, bits, 5, 1.0f};
    OSQ osq_plain(cfg_plain);
    OSQ osq_had(cfg_had);

    // Fixed random sign flips for the diagonal matrix D.
    std::mt19937_64 rng_signs(12345);
    std::vector<float> sign_flips(d_pad);
    for (uint32_t i = 0; i < d_pad; ++i)
        sign_flips[i] = (rng_signs() & 1) ? 1.0f : -1.0f;

    std::mt19937_64 rng(42);
    std::normal_distribution<float> dist(0, 1);

    double mse_plain = 0, mse_had = 0, nsq = 0;
    for (uint32_t v = 0; v < n_vecs; ++v) {
        std::vector<float> x(dim);
        for (auto& xi : x) xi = dist(rng);
        double xnorm2 = dot(x.data(), x.data(), dim);
        nsq += xnorm2;

        // --- Plain OSQ (no Hadamard) ---
        {
            OSQVector qv = osq_plain.quantize(x.data());
            std::vector<float> xh(dim);
            osq_plain.dequantize(qv, xh.data());
            for (uint32_t i = 0; i < dim; ++i) {
                double d = x[i] - xh[i];
                mse_plain += d * d;
            }
        }

        // --- Hadamard: pad, rotate, quantize, dequantize, inverse ---
        {
            std::vector<float> y(d_pad);
            rand_hadamard(y.data(), x.data(), dim, d_pad, sign_flips.data());

            OSQVector qv = osq_had.quantize(y.data());
            std::vector<float> yh(d_pad);
            osq_had.dequantize(qv, yh.data());

            std::vector<float> xr(dim);
            inv_rand_hadamard(xr.data(), yh.data(), dim, d_pad,
                              sign_flips.data());

            for (uint32_t i = 0; i < dim; ++i) {
                double d = x[i] - xr[i];
                mse_had += d * d;
            }
        }
    }
    double rel_plain = mse_plain / nsq;
    double rel_had   = mse_had / nsq;
    double ratio     = rel_plain / rel_had;
    double theory    = static_cast<double>(d_pad) / dim;
    std::printf("  bits=%u  plain=%.6f  hadamard=%.6f  ratio=%.3f  "
                "theory(d'/d)=%.3f\n",
                bits, rel_plain, rel_had, ratio, theory);
}

int main() {
    std::printf("=== MSE distortion ===\n");
    for (uint32_t b = 1; b <= 4; ++b)
        test_mse(768, b, 1000);

    std::printf("\n=== Asymmetric dot error ===\n");
    for (uint32_t b = 1; b <= 4; ++b)
        test_dot(768, b, 1000);

    std::printf("\n=== Symmetric dot error ===\n");
    for (uint32_t b = 1; b <= 4; ++b)
        test_symmetric(768, b, 1000);

    std::printf("\n=== Throughput: asymmetric vs symmetric ===\n");
    for (uint32_t b : {1u, 2u, 4u}) {
        bench_asymmetric(768, b, 10000);
        bench_symmetric(768, b, 10000);
    }

    std::printf("\n=== Mixed symmetric: 4-bit query × 1-bit doc ===\n");
    {
        // Production pattern: docs at 1-bit, query at 4-bit, symmetric.
        Config doc_cfg{768, 1};
        Config q_cfg{768, 4};
        OSQ osq_doc(doc_cfg);
        OSQ osq_q(q_cfg);

        std::mt19937_64 rng(0);
        std::normal_distribution<float> dist(0, 1);

        uint32_t n_docs = 10000;
        std::vector<OSQVector> docs(n_docs);
        for (auto& d : docs) {
            std::vector<float> x(768);
            for (auto& xi : x) xi = dist(rng);
            d = osq_doc.quantize(x.data());
        }
        std::vector<float> q(768);
        for (auto& qi : q) qi = dist(rng);
        OSQVector qq = osq_q.quantize(q.data());

        // Warm up.
        std::vector<float> scores(n_docs);
        for (uint32_t i = 0; i < n_docs; ++i)
            scores[i] = osq_doc.dot_symmetric(qq, docs[i]);

        constexpr int kReps = 100;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < kReps; ++r)
            for (uint32_t i = 0; i < n_docs; ++i)
                scores[i] = osq_doc.dot_symmetric(qq, docs[i]);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double ns_per = (ms * 1e6) / (kReps * n_docs);
        std::printf("  [sym 4-1] dim=768  docs=%u  %.1f ns/doc  (%.2f ms)\n",
                    n_docs, ns_per, ms / kReps);
    }

    std::printf("\n=== Hadamard rotation MSE (lambda=1, 768→1024) ===\n");
    std::printf("  Theory: if QE drops by d/d', ratio should be %.3f\n",
                1024.0 / 768.0);
    for (uint32_t b = 1; b <= 4; ++b)
        test_hadamard_mse(768, b, 1000);

    std::printf("\n=== Scaling with dimension ===\n");
    bench_asymmetric(1536, 4, 10000);
    bench_symmetric(1536, 4, 10000);

    std::printf("\nDone.\n");
    return 0;
}

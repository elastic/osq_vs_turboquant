#include "turbo_quant.h"

#include <cstdio>
#include <cmath>
#include <random>
#include <chrono>

using namespace turbo_quant;

static float dot(const float* a, const float* b, uint32_t n) {
    float s = 0;
    for (uint32_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

static float norm(const float* a, uint32_t n) {
    return std::sqrt(dot(a, a, n));
}

// Test roundtrip MSE distortion.
static void test_mse_distortion(uint32_t dim, uint32_t bit_width,
                                 uint32_t n_vecs) {
    Codebook cb = Codebook::gaussian_approx();
    Config cfg{dim, bit_width, /*use_prod=*/false, /*seed=*/123};
    TurboQuant tq(cfg, cb);

    std::mt19937_64 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    double total_mse = 0;
    double total_norm_sq = 0;

    for (uint32_t v = 0; v < n_vecs; ++v) {
        std::vector<float> x(dim);
        for (auto& xi : x) xi = dist(rng);

        QuantizedVector qv = tq.quantize(x.data());

        std::vector<float> x_hat(dim);
        tq.dequantize(qv, x_hat.data());

        double mse = 0;
        for (uint32_t i = 0; i < dim; ++i) {
            double d = x[i] - x_hat[i];
            mse += d * d;
        }
        total_mse += mse;
        total_norm_sq += dot(x.data(), x.data(), dim);
    }

    double relative_mse = total_mse / total_norm_sq;
    std::printf("  dim=%4u  bits=%u  relative_mse=%.6f\n",
                dim, bit_width, relative_mse);
}

// Test asymmetric dot product accuracy.
static void test_asymmetric_dot(uint32_t dim, uint32_t bit_width,
                                 uint32_t n_vecs) {
    Codebook cb = Codebook::gaussian_approx();
    Config cfg{dim, bit_width, /*use_prod=*/false, /*seed=*/456};
    TurboQuant tq(cfg, cb);

    std::mt19937_64 rng(99);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    double total_err_sq = 0;
    double total_dot_sq = 0;

    // Fixed query.
    std::vector<float> q(dim);
    for (auto& qi : q) qi = dist(rng);
    QueryState qs = tq.prepare_query(q.data());

    for (uint32_t v = 0; v < n_vecs; ++v) {
        std::vector<float> x(dim);
        for (auto& xi : x) xi = dist(rng);

        QuantizedVector qv = tq.quantize(x.data());
        float approx_dot = tq.dot_asymmetric(qs, qv);
        float true_dot = dot(q.data(), x.data(), dim);

        double err = approx_dot - true_dot;
        total_err_sq += err * err;
        total_dot_sq += true_dot * (double)true_dot;
    }

    double relative_err = std::sqrt(total_err_sq / total_dot_sq);
    std::printf("  dim=%4u  bits=%u  dot_relative_err=%.6f\n",
                dim, bit_width, relative_err);
}

// Benchmark asymmetric dot throughput.
static void bench_dot(uint32_t dim, uint32_t bit_width, uint32_t n_docs) {
    Codebook cb = Codebook::gaussian_approx();
    Config cfg{dim, bit_width, /*use_prod=*/false, /*seed=*/789};
    TurboQuant tq(cfg, cb);

    std::mt19937_64 rng(0);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Quantize documents.
    std::vector<QuantizedVector> docs(n_docs);
    for (uint32_t i = 0; i < n_docs; ++i) {
        std::vector<float> x(dim);
        for (auto& xi : x) xi = dist(rng);
        docs[i] = tq.quantize(x.data());
    }

    // Prepare query.
    std::vector<float> q(dim);
    for (auto& qi : q) qi = dist(rng);
    QueryState qs = tq.prepare_query(q.data());

    // Warm up.
    std::vector<float> scores(n_docs);
    tq.dot_asymmetric_batch(qs, docs.data(), n_docs, scores.data());

    // Timed run.
    constexpr int kReps = 100;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < kReps; ++r)
        tq.dot_asymmetric_batch(qs, docs.data(), n_docs, scores.data());
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double per_doc_ns = (ms * 1e6) / (kReps * n_docs);
    std::printf("  dim=%4u  bits=%u  docs=%u  %.1f ns/doc  (%.2f ms total)\n",
                dim, bit_width, n_docs, per_doc_ns, ms / kReps);
}

int main() {
    std::printf("=== MSE distortion (lower is better) ===\n");
    for (uint32_t b = 1; b <= 4; ++b)
        test_mse_distortion(768, b, 1000);

    std::printf("\n=== MSE distortion vs dimension ===\n");
    for (uint32_t dim : {128u, 256u, 512u, 768u, 1536u})
        test_mse_distortion(dim, 4, 500);

    std::printf("\n=== Asymmetric dot product error ===\n");
    for (uint32_t b = 1; b <= 4; ++b)
        test_asymmetric_dot(768, b, 1000);

    std::printf("\n=== Throughput: inline FMA ===\n");
    for (uint32_t b : {1u, 2u, 4u})
        bench_dot(768, b, 10000);

    std::printf("\n=== Throughput: precomputed ADC tables ===\n");
    for (uint32_t b : {1u, 2u, 4u}) {
        Codebook cb = Codebook::gaussian_approx();
        Config cfg{768, b, false, 789};
        TurboQuant tq(cfg, cb);

        std::mt19937_64 rng(0);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        uint32_t n_docs = 10000;
        std::vector<QuantizedVector> docs(n_docs);
        for (uint32_t i = 0; i < n_docs; ++i) {
            std::vector<float> x(768);
            for (auto& xi : x) xi = dist(rng);
            docs[i] = tq.quantize(x.data());
        }
        std::vector<float> q(768);
        for (auto& qi : q) qi = dist(rng);
        QueryState qs = tq.prepare_query(q.data());

        std::vector<float> scores(n_docs);
        tq.dot_asymmetric_lut_batch(qs, docs.data(), n_docs, scores.data());

        constexpr int kReps = 100;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < kReps; ++r)
            tq.dot_asymmetric_lut_batch(qs, docs.data(), n_docs, scores.data());
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double ns_per = (ms * 1e6) / (kReps * n_docs);
        std::printf("  dim= 768  bits=%u  docs=%u  %.1f ns/doc  (%.2f ms)\n",
                    b, n_docs, ns_per, ms / kReps);
    }
    bench_dot(1536, 4, 10000);

    std::printf("\nDone.\n");
    return 0;
}

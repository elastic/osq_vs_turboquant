#include "osq.h"
#include "turbo_quant.h"

#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

// ── Helpers ─────────────────────────────────────────────────────────────────

static double fdot(const float* a, const float* b, uint32_t n) {
    double s = 0;
    for (uint32_t i = 0; i < n; ++i) s += (double)a[i] * b[i];
    return s;
}

static float fnorm(const float* a, uint32_t n) {
    return static_cast<float>(std::sqrt(fdot(a, a, n)));
}

/// Generate a query at angle theta to x, preserving the natural scale.
///
///   q = cos(theta) * x + sin(theta) * z_perp
///
/// where z_perp is a random vector orthogonal to x with ‖z_perp‖ = ‖x‖.
/// This keeps the query at the same scale as the document, matching the
/// production usage where queries and documents are from the same
/// embedding distribution.
static void make_query_at_angle(const float* x, uint32_t n,
                                float theta,
                                std::mt19937_64& rng,
                                std::vector<float>& q) {
    q.resize(n);
    float x_norm = fnorm(x, n);
    float inv_x = 1.0f / x_norm;

    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> z(n);
    for (auto& zi : z) zi = gauss(rng);

    // Orthogonalize z against x.
    float zx = static_cast<float>(fdot(z.data(), x, n)) * inv_x * inv_x;
    for (uint32_t i = 0; i < n; ++i)
        z[i] -= zx * x[i];

    // Rescale z_perp to ‖x‖ so the query keeps the same norm.
    float z_norm = fnorm(z.data(), n);
    if (z_norm < 1e-10f) z_norm = 1e-10f;
    float scale_z = x_norm / z_norm;

    float ct = std::cos(theta);
    float st = std::sin(theta);
    for (uint32_t i = 0; i < n; ++i)
        q[i] = ct * x[i] + st * z[i] * scale_z;
}

// ── Corpus generation ───────────────────────────────────────────────────────

struct Corpus {
    std::vector<std::vector<float>> vecs;
    std::vector<float> centroid;
};

static Corpus generate_corpus(uint32_t dim, uint32_t n_vecs,
                              float shift, uint64_t seed) {
    Corpus c;
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(shift, 1.0f);

    c.vecs.resize(n_vecs);
    c.centroid.assign(dim, 0.0f);

    for (uint32_t v = 0; v < n_vecs; ++v) {
        c.vecs[v].resize(dim);
        for (uint32_t i = 0; i < dim; ++i) {
            c.vecs[v][i] = dist(rng);
            c.centroid[i] += c.vecs[v][i];
        }
    }
    float inv_n = 1.0f / static_cast<float>(n_vecs);
    for (uint32_t i = 0; i < dim; ++i)
        c.centroid[i] *= inv_n;

    return c;
}

// ── Bias/variance decomposition ────────────────────────────────────────────

struct ErrorPair { double exact, approx; };

struct BiasStats {
    double raw_rmse;
    double debiased_rmse;
};

static BiasStats compute_bias_stats(const std::vector<ErrorPair>& pairs) {
    double sum_ae = 0, sum_ee = 0, sum_err2 = 0;
    for (const auto& p : pairs) {
        sum_ae += p.approx * p.exact;
        sum_ee += p.exact * p.exact;
        double e = p.approx - p.exact;
        sum_err2 += e * e;
    }
    double raw_rmse = std::sqrt(sum_err2 / std::max(sum_ee, 1e-30));
    double alpha = sum_ae / std::max(sum_ee, 1e-30);

    double sum_db2 = 0;
    for (const auto& p : pairs) {
        double debiased = p.approx / alpha - p.exact;
        sum_db2 += debiased * debiased;
    }
    double debiased_rmse = std::sqrt(sum_db2 / std::max(sum_ee, 1e-30));

    return {raw_rmse, debiased_rmse};
}

// ── Test runner ─────────────────────────────────────────────────────────────
//
// Production pattern: both query and document are quantized (symmetric),
// documents at dbits, queries at qbits.  No unit normalization — the
// interval [a, b] captures the natural scale.

struct Result {
    float angle_deg;
    BiasStats osq_sym_centered;   // symmetric, centered
    BiasStats osq_sym_raw;        // symmetric, no centroid
    BiasStats osq_asym_centered;  // asymmetric (float query), centered
    BiasStats tq_mse;             // TurboQuant MSE at dbits (asymmetric)
    BiasStats tq_mse_4;           // TurboQuant MSE at 4-bit (reference)
};

static Result test_angle(uint32_t dim, uint32_t dbits, uint32_t qbits,
                         float angle_deg, const Corpus& corpus,
                         uint32_t n_queries_per_vec) {
    // ── OSQ: symmetric, centered, no normalization ──────────────────────
    osq::Config doc_cfg{dim, dbits, 5, 0.1f};
    osq::Config q_cfg{dim, qbits, 5, 0.1f};
    osq::OSQ osq_doc_c(doc_cfg);
    osq::OSQ osq_q_c(q_cfg);
    osq_doc_c.set_centroid(corpus.centroid.data());
    osq_q_c.set_centroid(corpus.centroid.data());

    // ── OSQ: symmetric, no centroid, no normalization ───────────────────
    osq::OSQ osq_doc_r(doc_cfg);
    osq::OSQ osq_q_r(q_cfg);

    // ── OSQ: asymmetric (float query), centered, no normalization ───────
    osq::OSQ osq_doc_ac(doc_cfg);
    osq_doc_ac.set_centroid(corpus.centroid.data());

    // ── TurboQuant MSE at dbits (storage-equivalent comparison) ────────
    turbo_quant::Codebook cb = turbo_quant::Codebook::gaussian_approx();
    turbo_quant::Config tq_cfg{dim, dbits, false, 42};
    turbo_quant::TurboQuant tq(tq_cfg, cb);

    // ── TurboQuant MSE at 4-bit (reference — TQ's best practical point) ─
    turbo_quant::Config tq4_cfg{dim, 4, false, 42};
    turbo_quant::TurboQuant tq4(tq4_cfg, cb);

    std::mt19937_64 rng(123 + static_cast<uint64_t>(angle_deg * 1000));
    float theta = angle_deg * static_cast<float>(M_PI) / 180.0f;

    std::vector<ErrorPair> sc_pairs, sr_pairs, ac_pairs, tq_pairs, tq4_pairs;

    for (const auto& x : corpus.vecs) {
        osq::OSQVector dv_c  = osq_doc_c.quantize(x.data());
        osq::OSQVector dv_r  = osq_doc_r.quantize(x.data());
        osq::OSQVector dv_ac = osq_doc_ac.quantize(x.data());
        turbo_quant::QuantizedVector tqv = tq.quantize(x.data());
        turbo_quant::QuantizedVector tqv4 = tq4.quantize(x.data());

        for (uint32_t qi = 0; qi < n_queries_per_vec; ++qi) {
            std::vector<float> q;
            make_query_at_angle(x.data(), dim, theta, rng, q);
            double exact = fdot(q.data(), x.data(), dim);

            // Symmetric centered.
            {
                osq::OSQVector qv = osq_q_c.quantize(q.data());
                double approx = osq_doc_c.dot_symmetric(qv, dv_c);
                sc_pairs.push_back({exact, approx});
            }

            // Symmetric raw (no centroid).
            {
                osq::OSQVector qv = osq_q_r.quantize(q.data());
                double approx = osq_doc_r.dot_symmetric(qv, dv_r);
                sr_pairs.push_back({exact, approx});
            }

            // Asymmetric centered (float query).
            {
                osq::QueryState qs = osq_doc_ac.prepare_query(q.data());
                double approx = osq_doc_ac.dot_asymmetric(qs, dv_ac);
                ac_pairs.push_back({exact, approx});
            }

            // TurboQuant at dbits.
            {
                turbo_quant::QueryState tqs = tq.prepare_query(q.data());
                double approx = tq.dot_asymmetric(tqs, tqv);
                tq_pairs.push_back({exact, approx});
            }

            // TurboQuant at 4-bit (reference).
            {
                turbo_quant::QueryState tqs4 = tq4.prepare_query(q.data());
                double approx = tq4.dot_asymmetric(tqs4, tqv4);
                tq4_pairs.push_back({exact, approx});
            }
        }
    }

    return {
        angle_deg,
        compute_bias_stats(sc_pairs),
        compute_bias_stats(sr_pairs),
        compute_bias_stats(ac_pairs),
        compute_bias_stats(tq_pairs),
        compute_bias_stats(tq4_pairs)
    };
}

static void run_test(const char* label, const Corpus& corpus,
                     uint32_t dim, uint32_t dbits, uint32_t qbits) {
    float angles[] = {0.0f, 5.0f, 10.0f, 20.0f, 30.0f, 45.0f, 60.0f};
    uint32_t n_angles = sizeof(angles) / sizeof(angles[0]);
    uint32_t n_queries = 5;

    std::printf("--- %s  doc=%u-bit  query=%u-bit ---\n", label, dbits, qbits);
    std::printf("  %7s  %10s %10s  %10s %10s  %10s %10s  %10s %10s  %10s %10s\n",
                "angle",
                "sym+c", "debiased",
                "sym raw", "debiased",
                "asym+c", "debiased",
                "TQ @doc", "debiased",
                "TQ @4b", "debiased");
    std::printf("  %7s  %10s %10s  %10s %10s  %10s %10s  %10s %10s  %10s %10s\n",
                "-------",
                "----------", "----------",
                "----------", "----------",
                "----------", "----------",
                "----------", "----------",
                "----------", "----------");

    for (uint32_t a = 0; a < n_angles; ++a) {
        Result r = test_angle(dim, dbits, qbits, angles[a], corpus, n_queries);
        std::printf("  %5.1f°   %10.6f %10.6f  %10.6f %10.6f  %10.6f %10.6f  %10.6f %10.6f  %10.6f %10.6f\n",
                    r.angle_deg,
                    r.osq_sym_centered.raw_rmse, r.osq_sym_centered.debiased_rmse,
                    r.osq_sym_raw.raw_rmse, r.osq_sym_raw.debiased_rmse,
                    r.osq_asym_centered.raw_rmse, r.osq_asym_centered.debiased_rmse,
                    r.tq_mse.raw_rmse, r.tq_mse.debiased_rmse,
                    r.tq_mse_4.raw_rmse, r.tq_mse_4.debiased_rmse);
    }
    std::printf("\n");
}

int main() {
    constexpr uint32_t dim = 768;
    constexpr uint32_t n_vecs = 500;

    // ── Zero-mean corpus ────────────────────────────────────────────────
    std::printf("=== Zero-mean corpus (shift=0) ===\n\n");
    {
        Corpus corpus = generate_corpus(dim, n_vecs, 0.0f, 7);
        run_test("shift=0", corpus, dim, 1, 4);  // production: 1-bit doc, 4-bit query
        run_test("shift=0", corpus, dim, 4, 4);
        run_test("shift=0", corpus, dim, 1, 1);
    }

    // ── Shifted corpus (models real embedding bias) ─────────────────────
    std::printf("=== Shifted corpus (shift=2.0) ===\n\n");
    {
        Corpus corpus = generate_corpus(dim, n_vecs, 2.0f, 7);
        run_test("shift=2", corpus, dim, 1, 4);  // production: 1-bit doc, 4-bit query
        run_test("shift=2", corpus, dim, 4, 4);
        run_test("shift=2", corpus, dim, 1, 1);
    }

    return 0;
}

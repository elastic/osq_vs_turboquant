#pragma once

/// Lightweight utilities shared across test programs.

#include <cmath>
#include <cstddef>

// ── Online mean and variance (Welford's algorithm) ─────────────────────────

class OnlineMeanAndVariance {
public:
    void add(double x) {
        ++n_;
        double delta = x - mean_;
        mean_ += delta / static_cast<double>(n_);
        double delta2 = x - mean_;
        m2_ += delta * delta2;
    }

    double mean() const { return mean_; }
    double var() const { return n_ > 1 ? m2_ / static_cast<double>(n_ - 1) : 0.0; }
    std::size_t count() const { return n_; }

private:
    std::size_t n_{0};
    double mean_{0.0};
    double m2_{0.0};
};

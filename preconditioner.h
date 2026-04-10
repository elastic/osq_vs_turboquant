#pragma once

/// Block-diagonal orthogonal preconditioner for scalar quantization.
///
/// Two dimension-assignment strategies:
///   EqualVariance – greedy assignment that balances total variance per block
///   Random        – random shuffle (production default)
///
/// Each block is an independent random orthogonal matrix (via Gram-Schmidt
/// on Gaussian draws).  The special case block_dim >= dim gives a single
/// dense random rotation.
///
/// Also provides a Hadamard rotation wrapper for comparison (pads to
/// the next power of 2).
///
/// NEON-optimised matrix-vector multiply for the block transforms.

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

enum class PermutationMethod {
    EqualVariance,
    Random
};

// ── Block-diagonal random orthogonal transform ─────────────────────────────

/// Generate random orthogonal blocks.
/// Returns (blocks, dim_blocks) where blocks[i] is a row-major
/// dim_blocks[i] × dim_blocks[i] orthogonal matrix.
std::pair<std::vector<std::vector<float>>, std::vector<std::size_t>>
random_orthogonal(std::size_t dim, std::size_t block_dim);

/// Apply block-diagonal random orthogonal transform in-place.
/// Permutes dimensions into blocks (according to method), then
/// applies per-block orthogonal rotation.  Both queries and corpus
/// are left in the permuted coordinate system (dot products are
/// invariant).
void random_orthogonal_transform(std::size_t dim,
                                 std::vector<float>& queries,
                                 std::vector<float>& corpus,
                                 std::size_t block_dim,
                                 PermutationMethod method);

// ── Hadamard rotation ──────────────────────────────────────────────────────

/// Apply randomised Hadamard rotation in-place: zero-pad dim → dim_padded
/// (next power of 2), random sign flips, Walsh-Hadamard butterfly,
/// normalise by 1/√dim_padded.
/// Returns dim_padded.  Vectors are resized to dim_padded.
uint32_t hadamard_transform(std::size_t dim,
                            std::vector<float>& queries,
                            std::vector<float>& corpus,
                            uint64_t seed = 12345);

// ── NEON-optimised matrix-vector multiply ──────────────────────────────────

/// y = M * x  where M is dim × dim row-major.
/// NEON path processes 4 output rows per iteration with vectorised
/// dot products.  Falls back to scalar on non-ARM.
void matrix_vector_multiply(std::size_t dim,
                            const float* M,
                            const float* x,
                            float* y);

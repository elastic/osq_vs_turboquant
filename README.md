# OSQ vs TurboQuant: Scalar Quantization Comparison

Reference implementations and benchmarks comparing Elastic's Optimised Scalar Quantization (OSQ) with Google's TurboQuant (ICLR 2026) for vector search.

## Files

| File | Description |
|------|-------------|
| `osq.h` | OSQ header: Config, OSQVector, QueryState, OSQ class |
| `osq.cpp` | OSQ implementation with NEON-optimized kernels |
| `turbo_quant.h` | TurboQuant header: precomputed Lloyd-Max codebooks |
| `turbo_quant.cpp` | TurboQuant implementation: Hadamard rotation, LUT scoring |
| `preconditioner.h` | Block-diagonal preconditioner and Hadamard rotation |
| `preconditioner.cpp` | NEON-optimized matrix-vector multiply, Gram-Schmidt, transforms |
| `utils.h` | Shared utilities (online mean/variance) |
| `osq_test.cpp` | MSE distortion, dot-product error, throughput benchmarks, Hadamard comparison |
| `turbo_quant_test.cpp` | TurboQuant MSE and dot-product error tests |
| `dot_angle_test.cpp` | Angle-dependent dot-product error with bias/variance decomposition |
| `preconditioner_test.cpp` | Block-diagonal preconditioner vs Hadamard rotation bake-off |

## Building

All files are C++20 with no external dependencies. On ARM (Apple Silicon / Linux aarch64), NEON intrinsics are auto-detected and enabled.

### GCC

```bash
# Main benchmark suite (MSE, dot error, throughput, Hadamard comparison)
g++ -O3 -std=c++20 -o osq_test osq_test.cpp osq.cpp turbo_quant.cpp

# TurboQuant standalone tests
g++ -O3 -std=c++20 -o turbo_quant_test turbo_quant_test.cpp turbo_quant.cpp

# Angle-dependent dot-product error (production config comparison)
g++ -O3 -std=c++20 -o dot_angle_test dot_angle_test.cpp osq.cpp turbo_quant.cpp

# Preconditioner bake-off (block-diagonal vs Hadamard)
g++ -O3 -std=c++20 -o preconditioner_test preconditioner_test.cpp preconditioner.cpp osq.cpp
```

### Clang (macOS / Apple Silicon)

```bash
# Main benchmark suite
clang++ -O3 -std=c++20 -o osq_test osq_test.cpp osq.cpp turbo_quant.cpp

# TurboQuant standalone tests
clang++ -O3 -std=c++20 -o turbo_quant_test turbo_quant_test.cpp turbo_quant.cpp

# Angle-dependent dot-product error
clang++ -O3 -std=c++20 -o dot_angle_test dot_angle_test.cpp osq.cpp turbo_quant.cpp

# Preconditioner bake-off
clang++ -O3 -std=c++20 -o preconditioner_test preconditioner_test.cpp preconditioner.cpp osq.cpp
```

On Apple Silicon, `clang++` enables NEON by default. For more aggressive optimization targeting the native CPU:

```bash
clang++ -O3 -std=c++20 -mcpu=native -o preconditioner_test preconditioner_test.cpp preconditioner.cpp osq.cpp
```

### Notes

On x86 without NEON, the code falls back to portable scalar paths. To force-disable NEON even on ARM, compile with `-DOSQ_NEON=0`.

## Running

```bash
./osq_test              # MSE, dot error, throughput, Hadamard comparison
./turbo_quant_test      # TurboQuant-only tests
./dot_angle_test        # angle sweep for both methods
./preconditioner_test   # block-diagonal vs Hadamard bake-off
```

## Key results (ARM NEON, d=768)

### MSE distortion (relative MSE = ||x - x_hat||^2 / ||x||^2)

| Bits | OSQ (lambda=0.1) | OSQ (lambda=1) | OSQ + Hadamard | TurboQuant |
|:----:|:---:|:---:|:---:|:---:|
| 1 | 0.512 | 0.362 | 0.306 | 0.307 |
| 2 | 0.138 | 0.118 | 0.092 | 0.092 |
| 3 | 0.038 | 0.037 | 0.028 | 0.026 |
| 4 | 0.011 | 0.011 | 0.009 | 0.007 |

OSQ with Hadamard rotation matches TurboQuant at 1-2 bits; the Lloyd-Max centroids contribute at most ~1.1x at 3-4 bits.

### Symmetric dot-product throughput (10k docs, 100 reps)

| Bits | OSQ asymmetric | OSQ symmetric | OSQ sym 1-4 | TurboQuant |
|:----:|:-:|:-:|:-:|:-:|
| 1 | 135 ns/doc | 7 ns/doc | -- | 298 ns/doc |
| 2 | 124 ns/doc | 14 ns/doc | -- | 286 ns/doc |
| 4 | 102 ns/doc | 22 ns/doc | 14 ns/doc | 206 ns/doc |

OSQ symmetric scoring is 10-40x faster than TurboQuant. The mixed 4-1 kernel (production config: 4-bit query, 1-bit doc) runs at 14 ns/doc via bit-plane decomposition.

### Production config dot-product error (1-bit doc, 4-bit query, centroid centered)

Raw dot-product error conflates multiplicative bias (ranking-irrelevant) with noise (ranking-relevant). After debiasing:

- **Zero-mean data**: OSQ has 1.2-1.4x lower ranking noise than TQ @1-bit at small angles (0-10°), thanks to the anisotropic loss concentrating accuracy along the query direction.
- **Shifted data**: Centroid centering (applied to both document and query) gives OSQ **0.001** debiased noise vs TQ @1-bit's **0.007** and TQ @4-bit's **0.006** at 0° — better ranking accuracy at less than 1/5 the storage.

### Preconditioner comparison (anisotropic data, sigma ramp 1..5)

**MSE (lambda=1):**

| Method | 1-bit | 2-bit | 4-bit |
|:-------|:---:|:---:|:---:|
| No transform | 0.443 | 0.157 | 0.018 |
| Block 32x32 | 0.368 | 0.121 | 0.012 |
| Block 64x64 | 0.365 | 0.119 | 0.012 |
| Full dense | 0.362 | 0.118 | 0.011 |
| Hadamard | 0.362 | 0.118 | 0.011 |

Block 32x32 already recovers 93% of the gap; block 64x64 is almost as good as matches Hadamard. On isotropic data all methods are equivalent.

**Dot-product accuracy (sym 1-4 centred):**

Hadamard's dot-product advantage (0.629 vs 0.723 on isotropic data) matches sqrt(d'/d) = 1.155 exactly, confirming the improvement is from the 33% extra storage (padding 768->1024), not better preconditioning.

## Architecture

### OSQ

Uniform-grid scalar quantizer with per-vector coordinate-descent refinement. The anisotropic loss $L = (1-\lambda)(x.e)^2/\|x\|^2 + \lambda\|e\|^2` trades MSE for dot-product accuracy ($\lambda=0.1$ in production). Key optimizations:

- **Bit-plane precomputation**: 2-bit and 4-bit vectors decompose into 1-bit packed planes at quantize time, enabling dot products via AND+popcount
- **NEON kernels**: dual-accumulator `vcntq_u8` popcount for 1-bit (7 ns), bit-plane popcount for 2-bit (14 ns), nibble multiply for 4-bit (22 ns)
- **Mixed-width scoring**: 4-bit query x 1-bit doc reduces to 4 popcount passes over precomputed planes (14 ns)
- **Centering**: subtracts corpus mean from both document and query before quantization, recovered exactly via scalar correction terms; centering the query in the asymmetric path reduces noise on shifted data by ~10x
- **Sparse preconditioner**: block-diagonal random orthogonal rotation (64x64 blocks) equalizes coordinate variances for arbitrary dimensions without power-of-2 padding

### TurboQuant

Randomized Hadamard rotation followed by Lloyd-Max optimal scalar quantization per coordinate. Provides formal MSE guarantees (within ~2.7x of information-theoretic lower bound). Inner products computed via precomputed centroid-product lookup tables. Optional QJL correction stage for unbiased inner-product estimation.

## Hadamard MSE experiment

The test applies a randomized Walsh-Hadamard transform (zero-pad 768->1024, random sign flips, WHT, normalize) before OSQ quantization. Theory predicts MSE improves by d/d' = 768/1024 = 0.75. Empirical results:

| Bits | Measured ratio (plain/hadamard) | Theory (d'/d) |
|:----:|:---:|:---:|
| 1 | 1.19 | 1.33 |
| 2 | 1.28 | 1.33 |
| 3 | 1.31 | 1.33 |
| 4 | 1.33 | 1.33 |

The shortfall at low bit-widths (1.19 vs 1.33 at 1-bit) quantifies the value of OSQ's data-dependent interval refinement — it already captures ~40% of the dimension expansion and equalization benefit that Hadamard provides.

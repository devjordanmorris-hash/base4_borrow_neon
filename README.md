# base4_borrow_neon – SIMD borrow preconditioner for base‑4 subtraction

**ARM NEON implementation** | **MIT License** | **Experimental arithmetic research**

## What it does

When you subtract two numbers in base 4 (2 bits per digit), borrows can cascade. Long borrow chains are a known bottleneck in:

- Big‑integer arithmetic (crypto, RSA, elliptic curves)
- Hardware borrow‑lookahead simulation
- Low‑level optimisation for multi‑precision subtraction

This code generates a **borrow mask** and a **residual B** such that:

`A - B = (A - residual_B) - mask`

The residual borrow chain is significantly shorter than the original, making subsequent subtraction faster in software big‑int libraries or hardware design.

## Files

- `base4_borrow_preconditioner_neon.c` – scalar + ARM NEON batch‑4 implementation
- `base4_fairbench` – compiled benchmark (macOS ARM64)

## Build & run (Apple Silicon / ARM64)

```bash
clang -O3 -mcpu=apple-m1 base4_borrow_preconditioner_neon.c -o base4_borrow_neon
./base4_borrow_neon

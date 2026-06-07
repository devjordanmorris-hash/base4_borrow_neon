/*
    base4_borrow_preconditioner_neon.c

    Experimental SIMD/NEON version of the base-4 adaptive borrow-mask generator.

    What this does:
      - Keeps the original scalar version.
      - Adds an ARM NEON batch-4 version that processes 4 independent
        A-B pairs at once.
      - Compares correctness and timings.

    Build on Apple Silicon:
      clang -O3 -mcpu=apple-m1 base4_borrow_preconditioner_neon.c -o base4_borrow_neon

    Or:
      clang -O3 -march=native base4_borrow_preconditioner_neon.c -o base4_borrow_neon

    Run:
      ./base4_borrow_neon

    Notes:
      This is still an experiment. The goal is not to beat native subtraction.
      The goal is to test whether borrow-topology/mask generation can be
      parallelized across independent values.
*/

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define HAVE_NEON 1
#else
#define HAVE_NEON 0
#endif

#define WIDTH 16
#define TRIALS 1000000u

typedef struct {
    uint32_t mask;
    uint32_t residual_b;
} MaskResult;

typedef struct {
    uint32_t borrow_count;
    uint32_t max_chain;
} BorrowStats;

static uint64_t rng_state = 0x123456789abcdefULL;

static inline uint32_t xorshift32(void) {
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng_state = x;
    return (uint32_t)(x >> 32) ^ (uint32_t)x;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static inline uint32_t get_b4_digit(uint32_t x, int pos) {
    int shift = (WIDTH - 1 - pos) * 2;
    return (x >> shift) & 3u;
}

static inline uint32_t set_b4_digit(uint32_t x, int pos, uint32_t digit) {
    int shift = (WIDTH - 1 - pos) * 2;
    uint32_t mask = 3u << shift;
    return (x & ~mask) | ((digit & 3u) << shift);
}

static BorrowStats borrow_stats_base4(uint32_t a, uint32_t b) {
    uint32_t borrow = 0;
    uint32_t count = 0;
    uint32_t chain = 0;
    uint32_t max_chain = 0;

    for (int pos = WIDTH - 1; pos >= 0; pos--) {
        uint32_t ad = get_b4_digit(a, pos);
        uint32_t bd = get_b4_digit(b, pos);
        int32_t ea = (int32_t)ad - (int32_t)borrow;

        if (ea < (int32_t)bd) {
            count++;
            chain++;
            if (chain > max_chain) max_chain = chain;
            borrow = 1;
        } else {
            chain = 0;
            borrow = 0;
        }
    }

    BorrowStats s = { count, max_chain };
    return s;
}

static MaskResult adaptive_mask_scalar(uint32_t a, uint32_t b) {
    uint32_t mask = 0;
    uint32_t residual_b = b;
    uint32_t borrow = 0;

    for (int pos = WIDTH - 1; pos >= 0; pos--) {
        uint32_t ad = get_b4_digit(a, pos);
        uint32_t bd = get_b4_digit(residual_b, pos);

        int32_t ea = (int32_t)ad - (int32_t)borrow;
        uint32_t new_bd = bd;

        if (ea < (int32_t)bd && bd > 0) {
            mask = set_b4_digit(mask, pos, 1);
            new_bd = bd - 1;
            residual_b = set_b4_digit(residual_b, pos, new_bd);
        }

        borrow = (ea < (int32_t)new_bd) ? 1u : 0u;
    }

    MaskResult r = { mask, residual_b };
    return r;
}

#if HAVE_NEON
/*
    NEON batch-4 adaptive mask.

    This vectorizes across 4 independent numbers.

    Each base-4 digit is extracted from all 4 lanes, the borrow state is
    updated per lane, and mask/residual_b are built in vector registers.

    It still has WIDTH sequential digit steps because borrow propagation
    within each number is inherently right-to-left. The parallelism comes
    from doing 4 numbers at once.
*/
static void adaptive_mask_neon4(
    const uint32_t *a_in,
    const uint32_t *b_in,
    uint32_t *mask_out,
    uint32_t *residual_b_out
) {
    uint32x4_t a = vld1q_u32(a_in);
    uint32x4_t b = vld1q_u32(b_in);

    uint32x4_t mask = vdupq_n_u32(0);
    uint32x4_t residual_b = b;
    int32x4_t borrow = vdupq_n_s32(0);

    for (int pos = WIDTH - 1; pos >= 0; pos--) {
        int shift = (WIDTH - 1 - pos) * 2;

        uint32x4_t ad_u = vandq_u32(vshlq_u32(a, vdupq_n_s32(-shift)), vdupq_n_u32(3));
        uint32x4_t bd_u = vandq_u32(vshlq_u32(residual_b, vdupq_n_s32(-shift)), vdupq_n_u32(3));

        int32x4_t ad = vreinterpretq_s32_u32(ad_u);
        int32x4_t bd = vreinterpretq_s32_u32(bd_u);

        int32x4_t ea = vsubq_s32(ad, borrow);

        uint32x4_t would_borrow = vcgtq_s32(bd, ea);
        uint32x4_t bd_gt_zero = vcgtq_s32(bd, vdupq_n_s32(0));
        uint32x4_t do_mask = vandq_u32(would_borrow, bd_gt_zero);

        /*
            do_mask is all-bits set or zero per lane.
            mask_digit is 1 or 0 per lane, shifted into base-4 position.
        */
        uint32x4_t one_or_zero = vandq_u32(do_mask, vdupq_n_u32(1));
        uint32x4_t digit_contrib = vshlq_u32(one_or_zero, vdupq_n_s32(shift));
        mask = vorrq_u32(mask, digit_contrib);

        /*
            residual_b -= 1 base-4 digit at this pos if do_mask.
            Since digit is guaranteed >0 when do_mask, no cross-digit borrow here.
        */
        uint32x4_t subtract_amount = digit_contrib;
        residual_b = vsubq_u32(residual_b, subtract_amount);

        /*
            new_bd = bd - do_mask_digit
        */
        int32x4_t new_bd = vsubq_s32(bd, vreinterpretq_s32_u32(one_or_zero));

        uint32x4_t next_borrow_u = vcgtq_s32(new_bd, ea);
        borrow = vandq_s32(vreinterpretq_s32_u32(next_borrow_u), vdupq_n_s32(1));
    }

    vst1q_u32(mask_out, mask);
    vst1q_u32(residual_b_out, residual_b);
}
#endif

static void print_base4(uint32_t x) {
    for (int i = 0; i < WIDTH; i++) {
        putchar((char)('0' + get_b4_digit(x, i)));
    }
}

int main(void) {
    printf("Base-4 Borrow Preconditioner — Scalar vs NEON batch-4\n");
    printf("WIDTH=%d base-4 digits (%d bits)\n", WIDTH, WIDTH * 2);
    printf("TRIALS=%u\n", TRIALS);
    printf("NEON available: %s\n\n", HAVE_NEON ? "yes" : "no");

    uint32_t *A = (uint32_t*)malloc(TRIALS * sizeof(uint32_t));
    uint32_t *B = (uint32_t*)malloc(TRIALS * sizeof(uint32_t));
    uint32_t *M_scalar = (uint32_t*)malloc(TRIALS * sizeof(uint32_t));
    uint32_t *R_scalar = (uint32_t*)malloc(TRIALS * sizeof(uint32_t));
    uint32_t *M_neon = (uint32_t*)malloc(TRIALS * sizeof(uint32_t));
    uint32_t *R_neon = (uint32_t*)malloc(TRIALS * sizeof(uint32_t));

    if (!A || !B || !M_scalar || !R_scalar || !M_neon || !R_neon) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    for (uint32_t i = 0; i < TRIALS; i++) {
        uint32_t a = xorshift32();
        uint32_t b = xorshift32();
        if (a < b) {
            uint32_t t = a;
            a = b;
            b = t;
        }
        A[i] = a;
        B[i] = b;
    }

    volatile uint32_t sink = 0;

    double t0 = now_sec();

    for (uint32_t i = 0; i < TRIALS; i++) {
        MaskResult r = adaptive_mask_scalar(A[i], B[i]);
        M_scalar[i] = r.mask;
        R_scalar[i] = r.residual_b;
        sink ^= r.mask ^ r.residual_b;
    }

    double t1 = now_sec();

    printf("--- Scalar mask generation ---\n");
    printf("elapsed: %.3f ms\n", (t1 - t0) * 1000.0);
    printf("rate:    %.2f M pairs/sec\n\n", (double)TRIALS / (t1 - t0) / 1e6);

#if HAVE_NEON
    double t2 = now_sec();

    uint32_t i = 0;
    for (; i + 3 < TRIALS; i += 4) {
        adaptive_mask_neon4(&A[i], &B[i], &M_neon[i], &R_neon[i]);
        sink ^= M_neon[i] ^ R_neon[i];
    }
    for (; i < TRIALS; i++) {
        MaskResult r = adaptive_mask_scalar(A[i], B[i]);
        M_neon[i] = r.mask;
        R_neon[i] = r.residual_b;
        sink ^= r.mask ^ r.residual_b;
    }

    double t3 = now_sec();

    printf("--- NEON batch-4 mask generation ---\n");
    printf("elapsed: %.3f ms\n", (t3 - t2) * 1000.0);
    printf("rate:    %.2f M pairs/sec\n", (double)TRIALS / (t3 - t2) / 1e6);
    printf("speedup: %.2fx\n\n", (t1 - t0) / (t3 - t2));

    uint64_t mismatches = 0;
    for (uint32_t j = 0; j < TRIALS; j++) {
        if (M_scalar[j] != M_neon[j] || R_scalar[j] != R_neon[j]) {
            mismatches++;
            if (mismatches <= 3) {
                printf("Mismatch at %u\n", j);
                printf("A:        "); print_base4(A[j]); puts("");
                printf("B:        "); print_base4(B[j]); puts("");
                printf("M scalar: "); print_base4(M_scalar[j]); puts("");
                printf("M neon:   "); print_base4(M_neon[j]); puts("");
                printf("R scalar: "); print_base4(R_scalar[j]); puts("");
                printf("R neon:   "); print_base4(R_neon[j]); puts("");
            }
        }
    }

    printf("--- Correctness ---\n");
    printf("NEON mismatches: %llu\n\n", (unsigned long long)mismatches);
#else
    printf("NEON not available on this compiler/target.\n\n");
#endif

    uint64_t correct = 0;
    uint64_t fewer = 0;
    uint64_t same = 0;
    uint64_t more = 0;

    uint64_t total_orig_borrows = 0;
    uint64_t total_resid_borrows = 0;
    uint64_t total_orig_chain = 0;
    uint64_t total_resid_chain = 0;

    for (uint32_t j = 0; j < TRIALS; j++) {
        uint32_t true_ans = A[j] - B[j];
        uint32_t decomposed = (A[j] - R_scalar[j]) - M_scalar[j];

        if (true_ans == decomposed) correct++;

        BorrowStats orig = borrow_stats_base4(A[j], B[j]);
        BorrowStats resid = borrow_stats_base4(A[j], R_scalar[j]);

        total_orig_borrows += orig.borrow_count;
        total_resid_borrows += resid.borrow_count;
        total_orig_chain += orig.max_chain;
        total_resid_chain += resid.max_chain;

        if (resid.borrow_count < orig.borrow_count) fewer++;
        else if (resid.borrow_count == orig.borrow_count) same++;
        else more++;
    }

    printf("--- Structural results ---\n");
    printf("correct decomposition: %.2f%%\n", 100.0 * (double)correct / (double)TRIALS);
    printf("fewer residual borrows %.2f%%\n", 100.0 * (double)fewer / (double)TRIALS);
    printf("same residual borrows  %.2f%%\n", 100.0 * (double)same / (double)TRIALS);
    printf("more residual borrows  %.2f%%\n", 100.0 * (double)more / (double)TRIALS);
    printf("avg original borrows   %.4f\n", (double)total_orig_borrows / (double)TRIALS);
    printf("avg residual borrows   %.4f\n", (double)total_resid_borrows / (double)TRIALS);
    printf("avg original max chain %.4f\n", (double)total_orig_chain / (double)TRIALS);
    printf("avg residual max chain %.4f\n", (double)total_resid_chain / (double)TRIALS);

    printf("\nsink=%u\n", sink);

    free(A);
    free(B);
    free(M_scalar);
    free(R_scalar);
    free(M_neon);
    free(R_neon);

    return 0;
}

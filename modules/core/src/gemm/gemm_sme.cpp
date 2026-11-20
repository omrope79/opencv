// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Built with the SME flags only when CMake found SME support. Everything here runs in streaming
// mode, so no NEON code and no library calls are allowed inside these functions.

#include "gemm_sme.hpp"

#if defined(CV_GEMM_HAVE_SME)
#include <arm_sme.h>

namespace cv { namespace gemm_engine { namespace sme {

__arm_locally_streaming static int vectorLengthImpl()
{
    return (int)svcntw();
}

int vectorLength()
{
    return vectorLengthImpl();
}

// Writes alpha*acc (+ beta*C) for one row of ZA through the column mask pc.
// With alpha 1 the row goes straight from ZA to memory: much faster than reading it out first.
#define CV_SME_STORE(tile, slice, ptr, pc)                                      \
    do {                                                                        \
        if (direct)                                                             \
        {                                                                       \
            svst1_hor_za32(tile, slice, pc, ptr);                               \
            break;                                                              \
        }                                                                       \
        svfloat32_t acc = svread_hor_za32_f32_m(svundef_f32(), pt, tile, slice); \
        if (beta == 0.f)                                                        \
            svst1_f32(pc, ptr, svmul_f32_x(pt, acc, va));                       \
        else                                                                    \
        {                                                                       \
            svfloat32_t cv = svld1_f32(pc, ptr);                                \
            if (beta != 1.f)                                                    \
                cv = svmul_f32_x(pt, cv, vb);                                   \
            svst1_f32(pc, ptr, svmla_f32_x(pt, cv, acc, va));                   \
        }                                                                       \
    } while (0)

// For beta 1 with a direct store, C is loaded into ZA instead of being added on the way out.
#define CV_SME_LOAD(tile, slice, ptr, pc) svld1_hor_za32(tile, slice, pc, ptr)

// 2VL x 2VL tiles: A panels of 2VL rows times B panels of 2VL columns, all four ZA tiles.
__arm_new("za") __arm_locally_streaming
static void kernel2x2(int m, int n, int k, const float* A, size_t aps, const float* B, size_t bps,
                      float alpha, float beta, float* C, size_t ldc)
{
    const int vl = (int)svcntw(), W = 2 * vl;
    const svbool_t pt = svptrue_b32();
    const svfloat32_t va = svdup_f32(alpha), vb = svdup_f32(beta);
    const bool direct = alpha == 1.f && (beta == 0.f || beta == 1.f);
    for (int j = 0; j < n; j += W)
    {
        const float* bp = B + (size_t)(j / W) * bps;
        const svbool_t pc0 = svwhilelt_b32(j, n), pc1 = svwhilelt_b32(j + vl, n);
        for (int i = 0; i < m; i += W)
        {
            const float* a = A + (size_t)(i / W) * aps;
            const float* b = bp;
            const int rows = m - i < W ? m - i : W;
            svzero_za();
            if (direct && beta != 0.f)
            {
                for (int r = 0; r < rows; r++)
                {
                    const float* c = C + (size_t)(i + r) * ldc + j;
                    if (r < vl)
                    {
                        CV_SME_LOAD(0, r, c, pc0);
                        CV_SME_LOAD(1, r, c + vl, pc1);
                    }
                    else
                    {
                        CV_SME_LOAD(2, r - vl, c, pc0);
                        CV_SME_LOAD(3, r - vl, c + vl, pc1);
                    }
                }
            }
            for (int p = 0; p < k; p++, a += W, b += W)
            {
                svfloat32_t a0 = svld1_f32(pt, a), a1 = svld1_f32(pt, a + vl);
                svfloat32_t b0 = svld1_f32(pt, b), b1 = svld1_f32(pt, b + vl);
                svmopa_za32_f32_m(0, pt, pt, a0, b0);
                svmopa_za32_f32_m(1, pt, pt, a0, b1);
                svmopa_za32_f32_m(2, pt, pt, a1, b0);
                svmopa_za32_f32_m(3, pt, pt, a1, b1);
            }
            for (int r = 0; r < rows; r++)
            {
                float* c = C + (size_t)(i + r) * ldc + j;
                if (r < vl)
                {
                    CV_SME_STORE(0, r, c, pc0);
                    CV_SME_STORE(1, r, c + vl, pc1);
                }
                else
                {
                    CV_SME_STORE(2, r - vl, c, pc0);
                    CV_SME_STORE(3, r - vl, c + vl, pc1);
                }
            }
        }
    }
}

// VL x 4VL tiles for short A: one A panel of VL rows times two B panels.
__arm_new("za") __arm_locally_streaming
static void kernel1x4(int m, int n, int k, const float* A, size_t aps, const float* B, size_t bps,
                      float alpha, float beta, float* C, size_t ldc)
{
    const int vl = (int)svcntw(), W = 2 * vl;
    const svbool_t pt = svptrue_b32();
    const svfloat32_t va = svdup_f32(alpha), vb = svdup_f32(beta);
    const bool direct = alpha == 1.f && (beta == 0.f || beta == 1.f);
    for (int j = 0; j < n; j += 2 * W)
    {
        const float* bp0 = B + (size_t)(j / W) * bps;
        const float* bp1 = j + W < n ? bp0 + bps : bp0;  // no second panel: reuse and mask it out
        const svbool_t pc0 = svwhilelt_b32(j, n), pc1 = svwhilelt_b32(j + vl, n);
        const svbool_t pc2 = svwhilelt_b32(j + W, n), pc3 = svwhilelt_b32(j + W + vl, n);
        for (int i = 0; i < m; i += vl)
        {
            const float* a = A + (size_t)(i / vl) * aps;
            const float* b0p = bp0;
            const float* b1p = bp1;
            const int rows = m - i < vl ? m - i : vl;
            svzero_za();
            if (direct && beta != 0.f)
            {
                for (int r = 0; r < rows; r++)
                {
                    const float* c = C + (size_t)(i + r) * ldc + j;
                    CV_SME_LOAD(0, r, c, pc0);
                    CV_SME_LOAD(1, r, c + vl, pc1);
                    CV_SME_LOAD(2, r, c + W, pc2);
                    CV_SME_LOAD(3, r, c + W + vl, pc3);
                }
            }
            for (int p = 0; p < k; p++, a += vl, b0p += W, b1p += W)
            {
                svfloat32_t a0 = svld1_f32(pt, a);
                svmopa_za32_f32_m(0, pt, pt, a0, svld1_f32(pt, b0p));
                svmopa_za32_f32_m(1, pt, pt, a0, svld1_f32(pt, b0p + vl));
                svmopa_za32_f32_m(2, pt, pt, a0, svld1_f32(pt, b1p));
                svmopa_za32_f32_m(3, pt, pt, a0, svld1_f32(pt, b1p + vl));
            }
            for (int r = 0; r < rows; r++)
            {
                float* c = C + (size_t)(i + r) * ldc + j;
                CV_SME_STORE(0, r, c, pc0);
                CV_SME_STORE(1, r, c + vl, pc1);
                CV_SME_STORE(2, r, c + W, pc2);
                CV_SME_STORE(3, r, c + W + vl, pc3);
            }
        }
    }
}
#undef CV_SME_STORE
#undef CV_SME_LOAD

#if defined(CV_GEMM_HAVE_SME2)
// SME2 versions of the two kernels above, same layouts and results. They load A and B with one
// two-register load and read ZA four rows at a time; with one ZA row per move, the copy-out after
// each tile took longer than the outer products when k is a few hundred.
#if defined(__clang__)
#define CV_SME2_TARGET __attribute__((target("sme2")))
#else
#define CV_SME2_TARGET __attribute__((target("+sme2")))
#endif

// Writes alpha*row (+ beta*C) for one output row through the column mask pc.
#define CV_SME2_ROW(row, ptr, pc)                                               \
    do {                                                                        \
        if (beta == 0.f)                                                        \
            svst1_f32(pc, ptr, svmul_f32_x(pt, row, va));                       \
        else                                                                    \
        {                                                                       \
            svfloat32_t cv = svld1_f32(pc, ptr);                                \
            if (beta != 1.f)                                                    \
                cv = svmul_f32_x(pt, cv, vb);                                   \
            svst1_f32(pc, ptr, svmla_f32_x(pt, cv, row, va));                   \
        }                                                                       \
    } while (0)

// Copies rows [r0, r0 + rn) of the tile pair (T0 | T1) to C rows starting at c, four at a time.
#define CV_SME2_STORE_PAIR(T0, T1, c, rn)                                       \
    do {                                                                        \
        int r = 0;                                                              \
        for (; r + 4 <= (rn); r += 4)                                           \
        {                                                                       \
            const svfloat32x4_t x0 = svread_hor_za32_f32_vg4(T0, r);           \
            const svfloat32x4_t x1 = svread_hor_za32_f32_vg4(T1, r);           \
            float* c0 = (c) + (size_t)r * ldc;                                  \
            CV_SME2_ROW(svget4(x0, 0), c0, pc0);                                \
            CV_SME2_ROW(svget4(x1, 0), c0 + vl, pc1);                           \
            CV_SME2_ROW(svget4(x0, 1), c0 + ldc, pc0);                          \
            CV_SME2_ROW(svget4(x1, 1), c0 + ldc + vl, pc1);                     \
            CV_SME2_ROW(svget4(x0, 2), c0 + 2 * ldc, pc0);                      \
            CV_SME2_ROW(svget4(x1, 2), c0 + 2 * ldc + vl, pc1);                 \
            CV_SME2_ROW(svget4(x0, 3), c0 + 3 * ldc, pc0);                      \
            CV_SME2_ROW(svget4(x1, 3), c0 + 3 * ldc + vl, pc1);                 \
        }                                                                       \
        for (; r < (rn); r++)                                                   \
        {                                                                       \
            float* c0 = (c) + (size_t)r * ldc;                                  \
            CV_SME2_ROW(svread_hor_za32_f32_m(svundef_f32(), pt, T0, r), c0, pc0);      \
            CV_SME2_ROW(svread_hor_za32_f32_m(svundef_f32(), pt, T1, r), c0 + vl, pc1); \
        }                                                                       \
    } while (0)

__arm_new("za") __arm_locally_streaming CV_SME2_TARGET
static void kernel2x2_sme2(int m, int n, int k, const float* A, size_t aps, const float* B, size_t bps,
                           float alpha, float beta, float* C, size_t ldc)
{
    const int vl = (int)svcntw(), W = 2 * vl;
    const svbool_t pt = svptrue_b32();
    const svcount_t pn = svptrue_c32();
    const svfloat32_t va = svdup_f32(alpha), vb = svdup_f32(beta);
    for (int j = 0; j < n; j += W)
    {
        const float* bp = B + (size_t)(j / W) * bps;
        const svbool_t pc0 = svwhilelt_b32(j, n), pc1 = svwhilelt_b32(j + vl, n);
        for (int i = 0; i < m; i += W)
        {
            const float* a = A + (size_t)(i / W) * aps;
            const float* b = bp;
            svzero_za();
            for (int p = 0; p < k; p++, a += W, b += W)
            {
                const svfloat32x2_t aa = svld1_f32_x2(pn, a), bb = svld1_f32_x2(pn, b);
                const svfloat32_t a0 = svget2(aa, 0), a1 = svget2(aa, 1);
                const svfloat32_t b0 = svget2(bb, 0), b1 = svget2(bb, 1);
                svmopa_za32_f32_m(0, pt, pt, a0, b0);
                svmopa_za32_f32_m(1, pt, pt, a0, b1);
                svmopa_za32_f32_m(2, pt, pt, a1, b0);
                svmopa_za32_f32_m(3, pt, pt, a1, b1);
            }
            const int rows = m - i < W ? m - i : W;
            float* c = C + (size_t)i * ldc + j;
            CV_SME2_STORE_PAIR(0, 1, c, rows < vl ? rows : vl);
            if (rows > vl)
                CV_SME2_STORE_PAIR(2, 3, c + (size_t)vl * ldc, rows - vl);
        }
    }
}

__arm_new("za") __arm_locally_streaming CV_SME2_TARGET
static void kernel1x4_sme2(int m, int n, int k, const float* A, size_t aps, const float* B, size_t bps,
                           float alpha, float beta, float* C, size_t ldc)
{
    const int vl = (int)svcntw(), W = 2 * vl;
    const svbool_t pt = svptrue_b32();
    const svcount_t pn = svptrue_c32();
    const svfloat32_t va = svdup_f32(alpha), vb = svdup_f32(beta);
    for (int j = 0; j < n; j += 2 * W)
    {
        const float* bp0 = B + (size_t)(j / W) * bps;
        const float* bp1 = j + W < n ? bp0 + bps : bp0;  // no second panel: reuse and mask it out
        for (int i = 0; i < m; i += vl)
        {
            const float* a = A + (size_t)(i / vl) * aps;
            const float* b0p = bp0;
            const float* b1p = bp1;
            svzero_za();
            for (int p = 0; p < k; p++, a += vl, b0p += W, b1p += W)
            {
                const svfloat32_t a0 = svld1_f32(pt, a);
                const svfloat32x2_t b0 = svld1_f32_x2(pn, b0p), b1 = svld1_f32_x2(pn, b1p);
                svmopa_za32_f32_m(0, pt, pt, a0, svget2(b0, 0));
                svmopa_za32_f32_m(1, pt, pt, a0, svget2(b0, 1));
                svmopa_za32_f32_m(2, pt, pt, a0, svget2(b1, 0));
                svmopa_za32_f32_m(3, pt, pt, a0, svget2(b1, 1));
            }
            const int rows = m - i < vl ? m - i : vl;
            float* c = C + (size_t)i * ldc + j;
            {
                const svbool_t pc0 = svwhilelt_b32(j, n), pc1 = svwhilelt_b32(j + vl, n);
                CV_SME2_STORE_PAIR(0, 1, c, rows);
            }
            if (j + W < n)
            {
                const svbool_t pc0 = svwhilelt_b32(j + W, n), pc1 = svwhilelt_b32(j + W + vl, n);
                CV_SME2_STORE_PAIR(2, 3, c + W, rows);
            }
        }
    }
}
#undef CV_SME2_STORE_PAIR
#undef CV_SME2_ROW
#undef CV_SME2_TARGET
#endif // CV_GEMM_HAVE_SME2

void gemmPanels(int m, int n, int k, int R, const float* A, size_t aPanelStride,
                const float* B, size_t bPanelStride, float alpha, float beta, float* C, size_t ldc, bool sme2)
{
#if defined(CV_GEMM_HAVE_SME2)
    if (sme2)
    {
        if (R == vectorLength())
            kernel1x4_sme2(m, n, k, A, aPanelStride, B, bPanelStride, alpha, beta, C, ldc);
        else
            kernel2x2_sme2(m, n, k, A, aPanelStride, B, bPanelStride, alpha, beta, C, ldc);
        return;
    }
#else
    (void)sme2;
#endif
    if (R == vectorLength())
        kernel1x4(m, n, k, A, aPanelStride, B, bPanelStride, alpha, beta, C, ldc);
    else
        kernel2x2(m, n, k, A, aPanelStride, B, bPanelStride, alpha, beta, C, ldc);
}

}}} // cv::gemm_engine::sme

#endif // CV_GEMM_HAVE_SME

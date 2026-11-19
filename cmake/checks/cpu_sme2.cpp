#include <stdio.h>

#if defined(__ARM_FEATURE_SME) && defined(__aarch64__)
#  include <arm_sme.h>
#  if defined(__clang__)
#    define CV_SME2_TARGET __attribute__((target("sme2")))
#  else
#    define CV_SME2_TARGET __attribute__((target("+sme2")))
#  endif
#else
#error "SME is not supported"
#endif

// SME2 multi-vector load and 4-slice ZA read in a function built for SME2 only,
// the rest of the file keeping the plain SME flags.
__arm_new("za") __arm_locally_streaming CV_SME2_TARGET
static float test_sme2(const float* a, float* c)
{
    const svbool_t pg = svptrue_b32();
    const svfloat32x2_t v = svld1_f32_x2(svptrue_c32(), a);
    svzero_za();
    svmopa_za32_f32_m(0, pg, pg, svget2(v, 0), svget2(v, 1));
    const svfloat32x4_t r = svread_hor_za32_f32_vg4(0, 0);
    svst1_f32(pg, c, svget4(r, 0));
    return c[0];
}

int main()
{
    float a[128] = { 1.f }, c[64] = { 0.f };
    printf("%f\n", test_sme2(a, c));
    return 0;
}

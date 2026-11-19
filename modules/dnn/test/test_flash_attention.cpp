// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"
#include <opencv2/dnn/all_layers.hpp>

namespace opencv_test { namespace {

// Plain attention in double: Y[b, i, h*Dv + :] = softmax(Q K^T / scale) V, with K given transposed.
static Mat referenceSDPA(const Mat& Q, const Mat& KT, const Mat& V, double scale)
{
    const int B = Q.size[0], H = Q.size[1], Sq = Q.size[2], D = Q.size[3];
    const int Skv = V.size[2], Dv = V.size[3];
    Mat Y(std::vector<int>{B, Sq, H * Dv}, CV_32F);
    std::vector<double> s(Skv);
    for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++)
            for (int i = 0; i < Sq; i++)
            {
                const float* q = Q.ptr<float>(b, h) + (size_t)i * D;
                const float* kt = KT.ptr<float>(b, h);
                const float* v = V.ptr<float>(b, h);
                double mx = -DBL_MAX, sum = 0;
                for (int j = 0; j < Skv; j++)
                {
                    double d = 0;
                    for (int c = 0; c < D; c++)
                        d += (double)q[c] * kt[(size_t)c * Skv + j];
                    s[j] = d / scale;
                    mx = std::max(mx, s[j]);
                }
                for (int j = 0; j < Skv; j++)
                    sum += (s[j] = std::exp(s[j] - mx));
                float* y = Y.ptr<float>(b, i) + h * Dv;
                for (int c = 0; c < Dv; c++)
                {
                    double acc = 0;
                    for (int j = 0; j < Skv; j++)
                        acc += s[j] * v[(size_t)j * Dv + c];
                    y[c] = (float)(acc / sum);
                }
            }
    return Y;
}

// B, H, Sq, Skv, D, Dv, input range: several Q and K/V blocks (256), Dv != D and large scores.
typedef testing::TestWithParam<tuple<int, int, int, int, int, int, float> > Test_SDPA_Accuracy;
TEST_P(Test_SDPA_Accuracy, Accuracy)
{
    const int B = get<0>(GetParam()), H = get<1>(GetParam()), Sq = get<2>(GetParam()), Skv = get<3>(GetParam());
    const int D = get<4>(GetParam()), Dv = get<5>(GetParam());
    const float range = get<6>(GetParam());
    Mat Q(std::vector<int>{B, H, Sq, D}, CV_32F), KT(std::vector<int>{B, H, D, Skv}, CV_32F);
    Mat V(std::vector<int>{B, H, Skv, Dv}, CV_32F);
    RNG& rng = theRNG();
    rng.fill(Q, RNG::UNIFORM, -range, range);
    rng.fill(KT, RNG::UNIFORM, -range, range);
    rng.fill(V, RNG::UNIFORM, -1.f, 1.f);
    const double scale = std::sqrt((double)D);

    LayerParams lp;
    lp.type = "SDPA";
    lp.name = "sdpa";
    lp.set("scale", (float)scale);
    Net net;
    const int id = net.addLayer(lp.name, lp.type, lp);
    net.connect(0, 0, id, 0);
    net.connect(0, 1, id, 1);
    net.connect(0, 2, id, 2);
    net.setInputsNames({"q", "kt", "v"});
    net.setInput(Q, "q");
    net.setInput(KT, "kt");
    net.setInput(V, "v");
    Mat out = net.forward();

    Mat ref = referenceSDPA(Q, KT, V, scale);
    ASSERT_EQ(ref.total(), out.total());
    EXPECT_LE(cvtest::norm(ref.reshape(1, 1), out.reshape(1, 1), NORM_INF), 2e-5);
}
INSTANTIATE_TEST_CASE_P(/**/, Test_SDPA_Accuracy, testing::Values(
    make_tuple(1, 2, 7, 5, 16, 16, 1.f),
    make_tuple(2, 3, 50, 50, 64, 64, 1.f),
    make_tuple(1, 2, 300, 600, 48, 40, 1.f),
    make_tuple(2, 2, 257, 513, 32, 24, 1.f),
    make_tuple(1, 2, 64, 700, 64, 32, 6.f)));

// Attention layer (projection + attention) in double with an additive mask broadcast to [B, H, S, S].
static Mat referenceAttention(const Mat& X, const Mat& W, const Mat& bias, const Mat& mask, int H, int Dqk, int Dv)
{
    const int B = X.size[0], S = X.size[1], C = X.size[2], N = W.cols;
    const int mdims = mask.dims;
    auto maskAt = [&](int b, int h, int i, int j) -> double {
        const int idx[4] = {b, h, i, j};
        size_t ofs = 0;
        for (int d = 0; d < mdims; d++)
        {
            const int t = 4 - mdims + d, n = mask.size[d];
            ofs = ofs * n + (n == 1 ? 0 : idx[t]);
        }
        return ((const float*)mask.data)[ofs];
    };
    std::vector<double> P((size_t)B * S * N);  // projected Q|K|V per token
    for (int t = 0; t < B * S; t++)
        for (int n = 0; n < N; n++)
        {
            double acc = bias.ptr<float>()[n];
            for (int c = 0; c < C; c++)
                acc += (double)X.ptr<float>()[(size_t)t * C + c] * W.at<float>(c, n);
            P[(size_t)t * N + n] = acc;
        }
    Mat Y(std::vector<int>{B, S, H * Dv}, CV_32F);
    std::vector<double> s(S);
    const double scale = 1.0 / std::sqrt((double)Dqk);
    for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++)
            for (int i = 0; i < S; i++)
            {
                const double* q = &P[((size_t)b * S + i) * N + h * Dqk];
                double mx = -DBL_MAX, sum = 0;
                for (int j = 0; j < S; j++)
                {
                    const double* k = &P[((size_t)b * S + j) * N + H * Dqk + h * Dqk];
                    double d = 0;
                    for (int c = 0; c < Dqk; c++)
                        d += q[c] * k[c];
                    s[j] = d * scale + maskAt(b, h, i, j);
                    mx = std::max(mx, s[j]);
                }
                for (int j = 0; j < S; j++)
                    sum += (s[j] = std::exp(s[j] - mx));
                float* y = Y.ptr<float>(b, i) + h * Dv;
                for (int c = 0; c < Dv; c++)
                {
                    double acc = 0;
                    for (int j = 0; j < S; j++)
                        acc += s[j] * P[((size_t)b * S + j) * N + 2 * H * Dqk + h * Dv + c];
                    y[c] = (float)(acc / sum);
                }
            }
    return Y;
}

// B, S, mask shape kind: 0 = [B, 1, 1, S] (BERT padding), 1 = [B, H, S, S], 2 = [S, S], 3 = [1, S].
// S = 300 spans two 256-row Q and K/V tiles.
typedef testing::TestWithParam<tuple<int, int, int> > Test_Attention_Mask;
TEST_P(Test_Attention_Mask, Accuracy)
{
    const int B = get<0>(GetParam()), S = get<1>(GetParam()), kind = get<2>(GetParam());
    const int H = 3, Dqk = 16, Dv = 8, C = 24;
    int hidden[] = {H * Dqk, H * Dqk, H * Dv};
    Mat W(C, 2 * H * Dqk + H * Dv, CV_32F), bias(MatShape{W.cols}, CV_32F);
    Mat X(std::vector<int>{B, S, C}, CV_32F);
    RNG& rng = theRNG();
    rng.fill(W, RNG::UNIFORM, -.5f, .5f);
    rng.fill(bias, RNG::UNIFORM, -.2f, .2f);
    rng.fill(X, RNG::UNIFORM, -1.f, 1.f);
    const std::vector<int> mshape = kind == 0 ? std::vector<int>{B, 1, 1, S} :
                                    kind == 1 ? std::vector<int>{B, H, S, S} :
                                    kind == 2 ? std::vector<int>{S, S} : std::vector<int>{1, S};
    Mat mask(mshape, CV_32F);
    rng.fill(mask, RNG::UNIFORM, -2.f, 2.f);
    // Masked-out keys the way exported models write them, including the last keys of the sequence.
    float* m = mask.ptr<float>();
    const size_t mrow = mshape.back();
    for (size_t i = 0; i < mask.total(); i++)
        if (i % mrow >= mrow - mrow / 4 || i % 7 == 3)
            m[i] = -10000.f;

    LayerParams lp;
    lp.set("num_heads", H);
    lp.set("qkv_hidden_sizes", DictValue::arrayInt(hidden, 3));
    lp.blobs = {W, bias};
    std::vector<Mat> inps{X, mask}, outs;
    runLayer(AttentionLayer::create(lp), inps, outs);

    Mat ref = referenceAttention(X, W, bias, mask, H, Dqk, Dv);
    ASSERT_EQ(ref.total(), outs[0].total());
    EXPECT_LE(cvtest::norm(ref.reshape(1, 1), outs[0].reshape(1, 1), NORM_INF), 2e-5);
}
INSTANTIATE_TEST_CASE_P(/**/, Test_Attention_Mask, testing::Combine(
    testing::Values(1, 2), testing::Values(7, 64, 300), testing::Values(0, 1, 2, 3)));

}} // namespace

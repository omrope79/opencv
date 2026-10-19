// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "cpu_kernels/mlas_gemm.hpp"

namespace cv { namespace dnn {

// Scaled dot-product attention: Q (B,H,Sq,D), K^T (B,H,D,Skv), V (B,H,Skv,Dv) -> Y (B,Sq,H*Dv), one MlasFlashAttention call.
// Batch and heads broadcast like the ONNX MatMuls this replaces: any of Q, K, V may have 1 there (multi-query, shared K/V).
// Params: `scale` divides Q*K^T (default 1); `head_dim_scale` divides by sqrt(D) instead, for graphs that compute it at run time.
// Created by the ONNX importer for an SDPA node and by the attention fusion pass for blocks whose Q, K and V are already split into heads.
class SDPALayerImpl CV_FINAL : public SDPALayer {
public:
    SDPALayerImpl(const LayerParams &params) {
        setParamsFrom(params);
        scale = params.get<float>("scale", 1.f);
        head_dim_scale = params.get<bool>("head_dim_scale", false);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE {
        return backendId == DNN_BACKEND_OPENCV;
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE {
        CV_CheckEQ(inputs.size(), 3u, "DNN/SDPA: expects Q, K^T, V");
        const auto &q  = inputs[0];   // (B, H, S_q,  D)
        const auto &kT = inputs[1];   // (B, H, D,    S_kv)
        const auto &v  = inputs[2];   // (B, H, S_kv, D_v)
        CV_CheckEQ(q.dims, 4, "DNN/SDPA: Q must be 4D");
        CV_CheckEQ(kT.dims, 4, "DNN/SDPA: K must be 4D");
        CV_CheckEQ(v.dims, 4, "DNN/SDPA: V must be 4D");
        const int S_q = q[2], D = q[3];
        const int S_kv = v[2], D_v = v[3];
        CV_CheckEQ(kT[2], D,    "DNN/SDPA: K^T inner dim must equal Q's head_dim");
        CV_CheckEQ(kT[3], S_kv, "DNN/SDPA: K^T outer dim must equal V's seq len");
        // Batch and heads broadcast as in the MatMuls this layer replaces: each is 1 or the common size.
        const int B = broadcastDim(q[0], kT[0], v[0]);
        const int H = broadcastDim(q[1], kT[1], v[1]);
        CV_CheckGT(B, 0, "DNN/SDPA: Q, K and V batch dims do not broadcast");
        CV_CheckGT(H, 0, "DNN/SDPA: Q, K and V head dims do not broadcast");

        outputs.assign(1, MatShape{B, S_q, H * D_v});
        // Scratch for K^T re-transposed to (B, H, S_kv, D), and for V broadcast to (B, H, S_kv, D_v) when it is not full size.
        const bool v_full = v[0] == B && v[1] == H;
        internals.assign(1, MatShape{B * H * S_kv * D});
        internals.push_back(MatShape{v_full ? 1 : B * H * S_kv * D_v});
        return false;
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const CV_OVERRIDE {
        const auto &q = inputs[0];
        const auto &kT = inputs[1];
        const auto &v = inputs[2];
        const int64 B = std::max(broadcastDim(q[0], kT[0], v[0]), 1);
        const int64 H = std::max(broadcastDim(q[1], kT[1], v[1]), 1);
        const int64 S_q = q[2], D = q[3];
        const int64 S_kv = v[2], D_v = v[3];
        return B * H * (CV_BIG_INT(2) * S_q * S_kv * D + 4 * S_q * S_kv
                        + CV_BIG_INT(2) * S_q * S_kv * D_v);
    }

    void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        if (inputs_arr.depth() == CV_16F) {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs, internals;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
        internals_arr.getMatVector(internals);

        const Mat& Q  = inputs[0];   // (Bq, Hq, S_q,  D)
        const Mat& KT = inputs[1];   // (Bk, Hk, D,    S_kv)
        const Mat& V  = inputs[2];   // (Bv, Hv, S_kv, D_v)
        Mat&       Y  = outputs[0];  // (B, S_q, H*D_v)

        const int Bq = Q.size[0],  Hq = Q.size[1];
        const int Bk = KT.size[0], Hk = KT.size[1];
        const int Bv = V.size[0],  Hv = V.size[1];
        const int B  = std::max(Bq, std::max(Bk, Bv));
        const int H  = std::max(Hq, std::max(Hk, Hv));
        const int S_q = Q.size[2];
        const int D   = Q.size[3];
        const int S_kv = V.size[2];
        const int D_v  = V.size[3];

        // Re-transpose K^T (Bk,Hk,D,S_kv) -> K (B,H,S_kv,D) for MlasFlashAttention, broadcasting a size-1 batch or head dim.
        Mat &K = internals[0];
        const float* kT_data = KT.ptr<const float>();
        float*       k_data  = K.ptr<float>();
        const size_t kT_bhd_stride = (size_t)D * S_kv;
        const size_t k_bhd_stride  = (size_t)S_kv * D;
        cv::parallel_for_(cv::Range(0, B * H), [&](const cv::Range& r) {
            for (int bh = r.start; bh < r.end; bh++) {
                const float* src = kT_data + srcBlock(bh, H, Bk, Hk) * kT_bhd_stride;
                float*       dst = k_data  + (size_t)bh * k_bhd_stride;
                for (int s = 0; s < S_kv; s++) {
                    for (int d = 0; d < D; d++) {
                        dst[(size_t)s * D + d] = src[(size_t)d * S_kv + s];
                    }
                }
            }
        });

        // V is read in place when full size, else broadcast into scratch.
        const float* v_data = V.ptr<const float>();
        if (Bv != B || Hv != H) {
            float* vb = internals[1].ptr<float>();
            const size_t v_bhd_stride = (size_t)S_kv * D_v;
            cv::parallel_for_(cv::Range(0, B * H), [&](const cv::Range& r) {
                for (int bh = r.start; bh < r.end; bh++)
                    std::memcpy(vb + (size_t)bh * v_bhd_stride, v_data + srcBlock(bh, H, Bv, Hv) * v_bhd_stride,
                                v_bhd_stride * sizeof(float));
            });
            v_data = vb;
        }

        const int q_block  = std::min(256, S_q);
        const int kv_block = std::min(256, S_kv);
        const int threads  = std::max(1, cv::getNumThreads());

        const size_t per_thread =
            mlasFlashAttentionBufferBytesPerThread(q_block, kv_block, D_v);
        // A broadcast Q goes to the fallback, which reads Q per output block.
        if (Bq == B && Hq == H && mlasAvailable() && per_thread > 0) {
            flash_scratch.resize((size_t)threads * per_thread);
            const float effective_scale = effectiveScale(D);
            if (mlasFlashAttention(Q.ptr<float>(), K.ptr<float>(), v_data,
                                   Y.ptr<float>(),
                                   B, H, S_q, S_kv, D, D_v, effective_scale,
                                   q_block, kv_block,
                                   flash_scratch.data(), threads))
            {
                return;
            }
        }

        forward_fallback_impl(Q.ptr<const float>(), Bq, Hq, K.ptr<const float>(), v_data, Y.ptr<float>(),
                              B, H, S_q, S_kv, D, D_v);
    }

private:
    // ONNX MatMul broadcast of one batch-like dim over Q, K^T and V: the common size, or -1 when they disagree.
    static int broadcastDim(int q, int k, int v)
    {
        const int d = std::max(q, std::max(k, v));
        for (int x : {q, k, v})
            if (x != 1 && x != d) return -1;
        return d;
    }

    // Block of an input with batch Bx and heads Hx that feeds output block bh; a size-1 dim is broadcast.
    static size_t srcBlock(int bh, int H, int Bx, int Hx)
    {
        const int b = bh / H, h = bh % H;
        return (size_t)(Bx == 1 ? 0 : b) * Hx + (Hx == 1 ? 0 : h);
    }

    // Multiplier on Q*K^T: 1/sqrt(head_dim) when the graph computed it at run time, else 1/scale.
    float effectiveScale(int D) const
    {
        if (head_dim_scale) return 1.f / std::sqrt((float)D);
        return (scale > 0.f) ? (1.f / scale) : 1.f;
    }

    // Unfused SDPA, used when MLAS isn't available or Q is broadcast. Not perf-critical.
    // K and V are full (B, H) blocks; Q has batch Bq and heads Hq and is broadcast per output block.
    void forward_fallback_impl(const float* q_p, int Bq, int Hq, const float* k_p, const float* v_p, float* y_p,
                               int B, int H, int S_q, int S_kv, int D, int D_v) const
    {
        const float inv_scale = effectiveScale(D);

        cv::parallel_for_(cv::Range(0, B * H), [&](const cv::Range& r) {
            std::vector<float> attn((size_t)S_q * S_kv);
            for (int bh = r.start; bh < r.end; bh++) {
                const int b = bh / H, h = bh % H;
                const float* Q_bh = q_p + srcBlock(bh, H, Bq, Hq) * S_q * D;
                const float* K_bh = k_p + (size_t)bh * S_kv * D;
                const float* V_bh = v_p + (size_t)bh * S_kv * D_v;
                float*       Y_b  = y_p + (size_t)b * S_q * H * D_v;

                for (int i = 0; i < S_q; i++) {
                    for (int j = 0; j < S_kv; j++) {
                        float s = 0.f;
                        for (int d = 0; d < D; d++)
                            s += Q_bh[(size_t)i * D + d] * K_bh[(size_t)j * D + d];
                        attn[(size_t)i * S_kv + j] = s * inv_scale;
                    }
                    float mx = attn[(size_t)i * S_kv];
                    for (int j = 1; j < S_kv; j++)
                        if (attn[(size_t)i * S_kv + j] > mx) mx = attn[(size_t)i * S_kv + j];
                    float sum = 0.f;
                    for (int j = 0; j < S_kv; j++) {
                        float e = std::exp(attn[(size_t)i * S_kv + j] - mx);
                        attn[(size_t)i * S_kv + j] = e;
                        sum += e;
                    }
                    float inv_sum = 1.f / sum;
                    for (int j = 0; j < S_kv; j++)
                        attn[(size_t)i * S_kv + j] *= inv_sum;

                    // out[b, i, h, :] = attn[i, :] @ V_bh
                    float* out_row = Y_b + ((size_t)i * H + h) * D_v;
                    for (int d = 0; d < D_v; d++) out_row[d] = 0.f;
                    for (int j = 0; j < S_kv; j++) {
                        float a = attn[(size_t)i * S_kv + j];
                        const float* V_row = V_bh + (size_t)j * D_v;
                        for (int d = 0; d < D_v; d++)
                            out_row[d] += a * V_row[d];
                    }
                }
            }
        });
    }

    float scale;
    bool head_dim_scale;
    std::vector<unsigned char> flash_scratch;
};

Ptr<SDPALayer> SDPALayer::create(const LayerParams &params) {
    return makePtr<SDPALayerImpl>(params);
}

}} // cv::dnn

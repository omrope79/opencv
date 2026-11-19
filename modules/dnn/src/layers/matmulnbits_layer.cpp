// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"

#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/core/hal/intrin.hpp>

// com.microsoft MatMulNBits (contrib op, since com.microsoft opset 1):
// https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftmatmulnbits
// Y = A @ dequant(B).T. Weights are block-quantized to {4,8} bits (symmetric, default
// zero-point 2^(bits-1)) and dequantized per block inside the GEMM (no fp32 copy).

namespace cv { namespace dnn {

class MatMulNBitsLayerImpl CV_FINAL : public MatMulNBitsLayer {
 public:
    MatMulNBitsLayerImpl(const LayerParams& params) {
        setParamsFrom(params);

        K = params.get<int>("K");
        N = params.get<int>("N");
        bits = params.get<int>("bits", 4);
        block_size = params.get<int>("block_size");

        CV_Check(bits, bits == 4 || bits == 8,
                 "DNN/MatMulNBits: only 4/8-bit weights are supported");
        CV_Check(block_size, block_size > 0 && (block_size * bits) % 8 == 0,
                 "DNN/MatMulNBits: block_size*bits must be a positive multiple of 8");
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE {
        return backendId == DNN_BACKEND_OPENCV;
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE {
        CV_CheckEQ(inputs.size(), static_cast<size_t>(1),
                   "DNN/MatMulNBits: expected a single activation input; weights and scales must be constants");
        CV_CheckEQ(blobs.size(), static_cast<size_t>(2),
                   "DNN/MatMulNBits: expected packed weights and scales as constant blobs");

        const int n_blk = (K + block_size - 1) / block_size;   // ceil: K need not divide block_size; last block may be partial
        CV_CheckEQ(static_cast<int>(blobs[0].total()), N * n_blk * (block_size * bits / 8),
                   "DNN/MatMulNBits: packed weight size mismatch");
        CV_CheckEQ(static_cast<int>(blobs[1].total()), N * n_blk,
                   "DNN/MatMulNBits: scales size mismatch");
        CV_CheckType(blobs[0].type(), blobs[0].type() == CV_8U,
                     "DNN/MatMulNBits: packed weights must be CV_8U; int32-packed weights are not supported");
        CV_CheckType(blobs[1].type(), blobs[1].type() == CV_32F,
                     "DNN/MatMulNBits: scales must be CV_32F");

        MatShape shape_A = inputs[0];
        CV_CheckGE(shape_A.size(), static_cast<size_t>(1), "DNN/MatMulNBits: invalid shape of input A");
        CV_CheckEQ(shape_A.back(), K, "DNN/MatMulNBits: last dimension of A must equal K");

        MatShape out = shape_A;
        out.back() = N;
        outputs.assign(1, out);
        internals.clear();
        return false;
    }

    void getTypes(const std::vector<MatType>& inputs,
                  const int requiredOutputs,
                  const int requiredInternals,
                  std::vector<MatType>& outputs,
                  std::vector<MatType>& internals) const CV_OVERRIDE {
        CV_CheckEQ(inputs.size(), static_cast<size_t>(1), "DNN/MatMulNBits: expected a single activation input");
        CV_CheckType(inputs[0], inputs[0] == CV_32F, "DNN/MatMulNBits: activation must be CV_32F");
        outputs.assign(requiredOutputs, CV_32F);
        internals.assign(requiredInternals, MatType(-1));
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &) const CV_OVERRIDE {
        CV_Assert(!inputs.empty());
        const int64 M = total(inputs[0]) / K;
        return CV_BIG_INT(2) * M * N * K;
    }

    // Dequantize one block into w[0..block_size-1]. BITS is compile-time so the per-byte
    // unpack unrolls and the shift folds to a constant (the hot half of the kernel).
    template<int BITS>
    static inline void dequantBlock(const uchar* packed, int blob_bytes, float scale, float* w) {
        const int EPB = 8 / BITS;            // elements per byte: 4-bit:2  8-bit:1
        const int QMASK = (1 << BITS) - 1;
        const int ZP = 1 << (BITS - 1);      // symmetric default zero-point
        for (int j = 0; j < blob_bytes; j++) {
            const uchar byte = packed[j];
            for (int e = 0; e < EPB; e++)     // LSB-first within the byte
                w[j * EPB + e] = (((byte >> (e * BITS)) & QMASK) - ZP) * scale;
        }
    }

#if CV_SIMD128
    // 32 4-bit weights (16 bytes, low nibble first) as (q - 8) * scale, in order, in w[0..7].
    static inline void dequant32x4(const uchar* p, const v_float32x4& s, const v_float32x4& zs,
                                   v_float32x4* w) {
        const v_uint8x16 b = v_load(p), mask = v_setall_u8(15);
        const v_uint8x16 lo = v_and(b, mask);
        const v_uint8x16 hi = v_and(v_reinterpret_as_u8(v_shr<4>(v_reinterpret_as_u16(b))), mask);
        v_uint8x16 q0, q1;
        v_zip(lo, hi, q0, q1);
        v_uint16x8 h[4];
        v_expand(q0, h[0], h[1]);
        v_expand(q1, h[2], h[3]);
        for (int i = 0; i < 4; i++) {
            v_uint32x4 u0, u1;
            v_expand(h[i], u0, u1);
            w[2 * i]     = v_fma(v_cvt_f32(v_reinterpret_as_s32(u0)), s, zs);
            w[2 * i + 1] = v_fma(v_cvt_f32(v_reinterpret_as_s32(u1)), s, zs);
        }
    }
#endif

    template<int BITS>
    void runImpl(const Mat& A, const Mat& B, const Mat& scales, Mat& Y) const {
        const int M = static_cast<int>(A.total() / K);
        const int n_blk = (K + block_size - 1) / block_size;
        const int blob_bytes = block_size * BITS / 8;   // packed weights per block

        const float* a = A.ptr<const float>();
        const uchar* bq = B.ptr<const uchar>();
        const float* sc = scales.ptr<const float>();
        float* y = Y.ptr<float>();
#if CV_SIMD128
        // 4-bit weights in whole 32-element groups: unpacked with SIMD, never stored for one row
        const bool simd4 = BITS == 4 && block_size % 32 == 0 && K % 32 == 0;
#else
        const bool simd4 = false;
#endif
        // ~64k multiply-adds per stripe: one stripe per column costs more than the column
        const double nstripes = std::max(1., std::min((double)N, (double)N * M * K / (1 << 16)));

        // Each range owns a disjoint set of output columns n, so the column writes are race-free.
        parallel_for_(Range(0, N), [&](const Range& r) {
            // Dequant the whole column once so it amortizes across the M rows.
            AutoBuffer<float, 1024> wcol(n_blk * block_size);   // whole blocks (>= K): the last block is dequantized in full
            for (int n = r.start; n < r.end; n++) {
                const uchar* bq_n = bq + static_cast<size_t>(n) * n_blk * blob_bytes;
                const float* sc_n = sc + static_cast<size_t>(n) * n_blk;
#if CV_SIMD128
                if (simd4) {
                    v_float32x4 w[8];
                    if (M == 1) {
                        v_float32x4 s0 = v_setzero_f32(), s1 = s0, s2 = s0, s3 = s0;
                        for (int k = 0; k < K; k += 32) {
                            const v_float32x4 s = v_setall_f32(sc_n[k / block_size]);
                            dequant32x4(bq_n + k / 2, s, v_mul(s, v_setall_f32(-8.f)), w);
                            s0 = v_fma(v_load(a + k),      w[0], s0);
                            s1 = v_fma(v_load(a + k + 4),  w[1], s1);
                            s2 = v_fma(v_load(a + k + 8),  w[2], s2);
                            s3 = v_fma(v_load(a + k + 12), w[3], s3);
                            s0 = v_fma(v_load(a + k + 16), w[4], s0);
                            s1 = v_fma(v_load(a + k + 20), w[5], s1);
                            s2 = v_fma(v_load(a + k + 24), w[6], s2);
                            s3 = v_fma(v_load(a + k + 28), w[7], s3);
                        }
                        y[n] = v_reduce_sum(v_add(v_add(s0, s1), v_add(s2, s3)));
                        continue;
                    }
                    for (int k = 0; k < K; k += 32) {
                        const v_float32x4 s = v_setall_f32(sc_n[k / block_size]);
                        dequant32x4(bq_n + k / 2, s, v_mul(s, v_setall_f32(-8.f)), w);
                        for (int i = 0; i < 8; i++)
                            v_store(wcol.data() + k + 4 * i, w[i]);
                    }
                } else
#endif
                for (int blk = 0; blk < n_blk; blk++)
                    dequantBlock<BITS>(bq_n + static_cast<size_t>(blk) * blob_bytes, blob_bytes,
                                       sc_n[blk], wcol.data() + blk * block_size);

                const float* w = wcol.data();
                for (int m = 0; m < M; m++) {
                    const float* arow = a + static_cast<size_t>(m) * K;
                    float acc = 0.f;
                    int t = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                    const int vlanes = VTraits<v_float32>::vlanes();
                    v_float32 vacc0 = vx_setzero_f32(), vacc1 = vx_setzero_f32();
                    v_float32 vacc2 = vx_setzero_f32(), vacc3 = vx_setzero_f32();
                    for (; t <= K - 4 * vlanes; t += 4 * vlanes) {
                        vacc0 = v_fma(vx_load(arow + t),              vx_load(w + t),              vacc0);
                        vacc1 = v_fma(vx_load(arow + t + vlanes),     vx_load(w + t + vlanes),     vacc1);
                        vacc2 = v_fma(vx_load(arow + t + 2 * vlanes), vx_load(w + t + 2 * vlanes), vacc2);
                        vacc3 = v_fma(vx_load(arow + t + 3 * vlanes), vx_load(w + t + 3 * vlanes), vacc3);
                    }
                    v_float32 vacc = v_add(v_add(vacc0, vacc1), v_add(vacc2, vacc3));
                    for (; t <= K - vlanes; t += vlanes)
                        vacc = v_fma(vx_load(arow + t), vx_load(w + t), vacc);
                    acc = v_reduce_sum(vacc);
#endif
                    for (; t < K; t++)
                        acc += arow[t] * w[t];
                    y[static_cast<size_t>(m) * N + n] = acc;
                }
            }
        }, nstripes);
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        const Mat& A = inputs[0];
        const Mat& B = blobs[0];        // packed weights, uint8 [N, n_blk, block_size*bits/8]
        const Mat& scales = blobs[1];   // fp32 [N, n_blk]
        Mat& Y = outputs[0];

        CV_Assert(A.isContinuous() && Y.isContinuous());

        switch (bits) {                 // bits ∈ {4,8} enforced in the constructor
            case 4:  runImpl<4>(A, B, scales, Y); break;
            default: runImpl<8>(A, B, scales, Y); break;
        }
    }
};

Ptr<MatMulNBitsLayer> MatMulNBitsLayer::create(const LayerParams& params)
{
    return makePtr<MatMulNBitsLayerImpl>(params);
}

}} // cv::dnn

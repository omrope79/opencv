// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_FLASH_ATTENTION_HPP
#define OPENCV_DNN_FLASH_ATTENTION_HPP

#include <opencv2/core.hpp>

namespace cv { namespace dnn {

// True when flashAttention() should run instead of MLAS flash attention (AArch64 only, see OPENCV_DNN_GEMM_PATH).
bool flashAttentionEnabled();

// Additive mask, added to the scaled scores: mask[b*sB + h*sH + i*sQ + j*sK] for query i and key j.
// A stride of 0 broadcasts that axis, so [B, 1, 1, Skv] is {S, 0, 0, 1} and [B, H, Sq, Skv] is fully strided.
struct FlashAttentionMask
{
    const float* data = nullptr;
    size_t sB = 0, sH = 0, sQ = 0, sK = 0;
};

// Element strides of Q, K and V over batch, head and row. Lets the kernel read heads in place from a
// [B*S, Q|K|V] projection, where a head is a column range and a row step is the full projection width.
struct FlashAttentionLayout
{
    size_t qB, qH, qS, kB, kH, kS, vB, vH, vS;
};

// out[b, i, h, :] = softmax(scale * Q[b, h, i, :] * K[b, h]^T + mask[b, h, i, :]) * V[b, h], without forming
// the full score matrix. Default layouts as in mlasFlashAttention: Q [B, H, Sq, D], V [B, H, Skv, Dv],
// out [B, Sq, H, Dv]; K is [B, H, Skv, D], or [B, H, D, Skv] when kTransposed is set.
// layout overrides the Q/K/V layouts (not with kTransposed); mask is optional.
void flashAttention(const float* Q, const float* K, const float* V, float* out,
                    int B, int H, int Sq, int Skv, int D, int Dv, float scale, bool kTransposed = false,
                    const FlashAttentionMask* mask = nullptr, const FlashAttentionLayout* layout = nullptr);

}} // cv::dnn

#endif // OPENCV_DNN_FLASH_ATTENTION_HPP

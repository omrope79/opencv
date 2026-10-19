// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Fuses multi-head attention subgraphs into a single Attention layer.

#include "precomp.hpp"
#include "net_impl.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

using std::vector;
using std::string;

// Concatenates the Q, K and V biases; a missing one becomes zeros, since the Attention layer
// always needs the full bias. Returns false if a present bias does not match its projection.
static bool concatQKVBias(const Mat (&b)[3], const int (&n)[3], Mat& bias_qkv)
{
    const int total = n[0] + n[1] + n[2];
    bias_qkv = Mat(1, &total, CV_32F, Scalar(0));
    float* dst = bias_qkv.ptr<float>();
    for (int i = 0; i < 3; dst += n[i++]) {
        if (b[i].empty())
            continue;
        if (b[i].type() != CV_32F || b[i].total() != (size_t)n[i] || !b[i].isContinuous())
            return false;
        std::memcpy(dst, b[i].ptr<float>(), n[i] * sizeof(float));
    }
    return true;
}

struct ModelFusionAttention
{
    ModelFusionAttention(Net::Impl* netimpl_) : netimpl(netimpl_) {}

    void fuse()
    {
        fuseGraph(netimpl->mainGraph);
    }

    int singleConsumer(Arg a) const
    {
        auto it = consumers_.find(a.idx);
        if (it == consumers_.end() || it->second.size() != 1)
            return -1;
        return it->second[0];
    }

    bool isReshape(const vector<Ptr<LayerInfo>>& prog, int idx) const
    {
        if (idx < 0 || idx >= (int)prog.size() || !prog[idx])
            return false;
        return dynamic_cast<Reshape2Layer*>(prog[idx].get()) != nullptr;
    }

    bool isTranspose(const vector<Ptr<LayerInfo>>& prog, int idx) const
    {
        if (idx < 0 || idx >= (int)prog.size() || !prog[idx])
            return false;
        return dynamic_cast<TransposeLayer*>(prog[idx].get()) != nullptr;
    }

    bool isSoftmax(const vector<Ptr<LayerInfo>>& prog, int idx) const
    {
        if (idx < 0 || idx >= (int)prog.size() || !prog[idx])
            return false;
        return prog[idx]->type == "Softmax";
    }

    bool isMatMul(const vector<Ptr<LayerInfo>>& prog, int idx) const
    {
        if (idx < 0 || idx >= (int)prog.size() || !prog[idx])
            return false;
        return dynamic_cast<MatMulLayer*>(prog[idx].get()) != nullptr;
    }

    static bool isProjCandidate(const Ptr<LayerInfo>& l)
    {
        if (l->blobs.empty() || l->inputs.size() != 1) return false;
        if (dynamic_cast<MatMulLayer*>(l.get()))
            return true;
        GemmLayer* g = dynamic_cast<GemmLayer*>(l.get());
        if (!g) return false;
        if (g->trans_a || g->alpha != 1.f) return false;
        if (l->blobs.size() == 2 && g->beta != 1.f) return false;
        return true;
    }

    // Returns the projection weight in [K, N] (input_hidden, output_hidden)
    // layout, transposing if the source is a Gemm with trans_b.
    static Mat getProjWeight(const Ptr<LayerInfo>& l)
    {
        const Mat& W = l->blobs[0];
        GemmLayer* g = dynamic_cast<GemmLayer*>(l.get());
        if (g && g->trans_b) {
            Mat Wt;
            cv::transpose(W, Wt);
            return Wt;
        }
        return W;
    }

    bool isScalarBinOp(const vector<Ptr<LayerInfo>>& prog, int idx,
                       NaryEltwiseLayer::OPERATION op, float* val) const
    {
        if (idx < 0 || idx >= (int)prog.size() || !prog[idx])
            return false;
        NaryEltwiseLayer* elt = dynamic_cast<NaryEltwiseLayer*>(prog[idx].get());
        if (!elt || elt->op != op)
            return false;
        const auto& inputs = prog[idx]->inputs;
        if (inputs.size() != 2) return false;
        // Mul commutes, but for Div the constant has to be the divisor.
        const int first = op == NaryEltwiseLayer::OPERATION::DIV ? 1 : 0;
        for (int k = first; k < 2; k++)
            if (readScalar(inputs[k], *val)) return true;
        return false;
    }

    int producerOf(Arg a) const
    {
        auto it = producer_.find(a.idx);
        return it == producer_.end() ? -1 : it->second;
    }

    bool readScalar(Arg a, float& v) const
    {
        if (!netimpl->isConstArg(a)) return false;
        Mat t = netimpl->argTensor(a);
        if (t.total() != 1) return false;
        if (t.type() == CV_32F) { v = t.at<float>(0); return true; }
        if (t.type() == CV_64F) { v = (float)t.at<double>(0); return true; }
        return false;
    }

    // Integer list held in a const arg, e.g. Split sizes or Squeeze axes passed as an input.
    bool readConstInts(Arg a, vector<int>& out) const
    {
        if (!netimpl->isConstArg(a)) return false;
        Mat t = netimpl->argTensor(a);
        out.clear();
        if (t.type() == CV_64S) {
            const int64_t* p = t.ptr<int64_t>();
            for (size_t i = 0; i < t.total(); i++) out.push_back(saturate_cast<int>(p[i]));
            return true;
        }
        if (t.type() == CV_32S) {
            const int* p = t.ptr<int>();
            out.assign(p, p + t.total());
            return true;
        }
        return false;
    }

    bool isScalarMul(const vector<Ptr<LayerInfo>>& prog, int idx, float* val) const
    {
        return isScalarBinOp(prog, idx, NaryEltwiseLayer::OPERATION::PROD, val);
    }

    bool isScalarDiv(const vector<Ptr<LayerInfo>>& prog, int idx, float* val) const
    {
        return isScalarBinOp(prog, idx, NaryEltwiseLayer::OPERATION::DIV, val);
    }

    // True if `arg` is head_dim^-1/4 computed at run time, Sqrt(Div(1, Sqrt(Slice(Shape(x), -1)))), with a Cast2 allowed between steps; x goes to `dim_src`.
    bool isRuntimeQKScaleChain(const vector<Ptr<LayerInfo>>& prog, Arg arg,
                                std::set<int>& chain_ops, Arg& dim_src) const
    {
        static const char* const expected[] = { "Sqrt", "Div", "Sqrt", "Slice2", "Shape" };
        std::set<int> visited;
        Arg cur = arg;
        for (const char* want_c : expected) {
            const std::string want = want_c;
            int idx = producerOf(cur);
            while (idx >= 0 && idx < (int)prog.size() && prog[idx] &&
                   prog[idx]->type == "Cast2" && !prog[idx]->inputs.empty()) {
                visited.insert(idx);
                cur = prog[idx]->inputs[0];
                idx = producerOf(cur);
            }
            if (idx < 0 || idx >= (int)prog.size() || !prog[idx]) return false;
            const Ptr<LayerInfo>& l = prog[idx];
            if (want == "Div") {
                NaryEltwiseLayer* elt = dynamic_cast<NaryEltwiseLayer*>(l.get());
                float one = 0.f;
                if (!elt || elt->op != NaryEltwiseLayer::OPERATION::DIV || l->inputs.size() != 2 ||
                    !readScalar(l->inputs[0], one) || std::abs(one - 1.f) > 1e-5f)
                    return false;
                cur = l->inputs[1];
            } else {
                if (l->type != want || l->inputs.empty()) return false;
                if (want == "Slice2" && !isLastElementSlice(l)) return false;
                cur = l->inputs[0];
            }
            visited.insert(idx);
        }
        chain_ops.insert(visited.begin(), visited.end());
        dim_src = cur;
        return true;
    }

    // Slice that keeps only the last element, as the scale chain does to read head_dim off a shape.
    bool isLastElementSlice(const Ptr<LayerInfo>& l) const
    {
        const Slice2Layer* s = dynamic_cast<const Slice2Layer*>(l.get());
        if (!s) return false;
        if (!s->starts.empty()) return s->starts.size() == 1 && s->starts[0] == -1;
        vector<int> starts;
        return l->inputs.size() > 1 && readConstInts(l->inputs[1], starts) &&
               starts.size() == 1 && starts[0] == -1;
    }

    // Accepts scores + mask in either operand order; the other operand is the mask.
    bool isMaskAdd(const vector<Ptr<LayerInfo>>& prog, int idx, Arg scores, Arg* out_mask) const
    {
        if (idx < 0 || idx >= (int)prog.size() || !prog[idx])
            return false;
        NaryEltwiseLayer* elt = dynamic_cast<NaryEltwiseLayer*>(prog[idx].get());
        if (!elt || elt->op != NaryEltwiseLayer::OPERATION::ADD)
            return false;
        const auto& inputs = prog[idx]->inputs;
        if (inputs.size() != 2) return false;
        if (inputs[0].idx == scores.idx) *out_mask = inputs[1];
        else if (inputs[1].idx == scores.idx) *out_mask = inputs[0];
        else return false;
        return out_mask->idx != scores.idx;
    }

    // Extract a scalar integer from a const-valued arg, possibly wrapped in an
    // Unsqueeze of a scalar const. Returns -1 if extraction fails.
    int extractConstInt(const vector<Ptr<LayerInfo>>& prog, Arg a) const
    {
        vector<int> v;
        if (readConstInts(a, v) && v.size() == 1 && v[0] > 0) return v[0];
        int pidx = producerOf(a);
        if (pidx < 0 || pidx >= (int)prog.size() || !prog[pidx]) return -1;
        if (prog[pidx]->type == "Unsqueeze" && !prog[pidx]->inputs.empty() &&
            readConstInts(prog[pidx]->inputs[0], v) && v.size() == 1)
            return v[0];
        return -1;
    }

    void collectShapeChain(const vector<Ptr<LayerInfo>>& prog, int concat_idx,
                           std::set<int>& chain) const
    {
        if (concat_idx < 0 || concat_idx >= (int)prog.size() || !prog[concat_idx])
            return;
        chain.insert(concat_idx);
        for (Arg ci : prog[concat_idx]->inputs) {
            if (netimpl->isConstArg(ci)) continue;
            int cur = -1;
            auto it = producer_.find(ci.idx);
            if (it != producer_.end()) cur = it->second;
            while (cur >= 0 && (int)prog.size() > cur && prog[cur]) {
                const std::string& t = prog[cur]->type;
                if (t != "Unsqueeze" && t != "Gather2" && t != "Shape") break;
                Arg cur_out = prog[cur]->outputs[0];
                auto cit = consumers_.find(cur_out.idx);
                if (cit == consumers_.end()) break;
                bool all_in_chain = true;
                for (int c : cit->second) if (!chain.count(c)) { all_in_chain = false; break; }
                if (!all_in_chain) break;
                chain.insert(cur);
                if (prog[cur]->inputs.empty()) break;
                Arg prev = prog[cur]->inputs[0];
                auto pit = producer_.find(prev.idx);
                if (pit == producer_.end()) break;
                cur = pit->second;
            }
        }
    }

    template <class Pred>
    int findMatchingConsumer(const vector<Ptr<LayerInfo>>& prog, Arg out,
                             Pred pred, std::set<int>* extra_shape_ops) const
    {
        auto it = consumers_.find(out.idx);
        if (it == consumers_.end()) return -1;
        int matched = -1;
        for (int c : it->second) {
            if (c < 0 || c >= (int)prog.size() || !prog[c]) return -1;
            if (pred(prog[c].get())) {
                if (matched >= 0) return -1;
                matched = c;
            } else if (prog[c]->type == "Shape") {
                if (extra_shape_ops) extra_shape_ops->insert(c);
            } else {
                return -1;
            }
        }
        return matched;
    }

    //! Scale a Q or K branch carries: a constant factor, or the run-time head_dim^-1/4.
    struct QKScale
    {
        int op = -1;
        float mult = 1.f;
        bool runtime = false;
        Arg dim_src;   //!< tensor whose last dim the run-time scale reads
    };

    //! Ops between Q*K^T and Softmax: an optional scalar Mul/Div, then an optional additive mask or bias.
    struct PostQK
    {
        int scale_op = -1, add_op = -1, softmax = -1;
        float mult = 1.f;
        Arg mask;
        bool has_mask = false;
    };

    // Steps from `a` over an optional scale op (x*c, x/c, or x times the run-time scale) and returns the op after it.
    int skipScale(const vector<Ptr<LayerInfo>>& prog, Arg a, QKScale& s,
                  std::set<int>& chain_ops, std::set<int>* shape_ops) const
    {
        s = QKScale();
        int next = findMatchingConsumer(prog, a,
            [](LayerInfo* L) { return L->type != "Shape"; }, shape_ops);
        if (next < 0) return -1;
        NaryEltwiseLayer* elt = dynamic_cast<NaryEltwiseLayer*>(prog[next].get());
        if (!elt || prog[next]->inputs.size() != 2) return next;
        float c = 0.f;
        if (isScalarMul(prog, next, &c)) {
            s.mult = c;
        } else if (isScalarDiv(prog, next, &c) && c != 0.f) {
            s.mult = 1.f / c;
        } else if (elt->op == NaryEltwiseLayer::OPERATION::PROD) {
            const auto& in = prog[next]->inputs;
            const Arg other = in[0].idx == a.idx ? in[1] : in[0];
            if (!isRuntimeQKScaleChain(prog, other, chain_ops, s.dim_src)) return next;
            s.runtime = true;
        } else {
            return next;
        }
        s.op = next;
        return singleConsumer(prog[next]->outputs[0]);
    }

    bool parsePostQK(const vector<Ptr<LayerInfo>>& prog, int qk_idx, bool allow_mask, PostQK& p) const
    {
        p = PostQK();
        Arg cur = prog[qk_idx]->outputs[0];
        int next = singleConsumer(cur);
        float c = 0.f;
        bool scaled = false;
        if (isScalarMul(prog, next, &c)) { p.mult = c; scaled = true; }
        else if (isScalarDiv(prog, next, &c) && c != 0.f) { p.mult = 1.f / c; scaled = true; }
        if (scaled) {
            p.scale_op = next;
            cur = prog[next]->outputs[0];
            next = singleConsumer(cur);
        }
        if (allow_mask && isMaskAdd(prog, next, cur, &p.mask)) {
            p.add_op = next;
            p.has_mask = true;
            cur = prog[next]->outputs[0];
            next = singleConsumer(cur);
        }
        if (!isSoftmax(prog, next)) return false;
        SoftmaxLayer* sm = dynamic_cast<SoftmaxLayer*>(prog[next].get());
        if (!sm || sm->logSoftMax) return false;
        p.softmax = next;
        return true;
    }

    // Divisor for the `scale` param; 0 means sqrt(head_dim) known only at run time (SDPA's head_dim_scale).
    static bool combineScale(const QKScale& q, const QKScale& k, const PostQK& p,
                             int head_dim, Arg q_head, float& divisor)
    {
        // Each side carries head_dim^-1/4, so only the pair makes 1/sqrt(d).
        if (q.runtime != k.runtime) return false;
        float m = q.mult * k.mult * p.mult;
        if (q.runtime) {
            // The fused layer takes head_dim from Q, so both chains must read Q's shape.
            if (q.dim_src.idx != q_head.idx || k.dim_src.idx != q_head.idx) return false;
            if (head_dim <= 0) {
                if (m != 1.f) return false;
                divisor = 0.f;
                return true;
            }
            m /= std::sqrt((float)head_dim);
        }
        if (!(m > 0.f) || !std::isfinite(m)) return false;
        divisor = 1.f / m;
        return true;
    }

    // Adds the run-time scale ops whose every consumer is being removed; Q and K share the chain, so it goes as a unit.
    void addRemovableChainOps(const vector<Ptr<LayerInfo>>& prog, const std::set<int>& chain_ops,
                              std::set<int>& to_remove) const
    {
        for (int op_idx : chain_ops) {
            if (op_idx < 0 || op_idx >= (int)prog.size() || !prog[op_idx]) continue;
            bool all_consumers_removed = true;
            for (Arg out : prog[op_idx]->outputs) {
                auto cit = consumers_.find(out.idx);
                if (cit == consumers_.end()) continue;
                for (int c : cit->second) {
                    if (to_remove.count(c) || chain_ops.count(c)) continue;
                    all_consumers_removed = false;
                    break;
                }
                if (!all_consumers_removed) break;
            }
            if (all_consumers_removed) to_remove.insert(op_idx);
        }
    }

    // Q, K and V of a [3, B, H, S, D] tensor: three Gathers on axis 0, or a 3-way Split on axis 0 with each part squeezed.
    bool splitQKV(const vector<Ptr<LayerInfo>>& prog, Arg packed, Arg (&heads)[3], std::set<int>& ops) const
    {
        auto cit = consumers_.find(packed.idx);
        if (cit == consumers_.end()) return false;
        const vector<int>& cs = cit->second;
        for (int c : cs)
            if (c < 0 || c >= (int)prog.size() || !prog[c]) return false;
        if (cs.size() == 1 && dynamic_cast<Split2Layer*>(prog[cs[0]].get())) {
            const Ptr<LayerInfo>& sp = prog[cs[0]];
            const Split2Layer* s = static_cast<const Split2Layer*>(sp.get());
            if (s->axis != 0 || sp->outputs.size() != 3) return false;
            vector<int> sizes = s->split;
            if (sizes.empty() && sp->inputs.size() > 1 && !readConstInts(sp->inputs[1], sizes)) return false;
            for (int v : sizes) if (v != 1) return false;
            ops.insert(cs[0]);
            for (int k = 0; k < 3; k++) {
                int sq = singleConsumer(sp->outputs[k]);
                SqueezeLayer* q = sq >= 0 && prog[sq] ? dynamic_cast<SqueezeLayer*>(prog[sq].get()) : nullptr;
                if (!q) return false;
                vector<int> axes = q->axes;
                if (axes.empty() && prog[sq]->inputs.size() > 1 && !readConstInts(prog[sq]->inputs[1], axes)) return false;
                if (axes.size() != 1 || (axes[0] != 0 && axes[0] != -5)) return false;
                ops.insert(sq);
                heads[k] = prog[sq]->outputs[0];
            }
            return true;
        }
        if (cs.size() != 3) return false;
        int found = 0;
        for (int c : cs) {
            Gather2Layer* g = dynamic_cast<Gather2Layer*>(prog[c].get());
            if (!g || g->axis != 0 || prog[c]->inputs.size() < 2) return false;
            vector<int> v;
            if (!readConstInts(prog[c]->inputs[1], v) || v.size() != 1 || v[0] < 0 || v[0] > 2) return false;
            if (found & (1 << v[0])) return false;
            found |= 1 << v[0];
            heads[v[0]] = prog[c]->outputs[0];
            ops.insert(c);
        }
        return found == 7;
    }

    // Parses a Reshape shape entry [Unsqueeze](Gather(Shape(x), axis)) into the tensor x and its dim `axis`.
    bool shapeDimSource(const vector<Ptr<LayerInfo>>& prog, Arg e, Arg& x, int& axis) const
    {
        int g = producerOf(e);
        if (g >= 0 && g < (int)prog.size() && prog[g] && prog[g]->type == "Unsqueeze" && !prog[g]->inputs.empty())
            g = producerOf(prog[g]->inputs[0]);
        if (g < 0 || g >= (int)prog.size() || !prog[g]) return false;
        const Gather2Layer* gl = dynamic_cast<const Gather2Layer*>(prog[g].get());
        vector<int> idx;
        if (!gl || gl->axis != 0 || prog[g]->inputs.size() < 2 ||
            !readConstInts(prog[g]->inputs[1], idx) || idx.size() != 1)
            return false;
        const int s = producerOf(prog[g]->inputs[0]);
        const ShapeLayer* sl = s >= 0 && s < (int)prog.size() && prog[s] ?
                               dynamic_cast<const ShapeLayer*>(prog[s].get()) : nullptr;
        if (!sl || sl->start != 0 || prog[s]->inputs.empty()) return false;
        x = prog[s]->inputs[0];
        axis = idx[0];
        return true;
    }

    // A Reshape shape entry that is a constant or one input dim.
    bool isPlainDim(const vector<Ptr<LayerInfo>>& prog, Arg a) const
    {
        Arg x;
        int axis = 0;
        return netimpl->isConstArg(a) || shapeDimSource(prog, a, x, axis);
    }

    // True if Reshape target entry `e` is dim `axis` of `x`: a const 0 (copies the dim), or that dim read off Shape(x).
    bool isDimOf(const vector<Ptr<LayerInfo>>& prog, Arg e, Arg x, int axis) const
    {
        vector<int> v;
        if (netimpl->isConstArg(e))
            return readConstInts(e, v) && v.size() == 1 && v[0] == 0;
        Arg src;
        int a = -1;
        return shapeDimSource(prog, e, src, a) && src.idx == x.idx && a == axis;
    }

    int followProjChain(const vector<Ptr<LayerInfo>>& prog,
                        int proj_matmul_idx,
                        int* out_reshape_idx,
                        int* out_num_heads,
                        vector<int>* out_perm,
                        std::set<int>* extra_ops_to_remove) const
    {
        if (proj_matmul_idx < 0) return -1;
        Arg proj_out = prog[proj_matmul_idx]->outputs[0];
        int reshape_idx = findMatchingConsumer(prog, proj_out,
            [](LayerInfo* L){ return dynamic_cast<Reshape2Layer*>(L) != nullptr; },
            extra_ops_to_remove);
        if (!isReshape(prog, reshape_idx)) return -1;

        const auto& reshape_inputs = prog[reshape_idx]->inputs;
        if (reshape_inputs.size() < 2) return -1;
        Arg shape_arg = reshape_inputs[1];

        int num_heads = -1;
        if (netimpl->isConstArg(shape_arg)) {
            Mat shape_mat = netimpl->argTensor(shape_arg);
            if (shape_mat.total() != 4) return -1;
            const int64_t* shape_data = shape_mat.ptr<int64_t>();
            num_heads = static_cast<int>(shape_data[2]);
        } else {
            auto it = producer_.find(shape_arg.idx);
            if (it == producer_.end()) return -1;
            int concat_idx = it->second;
            if (concat_idx < 0 || concat_idx >= (int)prog.size() || !prog[concat_idx])
                return -1;
            if (!dynamic_cast<Concat2Layer*>(prog[concat_idx].get()))
                return -1;
            const auto& cinputs = prog[concat_idx]->inputs;
            if (cinputs.size() != 4) return -1;
            // A batch entry like B*point_batch means the projection input is more than 3-D, which Attention can't take.
            if (!isPlainDim(prog, cinputs[0])) return -1;
            num_heads = extractConstInt(prog, cinputs[2]);
            if (num_heads <= 0) return -1;
            if (extra_ops_to_remove)
                collectShapeChain(prog, concat_idx, *extra_ops_to_remove);
        }
        *out_num_heads = num_heads;
        *out_reshape_idx = reshape_idx;

        Arg reshape_out = prog[reshape_idx]->outputs[0];
        int transpose_idx = singleConsumer(reshape_out);
        if (!isTranspose(prog, transpose_idx)) return -1;
        TransposeLayer* tr = dynamic_cast<TransposeLayer*>(prog[transpose_idx].get());
        if (!tr) return -1;
        *out_perm = tr->perm;
        return transpose_idx;
    }

    // Combined-QKV attention: QKV proj -> Reshape -> Transpose -> 3 Gathers (or Split + Squeeze) ->
    // [scale] QK^T [scale] -> [Mul/Div] -> [Add mask/bias] -> Softmax -> *V.
    bool tryFuseCombinedQKV(const vector<Ptr<LayerInfo>>& prog, int qkv_matmul_idx,
                            std::set<int>& removed_ops,
                            vector<std::pair<int, Ptr<LayerInfo>>>& replacements)
    {
        if (qkv_matmul_idx < 0 || qkv_matmul_idx >= (int)prog.size() || !prog[qkv_matmul_idx])
            return false;
        if (removed_ops.count(qkv_matmul_idx)) return false;
        if (!isProjCandidate(prog[qkv_matmul_idx])) return false;

        Mat W = getProjWeight(prog[qkv_matmul_idx]);
        if (W.dims != 2 || W.size[1] % 3 != 0) return false;
        const int proj_hidden  = W.size[1] / 3;
        const int total_hidden = W.size[1];

        Arg cur = prog[qkv_matmul_idx]->outputs[0];

        // Optional bias: separate Add op or Gemm's blobs[1].
        int add_idx = -1;
        Mat bias_mat;
        bool has_bias = false;
        if (prog[qkv_matmul_idx]->blobs.size() >= 2) {
            bias_mat = prog[qkv_matmul_idx]->blobs[1];
            has_bias = bias_mat.total() == (size_t)total_hidden && bias_mat.type() == CV_32F;
            if (!has_bias) return false;
        }
        int next = singleConsumer(cur);
        if (!has_bias && next >= 0 && next < (int)prog.size() && prog[next]) {
            NaryEltwiseLayer* e = dynamic_cast<NaryEltwiseLayer*>(prog[next].get());
            if (e && e->op == NaryEltwiseLayer::OPERATION::ADD &&
                prog[next]->inputs.size() == 2)
            {
                Mat b_candidate;
                for (Arg in : prog[next]->inputs) {
                    if (netimpl->isConstArg(in)) {
                        Mat t = netimpl->argTensor(in);
                        if (t.type() == CV_32F && (int)t.total() == total_hidden)
                            b_candidate = t;
                    }
                }
                if (!b_candidate.empty()) {
                    add_idx = next;
                    bias_mat = b_candidate;
                    has_bias = true;
                    cur = prog[add_idx]->outputs[0];
                    next = singleConsumer(cur);
                }
            }
        }

        if (!isReshape(prog, next)) return false;
        const int reshape_idx = next;
        const auto& rinputs = prog[reshape_idx]->inputs;
        if (rinputs.size() < 2) return false;

        std::set<int> extra_ops;
        int num_heads = -1, head_dim = -1;
        Arg shape_arg = rinputs[1];
        if (netimpl->isConstArg(shape_arg)) {
            vector<int> sd;
            if (!readConstInts(shape_arg, sd) || sd.size() != 5 || sd[2] != 3) return false;
            num_heads = sd[3];
            head_dim  = sd[4];
        } else {
            auto it = producer_.find(shape_arg.idx);
            if (it == producer_.end()) return false;
            int concat_idx = it->second;
            if (concat_idx < 0 || concat_idx >= (int)prog.size() || !prog[concat_idx])
                return false;
            if (!dynamic_cast<Concat2Layer*>(prog[concat_idx].get())) return false;
            const auto& cinputs = prog[concat_idx]->inputs;
            if (cinputs.size() != 5) return false;
            if (extractConstInt(prog, cinputs[2]) != 3) return false;
            num_heads = extractConstInt(prog, cinputs[3]);
            head_dim  = extractConstInt(prog, cinputs[4]);
            if (num_heads <= 0 || head_dim <= 0) return false;
            collectShapeChain(prog, concat_idx, extra_ops);
        }
        if (num_heads * head_dim != proj_hidden) return false;

        // Transpose must bring "3" to front; only perm[0]==2 is required.
        Arg reshape_out = prog[reshape_idx]->outputs[0];
        int transpose_idx = singleConsumer(reshape_out);
        if (!isTranspose(prog, transpose_idx)) return false;
        TransposeLayer* tr = dynamic_cast<TransposeLayer*>(prog[transpose_idx].get());
        if (!tr || tr->perm.size() != 5 || tr->perm[0] != 2) return false;

        Arg tr_out = prog[transpose_idx]->outputs[0];
        Arg heads[3];   // Q, K, V as [B, H, S, D]
        std::set<int> split_ops;
        if (!splitQKV(prog, tr_out, heads, split_ops)) return false;

        std::set<int> chain_ops;
        QKScale qs, ks;
        const int qk_matmul_idx = skipScale(prog, heads[0], qs, chain_ops, &extra_ops);
        if (!isMatMul(prog, qk_matmul_idx) || prog[qk_matmul_idx]->inputs.size() != 2) return false;

        const int k_trans_idx = findMatchingConsumer(prog, heads[1],
            [](LayerInfo* L) { return L->type != "Shape"; }, &extra_ops);
        if (!isTranspose(prog, k_trans_idx) ||
            static_cast<TransposeLayer*>(prog[k_trans_idx].get())->perm != vector<int>{0, 1, 3, 2}) return false;
        if (skipScale(prog, prog[k_trans_idx]->outputs[0], ks, chain_ops, &extra_ops) != qk_matmul_idx) return false;

        // Q must be the left operand of Q*K^T.
        const Arg q_in = qs.op >= 0 ? prog[qs.op]->outputs[0] : heads[0];
        if (prog[qk_matmul_idx]->inputs[0].idx != q_in.idx) return false;

        PostQK post;
        if (!parsePostQK(prog, qk_matmul_idx, /*allow_mask=*/true, post)) return false;
        const int softmax_idx = post.softmax;
        const int av_matmul_idx = singleConsumer(prog[softmax_idx]->outputs[0]);
        if (!isMatMul(prog, av_matmul_idx) || prog[av_matmul_idx]->inputs.size() != 2 ||
            prog[av_matmul_idx]->inputs[1].idx != heads[2].idx) return false;

        const int out_trans_idx = singleConsumer(prog[av_matmul_idx]->outputs[0]);
        if (!isTranspose(prog, out_trans_idx) ||
            static_cast<TransposeLayer*>(prog[out_trans_idx].get())->perm != vector<int>{0, 2, 1, 3}) return false;
        int out_reshape_idx = findMatchingConsumer(prog, prog[out_trans_idx]->outputs[0],
            [](LayerInfo* L){ return dynamic_cast<Reshape2Layer*>(L) != nullptr; },
            &extra_ops);
        if (!isReshape(prog, out_reshape_idx)) return false;

        const auto& or_inputs = prog[out_reshape_idx]->inputs;
        if (or_inputs.size() >= 2 && !netimpl->isConstArg(or_inputs[1])) {
            auto it = producer_.find(or_inputs[1].idx);
            if (it != producer_.end())
                collectShapeChain(prog, it->second, extra_ops);
        }

        // W is already laid out as [Q|K|V] along the output dim.
        Mat W_qkv = W.clone();
        Mat bias_qkv = has_bias ? bias_mat.reshape(1, 1, &total_hidden).clone()
                                : Mat(1, &total_hidden, CV_32F, Scalar(0));

        // Attention `scale` is the pre-softmax divisor of every factor met on Q, K and after Q*K^T.
        float param_scale;
        if (!combineScale(qs, ks, post, head_dim, heads[0], param_scale)) return false;

        LayerParams attn_params;
        attn_params.name = prog[qkv_matmul_idx]->name + "_fused_attention";
        attn_params.type = "Attention";
        attn_params.set("num_heads", num_heads);
        int qkv_sizes[3] = { proj_hidden, proj_hidden, proj_hidden };
        attn_params.set("qkv_hidden_sizes", DictValue::arrayInt(qkv_sizes, 3));
        attn_params.set("scale", param_scale);
        attn_params.set("output_ndims", 3);
        attn_params.blobs.push_back(W_qkv);
        attn_params.blobs.push_back(bias_qkv);

        Ptr<LayerInfo> attn_layer = LayerFactory::createLayerInstance(attn_params.type, attn_params);
        CV_Assert(attn_layer);
        Arg shared_input = prog[qkv_matmul_idx]->inputs[0];
        if (post.has_mask) attn_layer->inputs = { shared_input, post.mask };
        else               attn_layer->inputs = { shared_input };
        attn_layer->outputs = prog[out_reshape_idx]->outputs;
        attn_layer->netimpl = netimpl;

        std::set<int> to_remove = {
            qkv_matmul_idx, reshape_idx, transpose_idx, k_trans_idx,
            qk_matmul_idx, softmax_idx, av_matmul_idx, out_trans_idx, out_reshape_idx
        };
        for (int op : { add_idx, qs.op, ks.op, post.scale_op, post.add_op })
            if (op >= 0) to_remove.insert(op);
        for (int op : split_ops) to_remove.insert(op);
        for (int op : extra_ops) to_remove.insert(op);
        addRemovableChainOps(prog, chain_ops, to_remove);

        std::set<int> attn_out_args;
        for (const Arg& o : attn_layer->outputs) attn_out_args.insert(o.idx);
        if (!removalIsSelfContained(prog, to_remove, attn_out_args))
            return false;

        for (int op : to_remove) removed_ops.insert(op);
        // Placed where *V was: a mask built after the QKV projection must already exist when Attention runs.
        replacements.push_back({av_matmul_idx, attn_layer});
        return true;
    }

    // CLIP-branch trace: arg -> [Transpose3D(K^T)] -> Reshape3D -> Transpose ->
    // Reshape4D -> [Mul(Q scale)] -> [Add(bias)] -> proj_MatMul.
    int traceClipBranch(const vector<Ptr<LayerInfo>>& prog, Arg arg,
                        bool is_q_branch, bool is_k_branch,
                        Mat& out_W, Mat& out_bias, int& out_num_heads,
                        float& out_q_scale,
                        std::set<int>& ops_consumed) const
    {
        Arg cur = arg;

        // K side: peel the (B*H,S,D) -> (B*H,D,S) transpose-3D first.
        if (is_k_branch) {
            int idx = producerOf(cur);
            if (idx < 0 || !prog[idx]) return -1;
            TransposeLayer* tr = dynamic_cast<TransposeLayer*>(prog[idx].get());
            if (!tr || tr->perm.size() != 3) return -1;
            if (tr->perm[0] != 0 || tr->perm[1] != 2 || tr->perm[2] != 1) return -1;
            ops_consumed.insert(idx);
            cur = prog[idx]->inputs[0];
        }

        // (B,H,S,D) -> (B*H,S,D); shape may be dynamic, validated downstream.
        int r3d_idx = producerOf(cur);
        if (!isReshape(prog, r3d_idx)) return -1;
        ops_consumed.insert(r3d_idx);
        if (prog[r3d_idx]->inputs.size() < 2) return -1;
        Arg shape_arg_r3d = prog[r3d_idx]->inputs[1];
        if (!netimpl->isConstArg(shape_arg_r3d)) {
            int sh_idx = producerOf(shape_arg_r3d);
            if (sh_idx >= 0)
                collectShapeChain(prog, sh_idx, ops_consumed);
        }
        cur = prog[r3d_idx]->inputs[0];

        int t4d_idx = producerOf(cur);
        if (!isTranspose(prog, t4d_idx)) return -1;
        TransposeLayer* tr4 = dynamic_cast<TransposeLayer*>(prog[t4d_idx].get());
        if (!tr4 || tr4->perm.size() != 4) return -1;
        if (tr4->perm[0] != 0 || tr4->perm[1] != 2 ||
            tr4->perm[2] != 1 || tr4->perm[3] != 3) return -1;
        ops_consumed.insert(t4d_idx);
        cur = prog[t4d_idx]->inputs[0];

        // (B,S,H*D) -> (B,S,H,D); num_heads is the third shape entry.
        int r4d_idx = producerOf(cur);
        if (!isReshape(prog, r4d_idx)) return -1;
        if (prog[r4d_idx]->inputs.size() < 2) return -1;
        Arg shape_arg_r4d = prog[r4d_idx]->inputs[1];
        int num_heads = -1;
        if (netimpl->isConstArg(shape_arg_r4d)) {
            vector<int> shape_vals;
            if (!readConstInts(shape_arg_r4d, shape_vals) || shape_vals.size() != 4) return -1;
            num_heads = shape_vals[2];
        } else {
            int concat_idx = producerOf(shape_arg_r4d);
            if (concat_idx < 0 || !prog[concat_idx]) return -1;
            if (!dynamic_cast<Concat2Layer*>(prog[concat_idx].get())) return -1;
            const auto& cinputs = prog[concat_idx]->inputs;
            if (cinputs.size() != 4) return -1;
            num_heads = extractConstInt(prog, cinputs[2]);
            if (num_heads <= 0) return -1;
            collectShapeChain(prog, concat_idx, ops_consumed);
        }
        if (num_heads <= 0) return -1;
        out_num_heads = num_heads;
        ops_consumed.insert(r4d_idx);
        cur = prog[r4d_idx]->inputs[0];

        out_q_scale = 1.0f;
        if (is_q_branch) {
            int mul_idx = producerOf(cur);
            if (mul_idx < 0 || !prog[mul_idx]) return -1;
            NaryEltwiseLayer* mul =
                dynamic_cast<NaryEltwiseLayer*>(prog[mul_idx].get());
            if (!mul || mul->op != NaryEltwiseLayer::OPERATION::PROD) return -1;
            if (prog[mul_idx]->inputs.size() != 2) return -1;
            Arg runtime_arg;
            bool got_scale = false, got_runtime = false;
            for (Arg in : prog[mul_idx]->inputs) {
                if (netimpl->isConstArg(in)) {
                    if (!readScalar(in, out_q_scale)) return -1;
                    got_scale = true;
                } else {
                    runtime_arg = in;
                    got_runtime = true;
                }
            }
            if (!got_scale || !got_runtime) return -1;
            ops_consumed.insert(mul_idx);
            cur = runtime_arg;
        }

        int next_idx = producerOf(cur);
        if (next_idx < 0 || !prog[next_idx]) return -1;

        out_bias = Mat();
        int mm_idx = -1;
        if (dynamic_cast<NaryEltwiseLayer*>(prog[next_idx].get())) {
            NaryEltwiseLayer* add =
                dynamic_cast<NaryEltwiseLayer*>(prog[next_idx].get());
            if (!add || add->op != NaryEltwiseLayer::OPERATION::ADD) return -1;
            if (prog[next_idx]->inputs.size() != 2) return -1;
            Arg bias_arg;
            Arg matmul_out_arg;
            bool got_bias = false, got_runtime2 = false;
            for (Arg in : prog[next_idx]->inputs) {
                if (netimpl->isConstArg(in)) {
                    bias_arg = in;
                    got_bias = true;
                } else {
                    matmul_out_arg = in;
                    got_runtime2 = true;
                }
            }
            if (!got_bias || !got_runtime2) return -1;
            out_bias = netimpl->argTensor(bias_arg).clone();
            ops_consumed.insert(next_idx);
            mm_idx = producerOf(matmul_out_arg);
        } else {
            mm_idx = next_idx;
        }

        if (mm_idx < 0 || !prog[mm_idx]) return -1;
        if (!dynamic_cast<MatMulLayer*>(prog[mm_idx].get())) return -1;
        if (prog[mm_idx]->blobs.empty()) return -1;
        if (prog[mm_idx]->inputs.size() != 1) return -1;
        out_W = prog[mm_idx]->blobs[0].clone();
        // Folded MatMul carries bias as a second blob (real_ndims_C >= 1).
        if (out_bias.empty() && prog[mm_idx]->blobs.size() >= 2)
            out_bias = prog[mm_idx]->blobs.back().clone();
        return mm_idx;
    }

    // 3 separate q/k/v projections, Q scaled, K^T at
    // the QK^T matmul, output reshaped+transposed back to (B,S,H*D).
    bool tryFuseClipAttention(const vector<Ptr<LayerInfo>>& prog, int softmax_idx,
                              std::set<int>& removed_ops,
                              vector<std::pair<int, Ptr<LayerInfo>>>& replacements)
    {
        if (softmax_idx < 0 || softmax_idx >= (int)prog.size() || !prog[softmax_idx])
            return false;
        if (removed_ops.count(softmax_idx)) return false;
        SoftmaxLayer* sm = dynamic_cast<SoftmaxLayer*>(prog[softmax_idx].get());
        if (!sm || sm->logSoftMax) return false;
        if (prog[softmax_idx]->inputs.size() != 1) return false;

        Arg sm_in = prog[softmax_idx]->inputs[0];
        auto it = producer_.find(sm_in.idx);
        if (it == producer_.end()) return false;
        int qk_matmul_idx = it->second;
        if (qk_matmul_idx < 0 || !prog[qk_matmul_idx]) return false;
        if (!dynamic_cast<MatMulLayer*>(prog[qk_matmul_idx].get())) return false;
        if (!prog[qk_matmul_idx]->blobs.empty()) return false;
        if (prog[qk_matmul_idx]->inputs.size() != 2) return false;
        Arg q_arg = prog[qk_matmul_idx]->inputs[0];
        Arg k_arg = prog[qk_matmul_idx]->inputs[1];

        Arg sm_out = prog[softmax_idx]->outputs[0];
        int av_matmul_idx = singleConsumer(sm_out);
        if (av_matmul_idx < 0 || !prog[av_matmul_idx]) return false;
        if (!dynamic_cast<MatMulLayer*>(prog[av_matmul_idx].get())) return false;
        if (!prog[av_matmul_idx]->blobs.empty()) return false;
        if (prog[av_matmul_idx]->inputs.size() != 2) return false;
        Arg v_arg;
        bool got_v = false;
        for (Arg in : prog[av_matmul_idx]->inputs) {
            if (in.idx == sm_out.idx) continue;
            v_arg = in; got_v = true;
        }
        if (!got_v) return false;

        std::set<int> consumed;
        Mat Wq, Wk, Wv, bq, bk, bv;
        int nh_q = 0, nh_k = 0, nh_v = 0;
        float q_scale = 1.f;
        float dummy = 1.f;
        int q_mm_idx = traceClipBranch(prog, q_arg, /*is_q=*/true, /*is_k=*/false,
                                       Wq, bq, nh_q, q_scale, consumed);
        if (q_mm_idx < 0) return false;
        int k_mm_idx = traceClipBranch(prog, k_arg, /*is_q=*/false, /*is_k=*/true,
                                       Wk, bk, nh_k, dummy, consumed);
        if (k_mm_idx < 0) return false;
        int v_mm_idx = traceClipBranch(prog, v_arg, /*is_q=*/false, /*is_k=*/false,
                                       Wv, bv, nh_v, dummy, consumed);
        if (v_mm_idx < 0) return false;

        if (q_mm_idx == k_mm_idx || k_mm_idx == v_mm_idx || q_mm_idx == v_mm_idx)
            return false;
        if (nh_q != nh_k || nh_k != nh_v) return false;
        const int num_heads = nh_q;
        if (q_scale == 0.f) return false;

        // All three projections share the same input.
        Arg shared_input = prog[q_mm_idx]->inputs[0];
        if (prog[k_mm_idx]->inputs[0].idx != shared_input.idx) return false;
        if (prog[v_mm_idx]->inputs[0].idx != shared_input.idx) return false;

        if (Wq.dims != 2 || Wk.dims != 2 || Wv.dims != 2) return false;
        if (Wq.size[0] != Wk.size[0] || Wk.size[0] != Wv.size[0]) return false;
        const int q_hidden = Wq.size[1];
        const int k_hidden = Wk.size[1];
        const int v_hidden = Wv.size[1];
        if (q_hidden != k_hidden || k_hidden != v_hidden) return false;
        if (q_hidden % num_heads != 0) return false;

        // Output chain: Reshape4D -> Transpose([0,2,1,3]) -> Reshape3D, the
        // boundary of the fused block.
        Arg av_out = prog[av_matmul_idx]->outputs[0];
        int out_r4d = singleConsumer(av_out);
        if (!isReshape(prog, out_r4d)) return false;
        Arg out_r4d_out = prog[out_r4d]->outputs[0];
        int out_t4d = singleConsumer(out_r4d_out);
        if (!isTranspose(prog, out_t4d)) return false;
        TransposeLayer* out_tr =
            dynamic_cast<TransposeLayer*>(prog[out_t4d].get());
        if (!out_tr || out_tr->perm.size() != 4) return false;
        if (out_tr->perm[0] != 0 || out_tr->perm[1] != 2 ||
            out_tr->perm[2] != 1 || out_tr->perm[3] != 3) return false;
        Arg out_t4d_out = prog[out_t4d]->outputs[0];
        int out_r3d = singleConsumer(out_t4d_out);
        if (!isReshape(prog, out_r3d)) return false;

        // Combined [Q|K|V] weight along the output dim.
        Mat W_qkv;
        cv::hconcat(std::vector<Mat>{Wq, Wk, Wv}, W_qkv);
        Mat bias_qkv;
        if (!concatQKVBias({bq, bk, bv}, {q_hidden, k_hidden, v_hidden}, bias_qkv))
            return false;

        // scale (pre-softmax divisor) = 1/q_scale, since Q was pre-multiplied.
        const float param_scale = 1.f / q_scale;

        LayerParams attn_params;
        attn_params.name = prog[q_mm_idx]->name + "_fused_attention";
        attn_params.type = "Attention";
        attn_params.set("num_heads", num_heads);
        int qkv_sizes[3] = { q_hidden, k_hidden, v_hidden };
        attn_params.set("qkv_hidden_sizes", DictValue::arrayInt(qkv_sizes, 3));
        attn_params.set("scale", param_scale);
        attn_params.set("output_ndims", 3);
        attn_params.blobs.push_back(W_qkv);
        attn_params.blobs.push_back(bias_qkv);

        Ptr<LayerInfo> attn_layer =
            LayerFactory::createLayerInstance(attn_params.type, attn_params);
        if (!attn_layer) return false;
        attn_layer->inputs  = { shared_input };
        attn_layer->outputs = prog[out_r3d]->outputs;
        attn_layer->netimpl = netimpl;

        std::set<int> to_remove = {
            q_mm_idx, k_mm_idx, v_mm_idx,
            qk_matmul_idx, softmax_idx, av_matmul_idx,
            out_r4d, out_t4d, out_r3d
        };
        for (int op : consumed) to_remove.insert(op);

        std::set<int> attn_out_args;
        for (const Arg& o : attn_layer->outputs) attn_out_args.insert(o.idx);
        if (!removalIsSelfContained(prog, to_remove, attn_out_args))
            return false;

        for (int op : to_remove) removed_ops.insert(op);
        int insert_pos = *std::min_element(to_remove.begin(), to_remove.end());
        replacements.push_back({insert_pos, attn_layer});
        return true;
    }

    // Traces a Q*K^T operand back over an optional scale op to a Transpose with `perm`; `head` is that Transpose's output.
    bool headSplitInput(const vector<Ptr<LayerInfo>>& prog, Arg a, const vector<int>& perm,
                        Arg& head, QKScale& s, std::set<int>& chain_ops) const
    {
        s = QKScale();
        head = a;
        int p = producerOf(a);
        if (p < 0 || p >= (int)prog.size() || !prog[p]) return false;
        if (!isTranspose(prog, p)) {
            if (!dynamic_cast<NaryEltwiseLayer*>(prog[p].get()) || prog[p]->inputs.size() != 2) return false;
            int t = -1;
            for (int k = 0; k < 2 && t < 0; k++)
                if (isTranspose(prog, producerOf(prog[p]->inputs[k]))) t = k;
            if (t < 0) return false;
            head = prog[p]->inputs[t];
            if (skipScale(prog, head, s, chain_ops, nullptr) < 0 || s.op != p) return false;
            p = producerOf(head);
        }
        return static_cast<TransposeLayer*>(prog[p].get())->perm == perm;
    }

    // Q*K^T -> [scale] -> Softmax -> *V -> Transpose over head-split Q, K^T, V becomes one SDPA layer; the projections stay.
    bool tryFuseSDPA(const vector<Ptr<LayerInfo>>& prog, int softmax_idx,
                     std::set<int>& removed_ops,
                     vector<std::pair<int, Ptr<LayerInfo>>>& replacements)
    {
        if (removed_ops.count(softmax_idx)) return false;
        SoftmaxLayer* sm = dynamic_cast<SoftmaxLayer*>(prog[softmax_idx].get());
        if (!sm || sm->logSoftMax || prog[softmax_idx]->inputs.size() != 1) return false;

        // Back over an optional scalar Mul/Div to Q*K^T; parsePostQK below checks what was stepped over.
        int qk_idx = producerOf(prog[softmax_idx]->inputs[0]);
        if (qk_idx >= 0 && qk_idx < (int)prog.size() && prog[qk_idx] && !isMatMul(prog, qk_idx)) {
            const auto& in = prog[qk_idx]->inputs;
            const int data = in.size() == 2 && netimpl->isConstArg(in[0]) ? 1 : 0;
            qk_idx = in.empty() ? -1 : producerOf(in[data]);
        }
        if (!isMatMul(prog, qk_idx) || removed_ops.count(qk_idx) ||
            !prog[qk_idx]->blobs.empty() || prog[qk_idx]->inputs.size() != 2) return false;
        PostQK post;
        if (!parsePostQK(prog, qk_idx, /*allow_mask=*/false, post) || post.softmax != softmax_idx) return false;

        std::set<int> chain_ops, extra_ops;
        Arg q4, kT4;
        QKScale qs, ks;
        if (!headSplitInput(prog, prog[qk_idx]->inputs[0], {0, 2, 1, 3}, q4, qs, chain_ops)) return false;
        if (!headSplitInput(prog, prog[qk_idx]->inputs[1], {0, 2, 3, 1}, kT4, ks, chain_ops)) return false;

        const Arg sm_out = prog[softmax_idx]->outputs[0];
        const int av_idx = singleConsumer(sm_out);
        if (!isMatMul(prog, av_idx) || !prog[av_idx]->blobs.empty() || prog[av_idx]->inputs.size() != 2 ||
            prog[av_idx]->inputs[0].idx != sm_out.idx) return false;
        const Arg v4 = prog[av_idx]->inputs[1];
        const int vt = producerOf(v4);
        if (!isTranspose(prog, vt) ||
            static_cast<TransposeLayer*>(prog[vt].get())->perm != vector<int>{0, 2, 1, 3}) return false;

        const int out_tr = singleConsumer(prog[av_idx]->outputs[0]);
        if (!isTranspose(prog, out_tr) ||
            static_cast<TransposeLayer*>(prog[out_tr].get())->perm != vector<int>{0, 2, 1, 3}) return false;
        const Arg out_tr_out = prog[out_tr]->outputs[0];
        const int out_rs = findMatchingConsumer(prog, out_tr_out,
            [](LayerInfo* L){ return dynamic_cast<Reshape2Layer*>(L) != nullptr; }, &extra_ops);
        if (!isReshape(prog, out_rs) || prog[out_rs]->inputs.size() < 2) return false;

        // SDPA writes [B, S, H*Dv]: it replaces a head-merge Reshape whose first two dims are B and S, or feeds any const-shape Reshape left in place.
        std::set<int> to_remove = { qk_idx, softmax_idx, av_idx, out_tr };
        Arg sdpa_out;
        const Arg target = prog[out_rs]->inputs[1];
        const Arg av_out = prog[av_idx]->outputs[0];
        const bool const_target = netimpl->isConstArg(target);
        const int concat_idx = const_target ? -1 : producerOf(target);
        vector<int> tshape;
        if (const_target && !readConstInts(target, tshape)) return false;
        // B and S are dims 0, 1 of the merged [B, S, H, Dv] tensor, or dims 0, 2 of *V's [B, H, S, Dv] output.
        auto isBatchSeq = [&](const vector<Arg>& e) {
            return (isDimOf(prog, e[0], out_tr_out, 0) || isDimOf(prog, e[0], av_out, 0)) &&
                   (isDimOf(prog, e[1], out_tr_out, 1) || isDimOf(prog, e[1], av_out, 2));
        };
        if (const_target && tshape.size() == 3 && tshape[0] == 0 && tshape[1] == 0) {
            to_remove.insert(out_rs);
            sdpa_out = prog[out_rs]->outputs[0];
        } else if (concat_idx >= 0 && concat_idx < (int)prog.size() && prog[concat_idx] &&
                   dynamic_cast<Concat2Layer*>(prog[concat_idx].get()) && prog[concat_idx]->inputs.size() == 3 &&
                   isBatchSeq(prog[concat_idx]->inputs)) {
            to_remove.insert(out_rs);
            collectShapeChain(prog, concat_idx, extra_ops);
            sdpa_out = prog[out_rs]->outputs[0];
        } else if (const_target && extra_ops.empty() &&
                   std::find(tshape.begin(), tshape.end(), 0) == tshape.end()) {
            sdpa_out = out_tr_out;
        } else {
            return false;
        }

        float divisor;
        if (!combineScale(qs, ks, post, /*head_dim=*/-1, q4, divisor)) return false;

        LayerParams params;
        params.name = prog[qk_idx]->name + "_fused_sdpa";
        params.type = "SDPA";
        if (divisor > 0.f) params.set("scale", divisor);
        else               params.set("head_dim_scale", true);
        Ptr<LayerInfo> sdpa = LayerFactory::createLayerInstance(params.type, params);
        if (!sdpa) return false;
        sdpa->inputs  = { q4, kT4, v4 };
        sdpa->outputs = { sdpa_out };
        sdpa->netimpl = netimpl;

        for (int op : { qs.op, ks.op, post.scale_op })
            if (op >= 0) to_remove.insert(op);
        for (int op : extra_ops) to_remove.insert(op);
        addRemovableChainOps(prog, chain_ops, to_remove);
        for (int op : to_remove)
            if (removed_ops.count(op)) return false;

        std::set<int> out_args = { sdpa_out.idx };
        if (!removalIsSelfContained(prog, to_remove, out_args)) return false;

        for (int op : to_remove) removed_ops.insert(op);
        // Placed where *V was: V's Transpose may come after Q*K^T, so the lowest removed index can precede an input.
        replacements.push_back({av_idx, sdpa});
        return true;
    }

    // A removal set is self-contained if each removed op's output is re-produced by the
    // replacement layer or consumed only by other removed ops. A graph output, or an output
    // still feeding a surviving op, would otherwise lose its producer.
    bool removalIsSelfContained(const vector<Ptr<LayerInfo>>& prog,
                                const std::set<int>& to_remove,
                                const std::set<int>& produced_out_args) const
    {
        for (int op_idx : to_remove) {
            if (op_idx < 0 || op_idx >= (int)prog.size() || !prog[op_idx]) continue;
            for (const Arg& out : prog[op_idx]->outputs) {
                if (produced_out_args.count(out.idx)) continue;
                if (graph_outputs_.count(out.idx)) return false;
                auto cit = consumers_.find(out.idx);
                if (cit == consumers_.end()) continue;
                for (int c : cit->second)
                    if (!to_remove.count(c)) return false;
            }
        }
        return true;
    }

    bool fuseGraph(Ptr<Graph>& graph)
    {
        const vector<Ptr<LayerInfo>>& prog = graph->prog();
        size_t nops = prog.size();

        producer_.clear();
        consumers_.clear();
        for (size_t i = 0; i < nops; i++) {
            if (!prog[i]) continue;
            for (Arg out : prog[i]->outputs)
                producer_[out.idx] = (int)i;
            for (Arg inp : prog[i]->inputs)
                consumers_[inp.idx].push_back((int)i);
        }

        // A graph output is exported, not consumed by any op, so it never enters consumers_;
        // track it separately so a removed op cannot silently drop an exported arg.
        graph_outputs_.clear();
        for (Arg out : graph->outputs())
            graph_outputs_.insert(out.idx);

        bool modified = false;
        std::set<int> removed_ops;

        // Pass 1: combined-QKV blocks.
        for (size_t i = 0; i < nops; i++) {
            if (!prog[i] || removed_ops.count((int)i)) continue;
            if (tryFuseCombinedQKV(prog, (int)i, removed_ops, attention_replacements_))
                modified = true;
        }

        // Pass 2: the original 3-separate-projection path.
        std::map<int, vector<int>> qkv_candidates;
        for (size_t i = 0; i < nops; i++) {
            if (!prog[i] || removed_ops.count((int)i)) continue;
            if (!isProjCandidate(prog[i])) continue;
            Arg inp = prog[i]->inputs[0];
            qkv_candidates[inp.idx].push_back((int)i);
        }

        for (auto& candidate : qkv_candidates) {
            auto matmul_indices = candidate.second;
            if (matmul_indices.size() < 3) continue;

            for (size_t qi = 0; qi + 2 < matmul_indices.size(); qi++)
            for (size_t ki = qi + 1; ki + 1 < matmul_indices.size(); ki++)
            for (size_t vi = ki + 1; vi < matmul_indices.size(); vi++)
            {
                int indices[3] = { matmul_indices[qi], matmul_indices[ki], matmul_indices[vi] };

                bool any_removed = false;
                for (int idx : indices)
                    if (removed_ops.count(idx)) any_removed = true;
                if (any_removed) continue;

                bool shapes_ok = true;
                for (int k = 0; k < 3; k++) {
                    const Mat& w = prog[indices[k]]->blobs[0];
                    if (w.dims != 2) { shapes_ok = false; break; }
                }
                if (!shapes_ok) continue;

                int reshape_idx[3], num_heads[3], transpose_idx[3];
                vector<int> perms[3];
                std::set<int> extra_ops;
                bool pattern_ok = true;
                for (int k = 0; k < 3; k++) {
                    transpose_idx[k] = followProjChain(prog, indices[k],
                                                       &reshape_idx[k],
                                                       &num_heads[k], &perms[k],
                                                       &extra_ops);
                    if (transpose_idx[k] < 0) { pattern_ok = false; break; }
                }
                if (!pattern_ok) continue;
                if (num_heads[0] != num_heads[1] || num_heads[1] != num_heads[2])
                    continue;

                int k_slot = -1;
                vector<int> perm_k = {0, 2, 3, 1};
                for (int k = 0; k < 3; k++)
                    if (perms[k] == perm_k) k_slot = k;
                if (k_slot < 0) continue;

                int remaining[2];
                { int j = 0; for (int k = 0; k < 3; k++) if (k != k_slot) remaining[j++] = k; }

                // Q and K may each carry a scale on the way to Q*K^T; Shape consumers feed the run-time scale chain.
                std::set<int> chain_ops;
                QKScale ks;
                const int qk_matmul_idx = skipScale(prog, prog[transpose_idx[k_slot]]->outputs[0],
                                                    ks, chain_ops, &extra_ops);
                if (!isMatMul(prog, qk_matmul_idx) || prog[qk_matmul_idx]->inputs.size() != 2) continue;

                int q_slot = -1, v_slot = -1;
                for (int attempt = 0; attempt < 2; attempt++) {
                    q_slot = remaining[attempt];
                    v_slot = remaining[1 - attempt];

                    QKScale qs;
                    std::set<int> qk_chain_ops = chain_ops;
                    Arg q_tr_out = prog[transpose_idx[q_slot]]->outputs[0];
                    if (skipScale(prog, q_tr_out, qs, qk_chain_ops, &extra_ops) != qk_matmul_idx) continue;
                    const Arg q_in = qs.op >= 0 ? prog[qs.op]->outputs[0] : q_tr_out;
                    if (prog[qk_matmul_idx]->inputs[0].idx != q_in.idx) continue;

                    // Q*K^T -> [Mul/Div] -> [Add(mask)] -> Softmax covers BERT, ViT and SigLIP-style scaling.
                    PostQK post;
                    if (!parsePostQK(prog, qk_matmul_idx, /*allow_mask=*/true, post)) continue;
                    const int softmax_idx = post.softmax;

                    Arg softmax_out = prog[softmax_idx]->outputs[0];
                    int attnv_matmul_idx = singleConsumer(softmax_out);
                    if (!isMatMul(prog, attnv_matmul_idx)) continue;

                    Arg v_tr_out = prog[transpose_idx[v_slot]]->outputs[0];
                    const auto& av_inputs = prog[attnv_matmul_idx]->inputs;
                    bool v_connected = false;
                    for (auto& ai : av_inputs)
                        if (ai.idx == v_tr_out.idx) v_connected = true;
                    if (!v_connected) continue;

                    Arg attnv_out = prog[attnv_matmul_idx]->outputs[0];
                    int out_transpose_idx = singleConsumer(attnv_out);
                    if (!isTranspose(prog, out_transpose_idx)) continue;

                    Arg out_tr_out = prog[out_transpose_idx]->outputs[0];
                    int out_reshape_idx = findMatchingConsumer(prog, out_tr_out,
                        [](LayerInfo* L){ return dynamic_cast<Reshape2Layer*>(L) != nullptr; },
                        &extra_ops);
                    if (!isReshape(prog, out_reshape_idx)) continue;

                    const auto& or_inputs = prog[out_reshape_idx]->inputs;
                    if (or_inputs.size() >= 2 && !netimpl->isConstArg(or_inputs[1])) {
                        auto it = producer_.find(or_inputs[1].idx);
                        if (it != producer_.end())
                            collectShapeChain(prog, it->second, extra_ops);
                    }

                    // Normalize each projection weight to [K, N] regardless of
                    // whether the source op is MatMul or Gemm-with-trans_b.
                    Mat Wq = getProjWeight(prog[indices[q_slot]]);
                    Mat Wk = getProjWeight(prog[indices[k_slot]]);
                    Mat Wv = getProjWeight(prog[indices[v_slot]]);
                    int q_hidden = Wq.size[1];
                    int k_hidden = Wk.size[1];
                    int v_hidden = Wv.size[1];

                    Mat W_qkv;
                    cv::hconcat(std::vector<Mat>{Wq, Wk, Wv}, W_qkv);

                    auto projBias = [&](int slot) {
                        const auto& b = prog[indices[slot]]->blobs;
                        return b.size() >= 2 ? b[1] : Mat();
                    };
                    Mat bias_qkv;
                    if (!concatQKVBias({projBias(q_slot), projBias(k_slot), projBias(v_slot)},
                                       {q_hidden, k_hidden, v_hidden}, bias_qkv))
                        continue;

                    if (num_heads[0] <= 0 || q_hidden % num_heads[0] != 0) continue;
                    float param_scale;
                    if (!combineScale(qs, ks, post, q_hidden / num_heads[0], q_tr_out, param_scale)) continue;

                    LayerParams attn_params;
                    attn_params.name = prog[indices[q_slot]]->name + "_fused_attention";
                    attn_params.type = "Attention";
                    attn_params.set("num_heads", num_heads[0]);
                    DictValue qkv_sizes_param = DictValue::arrayInt(
                        std::vector<int>{q_hidden, k_hidden, v_hidden}.data(), 3);
                    attn_params.set("qkv_hidden_sizes", qkv_sizes_param);
                    attn_params.set("scale", param_scale);
                    attn_params.set("output_ndims", 3);

                    attn_params.blobs.push_back(W_qkv);
                    attn_params.blobs.push_back(bias_qkv);

                    Ptr<LayerInfo> attn_layer = LayerFactory::createLayerInstance(
                        attn_params.type, attn_params);
                    CV_Assert(attn_layer);

                    Arg shared_input = prog[indices[0]]->inputs[0];
                    if (post.has_mask)
                        attn_layer->inputs = { shared_input, post.mask };
                    else
                        attn_layer->inputs = { shared_input };
                    attn_layer->outputs = prog[out_reshape_idx]->outputs;
                    attn_layer->netimpl = netimpl;

                    std::set<int> to_remove = {
                        indices[0], indices[1], indices[2],
                        reshape_idx[0], reshape_idx[1], reshape_idx[2],
                        transpose_idx[0], transpose_idx[1], transpose_idx[2],
                        qk_matmul_idx, softmax_idx, attnv_matmul_idx,
                        out_transpose_idx, out_reshape_idx
                    };
                    for (int op : { qs.op, ks.op, post.scale_op, post.add_op })
                        if (op >= 0) to_remove.insert(op);
                    for (int op : extra_ops) to_remove.insert(op);
                    addRemovableChainOps(prog, qk_chain_ops, to_remove);

                    std::set<int> attn_out_args;
                    for (const Arg& o : attn_layer->outputs) attn_out_args.insert(o.idx);
                    if (!removalIsSelfContained(prog, to_remove, attn_out_args))
                        continue;

                    for (int op : to_remove)
                        removed_ops.insert(op);

                    // Placed where *V was: a mask built after the projections must already exist when Attention runs.
                    attention_replacements_.push_back({attnv_matmul_idx, attn_layer});
                    modified = true;
                    break;
                }
            }
        }

        // Pass 3: anchored on each Softmax.
        for (size_t i = 0; i < nops; i++) {
            if (!prog[i] || removed_ops.count((int)i)) continue;
            if (prog[i]->type != "Softmax") continue;
            if (tryFuseClipAttention(prog, (int)i, removed_ops,
                                     attention_replacements_))
                modified = true;
        }

        // Pass 4: blocks with Q, K and V already split into heads but not one shared projection input.
        if (netimpl->preferableBackend == DNN_BACKEND_OPENCV && netimpl->preferableTarget == DNN_TARGET_CPU) {
            for (size_t i = 0; i < nops; i++) {
                if (!prog[i] || removed_ops.count((int)i) || prog[i]->type != "Softmax") continue;
                if (tryFuseSDPA(prog, (int)i, removed_ops, attention_replacements_))
                    modified = true;
            }
        }

        if (modified) {
            vector<Ptr<LayerInfo>> newprog;
            std::sort(attention_replacements_.begin(), attention_replacements_.end(),
                      [](auto& a, auto& b) { return a.first < b.first; });

            size_t repl_idx = 0;
            for (size_t i = 0; i < nops; i++) {
                while (repl_idx < attention_replacements_.size() &&
                       attention_replacements_[repl_idx].first == (int)i) {
                    newprog.push_back(attention_replacements_[repl_idx].second);
                    repl_idx++;
                }
                if (removed_ops.count((int)i))
                    continue;
                newprog.push_back(prog[i]);
            }
            graph->setProg(newprog);
        }

        return modified;
    }

    Net::Impl* netimpl;

private:
    std::map<int, int> producer_;
    std::map<int, vector<int>> consumers_;
    std::set<int> graph_outputs_;
    vector<std::pair<int, Ptr<LayerInfo>>> attention_replacements_;
};

void Net::Impl::fuseAttention()
{
    ModelFusionAttention attnFusion(this);
    attnFusion.fuse();
}

CV__DNN_INLINE_NS_END
}}

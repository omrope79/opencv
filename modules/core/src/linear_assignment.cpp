// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Jonker-Volgenant rectangular assignment, implemented from:
//   D. F. Crouse, "On implementing 2D rectangular assignment algorithms",
//   IEEE Trans. Aerospace and Electronic Systems 52(4), 2016.
//   R. Jonker, A. Volgenant, "A shortest augmenting path algorithm for dense and sparse
//   linear assignment problems", Computing 38, 1987.

#include "precomp.hpp"

#include <algorithm>
#include <limits>

namespace cv {

// cvIsNaN/cvIsInf inspect the bit pattern, so unlike std::isfinite they still work under
// -ffast-math.
static inline bool isFiniteVal(double x)
{
    return !cvIsNaN(x) && !cvIsInf(x);
}

/*
    Assigns every row of `work` to a distinct column, minimising the total. `work` must have
    rows <= cols and contain finite values only. On return colOfRow[i] is the column taken by
    row i.

    Each pass grows the matching by one row, running a Dijkstra search over reduced costs
    work(i,j) - u[i] - v[j]. The duals keep every reduced cost non-negative and every matched
    pair at exactly zero, which is what makes the search a shortest-path problem and the result
    globally optimal rather than merely greedy.
*/
static void solveJV(const Mat& work, std::vector<int>& colOfRow)
{
    const int nrows = work.rows;
    const int ncols = work.cols;
    CV_DbgAssert(nrows <= ncols);

    std::vector<double> u((size_t)nrows, 0.0);
    std::vector<double> v((size_t)ncols, 0.0);
    std::vector<int> rowOfCol((size_t)ncols, -1);

    std::vector<double> dist((size_t)ncols);
    std::vector<int> prevRow((size_t)ncols);
    std::vector<uchar> labelled((size_t)ncols);

    colOfRow.assign((size_t)nrows, -1);

    for (int freeRow = 0; freeRow < nrows; freeRow++)
    {
        std::fill(dist.begin(), dist.end(), std::numeric_limits<double>::max());
        std::fill(labelled.begin(), labelled.end(), (uchar)0);

        int row = freeRow;      // row currently being expanded
        double delta = 0.0;     // length of the shortest path found so far
        int sink = -1;          // free column that ends the augmenting path

        while (sink < 0)
        {
            const double* rowPtr = work.ptr<double>(row);
            for (int j = 0; j < ncols; j++)
            {
                if (labelled[j])
                    continue;
                const double cand = delta + rowPtr[j] - u[row] - v[j];
                if (cand < dist[j])
                {
                    dist[j] = cand;
                    prevRow[j] = row;
                }
            }

            int next = -1;
            for (int j = 0; j < ncols; j++)
            {
                if (!labelled[j] && (next < 0 || dist[j] < dist[next]))
                    next = j;
            }
            // ncols >= nrows leaves at least one column free, and every cost is finite, so the
            // search can always reach it.
            CV_Assert(next >= 0 && dist[next] < std::numeric_limits<double>::max());

            labelled[next] = 1;
            delta = dist[next];

            if (rowOfCol[next] < 0)
                sink = next;
            else
                row = rowOfCol[next];   // occupied, so keep going through its row
        }

        // Shift the duals so the path edges land at zero reduced cost while the pairs already
        // matched stay there. Must run before the matching below is rewritten, since it reads
        // rowOfCol.
        u[freeRow] += delta;
        for (int j = 0; j < ncols; j++)
        {
            if (labelled[j] && j != sink)
            {
                const double shift = delta - dist[j];
                v[j] -= shift;
                u[rowOfCol[j]] += shift;
            }
        }

        // Walk the path back from the free column, flipping each edge.
        for (int j = sink;;)
        {
            const int i = prevRow[j];
            rowOfCol[j] = i;
            const int jPrev = colOfRow[i];
            colOfRow[i] = j;
            if (i == freeRow)
                break;
            j = jPrev;
        }
    }
}

double linearAssignment(InputArray _cost, std::vector<int>& assignment, double costThreshold)
{
    CV_INSTRUMENT_REGION();

    Mat cost = _cost.getMat();
    CV_Assert(cost.dims <= 2);

    assignment.assign((size_t)cost.rows, -1);
    if (cost.empty())
        return 0.0;

    CV_CheckType(cost.type(), cost.type() == CV_32FC1 || cost.type() == CV_64FC1,
                 "cost must be a single-channel floating point matrix");
    CV_Assert(!cvIsNaN(costThreshold));

    // The solver assigns every row, so it needs rows <= cols. Transposing also keeps the answer
    // independent of the orientation: the unassignment term is charged min(M,N) - matched times,
    // which is symmetric, so solving A and A transposed must agree.
    const bool transposed = cost.rows > cost.cols;
    Mat src;
    if (transposed)
        cv::transpose(cost, src);
    else
        src = cost;

    Mat orig;
    src.convertTo(orig, CV_64F);
    const int nrows = orig.rows;
    const int nreal = orig.cols;

    // Test for finiteness FIRST. NaN > costThreshold is false, so a bare comparison would let a
    // NaN through as allowed and poison every sum downstream.
    Mat allowed(nrows, nreal, CV_8U);
    double absSum = 0.0;
    for (int i = 0; i < nrows; i++)
    {
        const double* o = orig.ptr<double>(i);
        uchar* a = allowed.ptr<uchar>(i);
        for (int j = 0; j < nreal; j++)
        {
            const bool ok = isFiniteVal(o[j]) && o[j] <= costThreshold;
            a[j] = ok ? 1 : 0;
            if (ok)
                absSum += std::abs(o[j]);
        }
    }

    // Price of leaving a row unmatched. Beyond 2*absSum + 1 the price already dominates any
    // rearrangement of the real costs, so every larger threshold gives the same matching and we
    // clamp to keep the arithmetic small -- this is what handles the DBL_MAX default.
    //
    // The factor of 2 is required. Going from k matched pairs to k+1 augments along a path,
    // adding p+1 edges and dropping p, so the change in real cost is bounded by twice the total
    // and not by a single edge. absSum + 1 is not enough; do not "simplify" it.
    double dummy = 2.0 * absSum + 1.0;
    if (costThreshold < dummy)
        dummy = costThreshold;
    CV_Assert(isFiniteVal(dummy));

    // One dummy column per row, so a dummy is never contested. That is why a forbidden cell only
    // has to lose to a dummy: a row sitting on one can always move to a free dummy, which is a
    // single-row swap rather than an augmenting path, so dummy + 1 suffices.
    const double forbidden = dummy + 1.0;
    CV_Assert(isFiniteVal(forbidden));

    Mat work(nrows, nreal + nrows, CV_64F);
    for (int i = 0; i < nrows; i++)
    {
        const double* o = orig.ptr<double>(i);
        const uchar* a = allowed.ptr<uchar>(i);
        double* w = work.ptr<double>(i);
        for (int j = 0; j < nreal; j++)
            w[j] = a[j] ? o[j] : forbidden;
        for (int j = nreal; j < nreal + nrows; j++)
            w[j] = dummy;
    }

    std::vector<int> colOfRow;
    solveJV(work, colOfRow);

    double total = 0.0;
    for (int i = 0; i < nrows; i++)
    {
        const int j = colOfRow[i];
        if (j < 0 || j >= nreal || !allowed.at<uchar>(i, j))
            continue;   // landed on a dummy column or a forbidden cell, so it stays unmatched
        total += orig.at<double>(i, j);
        if (transposed)
            assignment[(size_t)j] = i;
        else
            assignment[(size_t)i] = j;
    }
    return total;
}

} // namespace cv

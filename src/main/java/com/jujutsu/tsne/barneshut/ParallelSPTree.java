package com.jujutsu.tsne.barneshut;

import java.util.stream.IntStream;

public class ParallelSPTree extends SPTree {

    public ParallelSPTree(int D, double[] inp_data, int N) {
        super(D, inp_data, N);
    }

    public ParallelSPTree(int D, double[] inp_data, int N, double[] inp_corner, double[] inp_width) {
        super(D, inp_data, N, inp_corner, inp_width);
    }

    public ParallelSPTree(int D, double[] inp_data, double[] inp_corner, double[] inp_width) {
        super(D, inp_data, inp_corner, inp_width);
    }

    public ParallelSPTree(SPTree inp_parent, int D, double[] inp_data, double[] inp_corner, double[] inp_width) {
        super(inp_parent, D, inp_data, inp_corner, inp_width);
    }

    public ParallelSPTree(SPTree inp_parent, int D, double[] inp_data, int N, double[] inp_corner, double[] inp_width) {
        super(inp_parent, D, inp_data, N, inp_corner, inp_width);
    }

    @Override
    SPTree[] getTreeArray(int no_children) {
        return new ParallelSPTree[no_children];
    }

    @Override
    SPTree getNewTree(SPTree root, double[] new_corner, double[] new_width) {
        return new ParallelSPTree(root, dimension, data, new_corner, new_width);
    }

    // Computes edge forces
    //
    // Runs in parallel over the rows of the sparse similarity matrix: row n only ever writes to
    // pos_f[n * dimension .. n * dimension + dimension - 1], so the rows are independent. The
    // difference vector is recomputed instead of being buffered, which for the two or three output
    // dimensions of t-SNE is cheaper than a per thread buffer.
    @Override
    void computeEdgeForces(int[] row_P, int[] col_P, double[] val_P, int N, double[] pos_f)
    {
        final int dim = dimension;
        final double[] Y = data;
        IntStream.range(0, N).parallel().forEach(n -> {
            final int ind1 = n * dim;
            for (int i = row_P[n]; i < row_P[n + 1]; i++)
            {
                // Compute pairwise distance and Q-value
                double D = 1.0;
                final int ind2 = col_P[i] * dim;
                for (int d = 0; d < dim; d++)
                {
                    double diff = Y[ind1 + d] - Y[ind2 + d];
                    D += diff * diff;
                }
                D = val_P[i] / D;

                // Sum positive force
                for (int d = 0; d < dim; d++) {
                    pos_f[ind1 + d] += D * (Y[ind1 + d] - Y[ind2 + d]);
                }
            }
        });
    }
}

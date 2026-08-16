/*
 *
 * This Java port of Barnes Hut t-SNE is Copyright (c) Leif Jonsson 2016 and 
 * Copyright (c) 2014, Laurens van der Maaten (Delft University of Technology)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    This product includes software developed by the Delft University of Technology.
 * 4. Neither the name of the Delft University of Technology nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL LAURENS VAN DER MAATEN BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 */
package com.jujutsu.tsne.barneshut;

import static java.lang.Math.exp;
import static java.lang.Math.log;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.stream.DoubleStream;

import com.jujutsu.tsne.TSneConfiguration;
import com.jujutsu.tsne.progress.ConsoleProgressListener;
import com.jujutsu.tsne.progress.ProgressListener;
import com.jujutsu.tsne.progress.TSneProgress;
import com.jujutsu.utils.MatrixOps;

import math.linalg.JacobiPCA;
import math.linalg.TruncatedPCA;

public class BHTSne implements BarnesHutTSne {

    /** Name of the progress task reported by the main training loop. */
    public static final String TASK_GRADIENT_DESCENT = "Calc T-Sne";

    protected final Distance distance = new EuclideanDistance();
    protected volatile boolean abort = false;

    /**
     * Normalization term of the Q distribution, as computed by the most recent
     * {@link #computeGradient} call. It belongs to the embedding that call was given: once Y has been
     * moved by {@link #updateGradient}, this is the normalization of the previous state.
     */
    protected double lastSumQ;

    /**
     * Seed for the vantage point choice of the ball tree, or {@code null} to leave it arbitrary,
     * which is the default and what this has always done.
     * <p>
     * The embedding does not depend on it - the kNN search returns the same neighbours whatever shape
     * the tree happens to take - but the time to find them does, by as much as a factor of 1.6 at
     * {@code N = 60 000}. Anything that measures this phase has to pin it down first, which is what
     * this is for. Package private on purpose: it is a handle for measurement, not API.
     */
    Long vpTreeSeed = null;

    /**
     * Work arrays of the gradient, held across the iterations of a run so that they are allocated once
     * instead of once per iteration. They fit one shape only, so {@link #computeGradient} reallocates
     * them whenever it is called with an {@code N} or a {@code D} they were not sized for.
     * <p>
     * {@code sum_Q}, {@code pos_f} and {@code neg_f} are accumulated into and have to be cleared before
     * each gradient step. {@code buff} is scratch that {@code SPTree.computeNonEdgeForces} fills before
     * it reads, so it is not cleared.
     */
    double[] sum_Q = null;
    double[] pos_f = null;
    double[][] neg_f = null;
    double[][] buff = null;

    /**
     * Makes the work arrays fit a gradient step of {@code N} points in {@code D} dimensions, and clears
     * the three that are accumulated into.
     * <p>
     * Keying this on the shape rather than on {@code pos_f == null} matters because the arrays outlive
     * a single run: a second, larger run on the same instance would otherwise index past the end of the
     * first one's arrays. Measured, {@code N} going from 300 to 600 threw an
     * {@code ArrayIndexOutOfBoundsException}, and so did {@code no_dims} going from 2 to 3. A smaller
     * second run happened to be correct - the arrays are cleared in full and the surplus is never read
     * - but it carried the larger allocation for the rest of the instance's life.
     * <p>
     * {@code pos_f.length} pins {@code D} down for any {@code N >= 1}, which avoids reading
     * {@code neg_f[0]} on a hypothetical empty input.
     */
    final void prepareWorkArrays(int N, int D) {
        if(neg_f == null || neg_f.length != N || pos_f.length != N * D) {
            sum_Q = new double[N];
            pos_f = new double[N * D];
            neg_f = new double[N][D];
            buff = new double[N][D];
            return;
        }
        Arrays.fill(sum_Q, 0.0);
        Arrays.fill(pos_f, 0.0);
        for(int n = 0; n < N; n++) {
            Arrays.fill(neg_f[n], 0.0);
        }
    }

    @Override
    public double[][] tsne(TSneConfiguration config) {
        return run(config);
    }

    /**
     * The leading {@code k} columns of an already reduced data set, mean centered - which is what a
     * PCA over that data set computes, since its principal directions are the canonical basis
     * vectors. Callers must have established that the columns are principal components in
     * descending order.
     */
    static double [][] leadingColumns(double [][] x, int k) {
        final int rows = x.length;
        double [] mean = new double[k];
        for (int i = 0; i < rows; i++) {
            for (int c = 0; c < k; c++) mean[c] += x[i][c];
        }
        for (int c = 0; c < k; c++) mean[c] /= rows;

        double [][] leading = new double[rows][k];
        for (int i = 0; i < rows; i++) {
            for (int c = 0; c < k; c++) leading[i][c] = x[i][c] - mean[c];
        }
        return leading;
    }

    /**
     * Feature count above which the truncated PCA is used. A full decomposition computes all
     * components in O(m n^2) + O(sweeps n^3) even when a handful are needed; measured at m = 5000 the
     * two methods break even around n = 32, and the truncated one is 9x to 25x ahead at n = 784
     * depending on the spectrum. Below the threshold the exact path costs at most about 20 ms, so
     * there is nothing to gain by approximating there.
     */
    private static final int TRUNCATED_PCA_MIN_DIMS = 64;

    /**
     * Fraction of the leading component's spread below which a trailing component is treated as
     * carrying no usable direction. Data with one dominant direction and noise behind it reaches
     * 6e-3 here.
     */
    private static final double NEGLIGIBLE_SPREAD = 1e-2;

    /**
     * Oversampling and subspace iterations of the input reduction. Unlike the embedding
     * initialization this keeps dozens of components, so the tail of the spectrum is nearly tied and
     * a stability test on the Ritz values would never be satisfied - a fixed count is the honest
     * form of the same decision. Chosen by measurement on 2500 x 784 MNIST reduced to 55 components:
     * this pair reproduces 99.96 % of the captured variance, preserves pairwise distances to 1.1e-3
     * in the median, and keeps 98.8 % of the 60 nearest neighbours, while making the reduction about
     * 6.6x cheaper. Cheaper settings measured no worse end to end, but leave less margin.
     */
    private static final int REDUCTION_OVERSAMPLING = 10;
    private static final int REDUCTION_ITERATIONS   = 6;

    /**
     * Largest share of the features the reduction may keep and still use the truncated method. The
     * truncated cost grows with the size of the search subspace, so once a large fraction of the
     * components is wanted there is nothing left to save and the exact decomposition is both faster
     * and exact.
     */
    private static final int REDUCTION_MAX_SUBSPACE_FRACTION = 4;

    /**
     * Reduces {@code x} to its leading {@code k} principal components - the transformation applied
     * when {@code usePca()} is set.
     * <p>
     * These components feed the kNN search and therefore the result, not just a starting point, so
     * the accuracy question is different from the one at {@link #initialComponents}. It is also more
     * forgiving than it looks: everything downstream sees only <em>distances</em> between the
     * reduced samples, and those are invariant under a rotation inside the retained subspace. The
     * truncated method has to find the right subspace, not the individual directions inside it -
     * which is what makes a nearly tied tail harmless here.
     * <p>
     * Measured end to end on the demo path, the truncated reduction reproduces the neighbourhoods of
     * the original high dimensional data as faithfully as the exact one (mean overlap of the 10
     * nearest neighbours in the embedding with those in the input: 0.4836 truncated against 0.4810
     * exact).
     */
    static double [][] reduceInput(double [][] x, int k) {
        if(worthTruncating(x[0].length, k)) {
            return TruncatedPCA.fixedIterations(REDUCTION_OVERSAMPLING, REDUCTION_ITERATIONS).pca(x, k);
        }
        return new JacobiPCA().pca(x, k);
    }

    /** Whether a truncated decomposition into {@code k} of {@code n} components can pay off. */
    private static boolean worthTruncating(int n, int k) {
        return n > TRUNCATED_PCA_MIN_DIMS
                && (long) (k + REDUCTION_OVERSAMPLING) * REDUCTION_MAX_SUBSPACE_FRACTION <= n;
    }

    /**
     * Where the C++ reference implementation ends the early phase, at its default of 1000 iterations
     * - a quarter of the run.
     */
    private static final int REFERENCE_EARLY_PHASE_END = 250;

    /**
     * Last iteration of the early phase: up to and including it the gradient sees the exaggerated
     * P values and the low momentum, afterwards it does not.
     * <p>
     * The reference implementation hard codes 250 against its default of 1000 iterations. Taken
     * literally that breaks every shorter run: at {@code maxIter <= 250} the exaggeration is never
     * switched off, the momentum never reaches its final value, and the embedding is returned in its
     * inflated state with no warning at all. Measured, a 200 iteration run produced embeddings whose
     * neighbourhoods had almost nothing to do with the input.
     * <p>
     * The early phase is therefore a quarter of the run, which reproduces the reference exactly at
     * 1000 iterations and beyond, and keeps the same proportion below. Runs between 250 and 1000
     * iterations do change: they used to exaggerate for a larger share of the schedule than the
     * reference intends.
     */
    static int earlyPhaseEnd(int maxIter) {
        return Math.min(REFERENCE_EARLY_PHASE_END, maxIter / 4);
    }

    /**
     * The leading {@code k} principal components of {@code x}, used to initialize the embedding.
     * Uses the truncated method where it pays off, and falls back to the exact decomposition when
     * the truncated one neither reaches its tolerance nor is left with directions that do not
     * matter anyway.
     */
    static double [][] initialComponents(double [][] x, int k) {
        if(x[0].length > TRUNCATED_PCA_MIN_DIMS) {
            TruncatedPCA truncated = new TruncatedPCA();
            double [][] projected = truncated.pca(x, k);
            if(truncated.converged() || onlyTheLeadingComponentCarriesVariance(projected)) {
                return projected;
            }
        }
        return new JacobiPCA().pca(x, k);
    }

    /**
     * Whether every component behind the first carries a negligible share of its spread. Where that
     * holds, those directions are not determined by the data - an exact decomposition picks an
     * equally arbitrary one - and an embedding initialization that is scaled down to 1e-4 anyway
     * gains nothing from resolving them.
     */
    private static boolean onlyTheLeadingComponentCarriesVariance(double [][] projected) {
        if(projected[0].length < 2) return true;
        double leading = spread(projected, 0);
        if(!(leading > 0.0)) return false;
        for (int c = 1; c < projected[0].length; c++) {
            if(spread(projected, c) > NEGLIGIBLE_SPREAD * leading) return false;
        }
        return true;
    }

    private static double spread(double [][] projected, int c) {
        final int rows = projected.length;
        double mean = 0.0;
        for (int i = 0; i < rows; i++) mean += projected[i][c];
        mean /= rows;
        double variance = 0.0;
        for (int i = 0; i < rows; i++) {
            double d = projected[i][c] - mean;
            variance += d * d;
        }
        return Math.sqrt(variance / rows);
    }

    private double[] flatten(double[][] x) {
        int noCols = x[0].length;
        double [] flat = new double[x.length*x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                flat[i*noCols+j] = x[i][j];
            }
        }
        return flat;
    }

    private double [][] expand(double[]x, int N, int D) {
        double [][] expanded = new double[N][D];
        for (int row = 0; row < N; row++) {
            for (int col = 0; col < D; col++) {
                expanded[row][col] = x[row*D+col];
            }
        }
        return expanded;
    }

    public static double sd(double[] array) {
        // get the sum of array
        double sum = 0.0;
        for (double i : array) {
            sum += i;
        }

        // get the mean of array
        int length = array.length;
        double mean = sum / length;

        // calculate the standard deviation
        double standardDeviation = 0.0;
        for (double num : array) {
            standardDeviation += Math.pow(num - mean, 2);
        }

        return Math.sqrt(standardDeviation / length);
    }

    static double sign_tsne(double x) { return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); }

    public static void write (String filename, double[]arr) throws IOException {
        BufferedWriter ow = null;
        ow = new BufferedWriter(new FileWriter(filename));
        for (int i = 0; i < arr.length; i++) {
 
            ow.write(arr[i]+"");
            ow.newLine();
        }
        ow.flush();
        ow.close();
    }

    public static void write (String filename, int[]arr) throws IOException {
        BufferedWriter ow = null;
        ow = new BufferedWriter(new FileWriter(filename));
        for (int i = 0; i < arr.length; i++) {
 
            ow.write(arr[i]+"");
            ow.newLine();
        }
        ow.flush();
        ow.close();
    }

    public static void write (String filename, double [][] arr) throws IOException {
        BufferedWriter ow = null;
        ow = new BufferedWriter(new FileWriter(filename));
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[i].length; j++) {
                ow.write(arr[i][j]+"");
                if (j+1 == arr[i].length) {
                    ow.newLine();
                } else {
                    ow.write(",");
                }
            }
        }
        ow.flush();
        ow.close();
    }

    public static void write (String filename, List<Double> arr) throws IOException {
        BufferedWriter ow = null;
        ow = new BufferedWriter(new FileWriter(filename));
        for (int i = 0; i < arr.size(); i++) {
 
            ow.write(arr.get(i)+"");
            ow.newLine();
        }
        ow.flush();
        ow.close();
    }

    // Perform t-SNE
    double [][] run(TSneConfiguration parameterObject) {
        // silent() only controls the built-in console rendering; listeners registered from the
        // outside via TSneProgress are notified in either case
        ProgressListener console = parameterObject.silent() ? null : new ConsoleProgressListener();
        if(console != null) {
            TSneProgress.addProgressListener(console);
        }
        try {
            return runInternal(parameterObject);
        } finally {
            if(console != null) {
                TSneProgress.removeProgressListener(console);
            }
        }
    }

    private double [][] runInternal(TSneConfiguration parameterObject) {
        int D = parameterObject.getXStartDim();
        double[][] Xin = parameterObject.getXin();

        // theta == 0.0 asks for the exact O(N^2) formulation, which this class does not implement.
        if(parameterObject.getTheta() == .0) {
            throw new IllegalArgumentException(
                    "theta == 0.0 requests exact t-SNE, which the Barnes-Hut implementation does not provide. Use a positive theta; 0.5 is the usual choice.");
        }

        int N = parameterObject.getNrRows();
        int no_dims = parameterObject.getOutputDims();

        boolean reduced = false;
        if(parameterObject.usePca() && D > parameterObject.getInitialDims() && parameterObject.getInitialDims() > 0) {
            Xin = reduceInput(Xin, parameterObject.getInitialDims());
            D = parameterObject.getInitialDims();
            reduced = true;
            System.out.println("X:Shape after PCA is = " + Xin.length + " x " + Xin[0].length);
        }

        double [] X = flatten(Xin);

        // The reduction above already expresses the data in its own principal basis, so its
        // covariance is diagonal with descending entries. The principal directions of the reduced
        // data are then the canonical basis vectors, which canonicalizeSigns fixes to +e_k, and a
        // second decomposition would only reproduce the leading columns it already has. This holds
        // for the truncated reduction too: it returns its components in descending order under the
        // same sign convention, and the leading ones are the accurate end of its output.
        double [][] Yinit = (reduced && D >= no_dims) ? leadingColumns(Xin, no_dims)
                                                     : initialComponents(Xin, no_dims);
        double [] pc1 = MatrixOps.transposeSerial(Yinit)[0];
        double sd = sd(pc1);

        // Init y with PCA (from: The art of using t-SNE for single-cell transcriptomics)
        double [] Y = new double[N*no_dims];
        for(int n = 0; n < N; n++) {
            for(int d = 0; d < no_dims; d++) {
                Y[n*no_dims+d] = (Yinit[n][d] / sd) * 0.0001;
            }
        }
        System.out.println("X:Shape is = " + N + " x " + D);
        double perplexity = parameterObject.getPerplexity();
        if(N - 1 < 3 * perplexity) { throw new IllegalArgumentException("Perplexity too large for the number of data points!\n"); }
        System.out.printf("Using no_dims = %d, perplexity = %f, and theta = %f\n", no_dims, perplexity, parameterObject.getTheta());

        // Set learning parameters
        double total_time = 0;
        int stop_lying_iter = earlyPhaseEnd(parameterObject.getMaxIter());
        int mom_switch_iter = stop_lying_iter;
        double momentum = .5, final_momentum = .8;
        double eta = 200.0;

        // Allocate some memory
        double [] dY    = new double[N * no_dims];
        double [] uY    = new double[N * no_dims];
        double [] gains = new double[N * no_dims];
        for(int i = 0; i < N * no_dims; i++) gains[i] = 1.0;

        // This is where the input would be normalized, and it never has been: the two statements that
        // did it - a zeroMean call and a division by the largest value - stood here commented out, and
        // all that ran was the scan producing the divisor, which nothing then read. The scan has gone
        // with them. Note the scan took no absolute value, so on input whose extreme is negative it
        // would not even have been a usable divisor.
        //
        // Leaving the normalization out does not change the embedding. A common factor on all
        // distances is absorbed by the perplexity search, which solves for a beta that scales
        // inversely. What it costs is numerical headroom on inputs of extreme magnitude.
        System.out.println("Computing input similarities...");
        long start = System.currentTimeMillis();

        int K  = (int) (3 * perplexity);
        int [] row_P = new int[N+1];
        int [] col_P = new int[N*K];
        double [] val_P = new double[N*K];

        // Compute asymmetric pairwise input similarities
        computeGaussianPerplexity(X, N, D, row_P, col_P, val_P, perplexity, K);

        // Symmetrize input similarities
        SymResult res = symmetrizeMatrix(row_P, col_P, val_P, N);
        row_P = res.sym_row_P;
        col_P = res.sym_col_P;
        val_P = res.sym_val_P;

        double sum_P = .0;
        for(int i = 0; i < row_P[N]; i++) sum_P += val_P[i];
        for(int i = 0; i < row_P[N]; i++) val_P[i] /= sum_P;

        long end = System.currentTimeMillis();

        // Lie about the P-values
        for(int i = 0; i < row_P[N]; i++) val_P[i] *= 12.0;

        // Perform main training loop
        System.out.printf("Done in %4.2f seconds (sparsity = %f)!\nLearning embedding...\n", (end - start) / 1000.0, (double) row_P[N] / ((double) N * (double) N));
        start = System.currentTimeMillis();

        TSneProgress.reset(TASK_GRADIENT_DESCENT, parameterObject.getMaxIter());
        for(int iter = 0; iter < parameterObject.getMaxIter() && !abort; iter++) {

            // Compute (approximate) gradient
            computeGradient(row_P, col_P, val_P, Y, N, no_dims, dY, parameterObject.getTheta(), iter);

            // Print out progress. This runs before the update so that the cost can reuse the
            // normalization the gradient step has just computed for exactly this embedding, instead
            // of building a second space partitioning tree. The reported value therefore describes Y
            // as it entered this iteration.
            if ( ((iter > 0 && iter % 50 == 0) || iter == parameterObject.getMaxIter() - 1) && !parameterObject.silent() ) {
                String err_string = "not_calculated";
                if(parameterObject.printError()) {
                    // approximate, see klDivergence
                    err_string = "" + klDivergence(row_P, col_P, val_P, Y, N, no_dims, lastSumQ);
                }
                TSneProgress.setMessage("Err: " + err_string);
            }

            updateGradient(N, no_dims, Y, momentum, eta, dY, uY, gains);

            // Make solution zero-mean
            zeroMean(Y, N, no_dims);

            // Stop lying about the P-values after a while, and switch momentum
            if(iter == stop_lying_iter) {
                for(int i = 0; i < row_P[N]; i++) val_P[i] /= 12.0;
            }
            if(iter == mom_switch_iter) momentum = final_momentum;

            TSneProgress.updateTo(iter + 1);
        }
        TSneProgress.finished();

        end = System.currentTimeMillis();
        total_time += (end - start) / 1000.0;

        System.out.printf("Fitting performed in %4.2f seconds.\n", total_time);
        System.out.flush();
        return expand(Y,N,no_dims);
    }

    void updateGradient(int N, int no_dims, double[] Y, double momentum, double eta, double[] dY, double[] uY,
            double[] gains) {
        for(int i = 0; i < N * no_dims; i++)  {
            // Update gains
            gains[i] = (sign_tsne(dY[i]) != sign_tsne(uY[i])) ? (gains[i] + .2) : (gains[i] * .8);
            if(gains[i] < .01) gains[i] = .01;

            // Perform gradient update (with momentum and gains)
            Y[i] = Y[i] + uY[i];
            uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
        }
    }

    // Compute gradient of the t-SNE cost function (using Barnes-Hut algorithm)
    void computeGradient(int[] inp_row_P,
            int[] inp_col_P, double[] inp_val_P, double[] Y, int N, int D,
            double[] dC, double theta, int iter)
    {
        // Construct space-partitioning tree on current map
        SPTree tree = new SPTree(D, Y, N);

        prepareWorkArrays(N, D);

        double totalSum_Q = 0.0;
        tree.computeEdgeForces(inp_row_P, inp_col_P, inp_val_P, N, pos_f);

        // Compute all terms required for t-SNE gradient
        for (int n = 0; n < N; n++)
            tree.computeNonEdgeForces(n, theta, neg_f[n], buff[n], sum_Q);
        totalSum_Q = DoubleStream.of(sum_Q).sum();
        lastSumQ = totalSum_Q;

        // Compute final t-SNE gradient
        for (int n = 0; n < N; n++)
        {
            for (int d = 0; d < D; d++)
            {
                dC[n * D + d] = pos_f[n * D + d] - (neg_f[n][d] / totalSum_Q);
            }
        }
    }

    /**
     * Kullback-Leibler divergence between the sparse input similarities and the current embedding,
     * which is the t-SNE cost function.
     * <p>
     * This builds no space partitioning tree of its own: it takes the normalization term that the
     * gradient step of the same iteration has already computed from exactly this embedding, so only
     * the sum over the edges of the sparse similarity matrix is left to do.
     *
     * @param row_P row offsets of the sparse input similarities
     * @param col_P column indices of the sparse input similarities
     * @param val_P values of the sparse input similarities
     * @param Y the embedding, flat, N times D
     * @param N the number of points
     * @param D the number of output dimensions
     * @param totalSum_Q normalization of the Q distribution for this embedding, see {@link #lastSumQ}
     * @return the cost
     */
    double klDivergence(int [] row_P, int [] col_P, double [] val_P, double [] Y, int N, int D, double totalSum_Q) {
        double C = .0;
        for (int n = 0; n < N; n++) {
            C += klDivergenceOfRow(n, row_P, col_P, val_P, Y, D, totalSum_Q);
        }
        return C;
    }

    /**
     * Contribution of one row of the sparse similarity matrix to {@link #klDivergence}.
     */
    final double klDivergenceOfRow(int n, int [] row_P, int [] col_P, double [] val_P, double [] Y, int D,
            double totalSum_Q) {
        double C = .0;
        final int ind1 = n * D;
        for (int i = row_P[n]; i < row_P[n + 1]; i++) {
            final int ind2 = col_P[i] * D;
            double Q = .0;
            for (int d = 0; d < D; d++) {
                double diff = Y[ind1 + d] - Y[ind2 + d];
                Q += diff * diff;
            }
            Q = (1.0 / (1.0 + Q)) / totalSum_Q;
            C += val_P[i] * log((val_P[i] + Double.MIN_VALUE) / (Q + Double.MIN_VALUE));
        }
        return C;
    }

    /**
     * The {@code N} rows of the flat {@code N x D} data matrix, as the points of the ball tree.
     * <p>
     * The rows are not copied. They used to be copied twice per point - once out of {@code X} by
     * {@code MatrixOps.extractRowFromFlatMatrix} and once more by the {@link DataPoint} constructor,
     * the first copy being garbage before the loop reached the next point. That was 17 MB of transient
     * allocation for 20 000 points of 50 dimensions and 63 MB for 5000 points of 784. Nothing writes
     * to a point's coordinates anywhere in the tree, so neither copy protected anything.
     *
     * @param X the data, row major, at least {@code N * D} elements
     * @param N the number of points
     * @param D the dimensionality
     * @return one point per row, in row order
     */
    static DataPoint [] rowViews(double [] X, int N, int D) {
        final DataPoint [] obj_X = new DataPoint [N];
        for(int n = 0; n < N; n++) {
            obj_X[n] = new DataPoint(X, n * D, D, n);
        }
        return obj_X;
    }

    // Compute input similarities with a fixed perplexity using ball trees
    void computeGaussianPerplexity(double [] X, int N, int D, int [] _row_P, int [] _col_P, double [] _val_P, double perplexity, int K) {
        if(perplexity > K) System.out.println("Perplexity should be lower than K!");

        // Allocate the memory we need
        int [] row_P = _row_P;
        int [] col_P = _col_P;
        double [] val_P = _val_P;
        double [] cur_P = new double[N - 1];

        row_P[0] = 0;
        for(int n = 0; n < N; n++) row_P[n + 1] = row_P[n] + K;    

        // Build ball tree on data set
        VpTree<DataPoint> tree = vpTreeSeed == null ? new VpTree<DataPoint>(distance)
                : new VpTree<DataPoint>(distance, vpTreeSeed.longValue());
        final DataPoint [] obj_X = rowViews(X, N, D);
        tree.create(obj_X);

        // Loop over all points to find nearest neighbors
        System.out.println("Building tree...");
        // reused across all points, the search overwrites them
        DataPoint [] indices = new DataPoint[K + 1];
        double [] distances = new double[K + 1];
        for(int n = 0; n < N; n++) {

            if(n % 10000 == 0) System.out.printf(" - point %d of %d\n", n, N);

            // Find nearest neighbors
            tree.search(obj_X[n], K + 1, indices, distances);

            // Initialize some variables for binary search
            boolean found = false;
            double beta = 1.0;
            double min_beta = -Double.MAX_VALUE;
            double max_beta =  Double.MAX_VALUE;
            double tol = 1e-5;

            // Iterate until we found a good perplexity
            int iter = 0; 
            double sum_P = 0.;
            while(!found && iter < 200) {

                // Compute Gaussian kernel row and entropy of current row
                sum_P = Double.MIN_VALUE;
                double H = .0;
                for(int m = 0; m < K; m++) {
                    cur_P[m] = exp(-beta * distances[m + 1]);
                    sum_P += cur_P[m];
                    H += beta * (distances[m + 1] * cur_P[m]);
                }
                H = (H / sum_P) + log(sum_P);

                // Evaluate whether the entropy is within the tolerance level
                double Hdiff = H - log(perplexity);
                if(Hdiff < tol && -Hdiff < tol) {
                    found = true;
                }
                else {
                    if(Hdiff > 0) {
                        min_beta = beta;
                        if(max_beta == Double.MAX_VALUE || max_beta == -Double.MAX_VALUE)
                            beta *= 2.0;
                        else
                            beta = (beta + max_beta) / 2.0;
                    }
                    else {
                        max_beta = beta;
                        if(min_beta == -Double.MAX_VALUE || min_beta == Double.MAX_VALUE)
                            beta /= 2.0;
                        else
                            beta = (beta + min_beta) / 2.0;
                    }
                }

                // Update iteration counter
                iter++;
            }

            // Row-normalize current row of P and store in matrix 
            for(int m = 0; m < K; m++) {
                cur_P[m] /= sum_P;
                col_P[row_P[n] + m] = indices[m + 1].index();
                val_P[row_P[n] + m] = cur_P[m];
            }
        }
    }

    class SymResult {
        int []    sym_row_P;
        int []    sym_col_P;
        double [] sym_val_P;

        public SymResult(int[] sym_row_P, int[] sym_col_P, double[] sym_val_P) {
            super();
            this.sym_row_P = sym_row_P;
            this.sym_col_P = sym_col_P;
            this.sym_val_P = sym_val_P;
        }
    }

    SymResult symmetrizeMatrix(int [] _row_P, int [] _col_P, double [] _val_P, int N) {

        // Get sparse matrix
        int [] row_P = _row_P;
        int [] col_P = _col_P;
        double [] val_P = _val_P;

        // Count number of elements and row counts of symmetric matrix
        int [] row_counts = new int[N];
        for(int n = 0; n < N; n++) {
            for(int i = row_P[n]; i < row_P[n + 1]; i++) {

                // Check whether element (col_P[i], n) is present
                boolean present = false;
                for(int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
                    if(col_P[m] == n) {
                        present = true;
                        break;
                    }
                }
                if(present) row_counts[n]++;
                else {
                    row_counts[n]++;
                    row_counts[col_P[i]]++;
                }
            }
        }
        int no_elem = 0;
        for(int n = 0; n < N; n++) no_elem += row_counts[n];

        // Allocate memory for symmetrized matrix
        int []    sym_row_P = new int[N + 1];
        int []    sym_col_P = new int[no_elem];
        double [] sym_val_P = new double[no_elem];

        // Construct new row indices for symmetric matrix
        sym_row_P[0] = 0;
        for(int n = 0; n < N; n++) sym_row_P[n + 1] = sym_row_P[n] + row_counts[n];

        // Fill the result matrix
        int [] offset = new int[N];
        for(int n = 0; n < N; n++) {
            for(int i = row_P[n]; i < row_P[n + 1]; i++) {                                  // considering element(n, col_P[i])

                // Check whether element (col_P[i], n) is present
                boolean present = false;
                for(int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
                    if(col_P[m] == n) {
                        present = true;
                        if(n <= col_P[i]) {                                                 // make sure we do not add elements twice
                            sym_col_P[sym_row_P[n]        + offset[n]]        = col_P[i];
                            sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
                            sym_val_P[sym_row_P[n]        + offset[n]]        = val_P[i] + val_P[m];
                            sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i] + val_P[m];
                        }
                    }
                }

                // If (col_P[i], n) is not present, there is no addition involved
                if(!present) {
                    sym_col_P[sym_row_P[n]        + offset[n]]        = col_P[i];
                    sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
                    sym_val_P[sym_row_P[n]        + offset[n]]        = val_P[i];
                    sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i];
                }

                // Update offsets
                if(!present || (present && n <= col_P[i])) {
                    offset[n]++;
                    if(col_P[i] != n) offset[col_P[i]]++;               
                }
            }
        }

        // Divide the result by two
        for(int i = 0; i < no_elem; i++) sym_val_P[i] /= 2.0;

        return new SymResult(sym_row_P, sym_col_P, sym_val_P);
    }

    // Makes data zero-mean
    void zeroMean(double [] X, int N, int D) {

        // Compute data mean
        double [] mean = new double[D];
        for(int n = 0; n < N; n++) {
            for(int d = 0; d < D; d++) {
                mean[d] += X[n * D + d];
            }
        }
        for(int d = 0; d < D; d++) {
            mean[d] /= (double) N;
        }

        // Subtract data mean
        for(int n = 0; n < N; n++) {
            for(int d = 0; d < D; d++) {
                X[n * D + d] -= mean[d];
            }
        }
    }

    // Makes data zero-mean
    void zeroMean(double [][] X, int N, int D) {

        // Compute data mean
        double [] mean = new double[D];
        for(int n = 0; n < N; n++) {
            for(int d = 0; d < D; d++) {
                mean[d] += X[n][d];
            }
        }
        for(int d = 0; d < D; d++) {
            mean[d] /= (double) N;
        }

        // Subtract data mean
        for(int n = 0; n < N; n++) {
            for(int d = 0; d < D; d++) {
                X[n][d] -= mean[d];
            }
        }
    }

    @Override
    public void abort() {
        abort = true;
    }
}

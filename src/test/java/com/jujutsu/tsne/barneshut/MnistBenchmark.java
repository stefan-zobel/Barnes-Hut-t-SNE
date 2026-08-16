package com.jujutsu.tsne.barneshut;

import com.jujutsu.tsne.TSneConfiguration;
import com.jujutsu.utils.MnistData;
import com.jujutsu.utils.TSneUtils;

/**
 * One full t-SNE run over MNIST - the measurement `Performance.md` reports for the 60 000 image set.
 * <p>
 * Not a test. It has no assertions and takes minutes; it exists so that the numbers in the
 * documentation can be reproduced and so that a change can be weighed against them. See the README
 * for how to build and run it.
 * <p>
 * The configuration is the one the {@code TSneTest} demo uses, scaled up: two output dimensions, the
 * input reduced to 55 by PCA, perplexity 20. The iteration count matters for comparability - at 1000
 * the early exaggeration ends at iteration 250, which is what the reference implementation does, so
 * runs of 1000 or more are comparable to it and to each other.
 */
public final class MnistBenchmark {

    private static final int DEFAULT_POINTS = 60000;
    private static final int DEFAULT_ITERATIONS = 1000;

    private MnistBenchmark() {
    }

    /**
     * @param args optional: number of images, then number of iterations
     */
    public static void main(String[] args) {
        int n = args.length > 0 ? Integer.parseInt(args[0]) : DEFAULT_POINTS;
        int iterations = args.length > 1 ? Integer.parseInt(args[1]) : DEFAULT_ITERATIONS;

        long l0 = System.nanoTime();
        double[][] X = MnistData.load(n, MnistData.Encoding.LEGACY);
        double load = (System.nanoTime() - l0) / 1e9;

        TSneConfiguration config = TSneUtils.buildConfig(X, 2, 55, 20.0, iterations);
        long t0 = System.nanoTime();
        double[][] Y = new ParallelBHTsne().tsne(config);
        double total = (System.nanoTime() - t0) / 1e9;

        double checksum = 0.0;
        for (double[] row : Y) {
            for (double v : row) {
                checksum += v * v;
            }
        }

        System.out.printf("%n--- MnistBenchmark: n = %d, %d iterations ---%n", n, iterations);
        System.out.printf("reading the data        : %8.2f s%n", load);
        System.out.printf("tsne()                  : %8.2f s   <- the number to compare%n", total);
        System.out.printf("embedding               : %d x %d, sum of squares %.6e%n",
                Y.length, Y[0].length, checksum);
    }
}

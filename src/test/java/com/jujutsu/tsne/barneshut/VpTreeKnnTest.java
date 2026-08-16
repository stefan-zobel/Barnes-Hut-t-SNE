package com.jujutsu.tsne.barneshut;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.junit.Test;

import com.jujutsu.utils.MatrixOps;

/**
 * Verifies the k nearest neighbour search of the vantage point tree against a brute force reference.
 * <p>
 * The tree picks its vantage points with {@link java.util.concurrent.ThreadLocalRandom}, so its shape
 * differs from run to run and the result cannot be compared against a recorded golden output. What
 * has to hold regardless of the shape is that the search returns exactly the neighbours a brute force
 * scan finds, which is what these tests check.
 * <p>
 * Every check runs twice, once over points that own their row and once over points that are views of
 * one flat matrix, which is how {@link BHTSne#rowViews} builds them. Both sources produce the same
 * numbers, so both runs have to produce the same neighbours.
 */
public class VpTreeKnnTest {

    private static final int N = 400;
    private static final int D = 12;
    private static final int K = 20;
    private static final double TOLERANCE = 1e-12;

    private static double[] randomFlatMatrix() {
        final Random random = new Random(20260815L);
        final double[] x = new double[N * D];
        for (int n = 0; n < N; n++) {
            for (int d = 0; d < D; d++) {
                x[n * D + d] = random.nextGaussian() + (n % 5) * 2.5;
            }
        }
        return x;
    }

    /** points that own their row, as callers outside the ball tree build them */
    private static DataPoint[] owningPoints() {
        final double[] x = randomFlatMatrix();
        final DataPoint[] points = new DataPoint[N];
        for (int n = 0; n < N; n++) {
            points[n] = new DataPoint(D, n, MatrixOps.extractRowFromFlatMatrix(x, n, D));
        }
        return points;
    }

    /** the same points as views of one flat matrix, as the perplexity phase builds them */
    private static DataPoint[] viewPoints() {
        return BHTSne.rowViews(randomFlatMatrix(), N, D);
    }

    /** Reference: the K + 1 nearest neighbours of {@code target}, the point itself first. */
    private static int[] bruteForce(DataPoint[] points, int target, double[] outDistances) {
        final EuclideanDistance distance = new EuclideanDistance();
        final Integer[] order = new Integer[points.length];
        final double[] all = new double[points.length];
        for (int n = 0; n < points.length; n++) {
            order[n] = n;
            all[n] = distance.distance(points[target], points[n]);
        }
        Arrays.sort(order, (a, b) -> all[a] != all[b] ? Double.compare(all[a], all[b]) : Integer.compare(a, b));
        final int[] result = new int[K + 1];
        for (int m = 0; m <= K; m++) {
            result[m] = order[m];
            outDistances[m] = all[order[m]];
        }
        return result;
    }

    @Test
    public void searchFindsTheSameNeighboursAsABruteForceScan() {
        assertSearchMatchesBruteForce(owningPoints());
    }

    @Test
    public void searchOverViewsOfAFlatMatrixFindsTheSameNeighbours() {
        assertSearchMatchesBruteForce(viewPoints());
    }

    @Test
    public void searchMultipleFindsTheSameNeighboursAsABruteForceScan() {
        assertSearchMultipleMatchesBruteForce(owningPoints());
    }

    @Test
    public void searchMultipleOverViewsOfAFlatMatrixFindsTheSameNeighbours() {
        assertSearchMultipleMatchesBruteForce(viewPoints());
    }

    private static void assertSearchMatchesBruteForce(final DataPoint[] points) {
        final VpTree<DataPoint> tree = new VpTree<>(new EuclideanDistance());
        tree.create(points);

        final List<DataPoint> neighbours = new ArrayList<>();
        final List<Double> distances = new ArrayList<>();
        final double[] expectedDistances = new double[K + 1];

        for (int n = 0; n < N; n++) {
            tree.search(points[n], K + 1, neighbours, distances);
            final int[] expected = bruteForce(points, n, expectedDistances);

            assertEquals("number of neighbours for point " + n, K + 1, neighbours.size());
            assertEquals("number of distances for point " + n, K + 1, distances.size());
            // the perplexity computation relies on element 0 being the query point itself
            assertEquals("first neighbour of point " + n, n, neighbours.get(0).index());
            assertEquals("distance to itself for point " + n, 0.0, distances.get(0), TOLERANCE);

            for (int m = 0; m <= K; m++) {
                assertEquals("neighbour " + m + " of point " + n, expected[m], neighbours.get(m).index());
                assertEquals("distance " + m + " of point " + n, expectedDistances[m], distances.get(m), TOLERANCE);
            }
        }
    }

    private static void assertSearchMultipleMatchesBruteForce(final DataPoint[] points) {
        final ParallelVpTree<DataPoint> tree = new ParallelVpTree<>(new EuclideanDistance());
        tree.create(points);

        final List<TreeSearchResult> results = tree.searchMultiple(tree, points, K + 1);
        assertEquals("one result per point", N, results.size());

        // the results are collected concurrently, so they arrive in an arbitrary order
        final TreeSearchResult[] byPoint = new TreeSearchResult[N];
        for (TreeSearchResult result : results) {
            byPoint[result.getIndex()] = result;
        }

        final double[] expectedDistances = new double[K + 1];
        for (int n = 0; n < N; n++) {
            final TreeSearchResult result = byPoint[n];
            assertNotNull("missing result for point " + n, result);
            final List<DataPoint> neighbours = result.getIndices();
            final List<Double> distances = result.getDistances();
            final int[] expected = bruteForce(points, n, expectedDistances);

            assertEquals("number of neighbours for point " + n, K + 1, neighbours.size());
            assertEquals("first neighbour of point " + n, n, neighbours.get(0).index());

            for (int m = 0; m <= K; m++) {
                assertEquals("neighbour " + m + " of point " + n, expected[m], neighbours.get(m).index());
                assertEquals("distance " + m + " of point " + n, expectedDistances[m], distances.get(m), TOLERANCE);
            }
        }
    }
}

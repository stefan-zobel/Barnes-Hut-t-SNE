package com.jujutsu.tsne.barneshut;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;

import java.util.Random;

import org.junit.Test;

/**
 * The vantage point tree picks the point it partitions around at random, so it builds a different
 * tree every run. That is harmless for the answer - the nearest neighbours of a point are the nearest
 * neighbours whatever the tree looks like - but not for the cost of finding them: measured at
 * {@code N = 60 000}, the same build over the same input took between 46.8 s and 93.5 s depending on
 * how the vantage points fell. Nothing can be measured in that phase until the choice is pinned down.
 * <p>
 * So a seeded tree has to build reproducibly, and it must not change what the search returns. Both are
 * checked here, along with the property that makes it usable more than once: {@code create} resets the
 * generator instead of carrying it on, so the second build over the same points is the first one
 * again.
 */
public class VpTreeSeedTest {

    private static final int N = 300;
    private static final int D = 6;
    private static final int K = 12;

    private static DataPoint[] points(long seed) {
        Random random = new Random(seed);
        double[] x = new double[N * D];
        for (int i = 0; i < x.length; i++) {
            x[i] = random.nextGaussian();
        }
        return BHTSne.rowPoints(x, N, D);
    }

    /** A description of the tree's shape: every node's item index and threshold, in build order. */
    private static String shapeOf(VpTree<DataPoint> tree) {
        StringBuilder out = new StringBuilder();
        describe(tree.getRoot(), out);
        return out.toString();
    }

    private static void describe(VpTree<DataPoint>.Node node, StringBuilder out) {
        if (node == null) {
            out.append('.');
            return;
        }
        out.append('(').append(node.index).append(':').append(Double.toHexString(node.threshold));
        describe(node.getLeft(), out);
        describe(node.getRight(), out);
        out.append(')');
    }

    @Test
    public void anUnseededTreeSaysSoAndASeededOneSaysSo() {
        assertFalse(new VpTree<DataPoint>(new EuclideanDistance()).isSeeded());
        assertFalse(new VpTree<DataPoint>().isSeeded());
        assertTrue(new VpTree<DataPoint>(new EuclideanDistance(), 42L).isSeeded());
        assertTrue(new ParallelVpTree<DataPoint>(new EuclideanDistance(), 42L).isSeeded());
    }

    @Test
    public void twoTreesWithTheSameSeedHaveTheSameShape() {
        VpTree<DataPoint> first = new VpTree<DataPoint>(new EuclideanDistance(), 4711L);
        VpTree<DataPoint> second = new VpTree<DataPoint>(new EuclideanDistance(), 4711L);
        first.create(points(17L));
        second.create(points(17L));

        assertEquals(shapeOf(first), shapeOf(second));
    }

    @Test
    public void rebuildingWithTheSameSeedGivesTheSameTreeAgain() {
        // create() resets the generator. Without that only the first build would be reproducible,
        // which is the trap a benchmark that reuses one tree instance would fall into.
        VpTree<DataPoint> tree = new VpTree<DataPoint>(new EuclideanDistance(), 4711L);
        tree.create(points(17L));
        String first = shapeOf(tree);

        tree.create(points(17L));

        assertEquals(first, shapeOf(tree));
    }

    @Test
    public void differentSeedsBuildDifferentTrees() {
        // otherwise the seed could be quietly ignored and every test above would still pass
        VpTree<DataPoint> first = new VpTree<DataPoint>(new EuclideanDistance(), 1L);
        VpTree<DataPoint> second = new VpTree<DataPoint>(new EuclideanDistance(), 2L);
        first.create(points(17L));
        second.create(points(17L));

        assertNotEquals(shapeOf(first), shapeOf(second));
    }

    @Test
    public void theSeedDoesNotChangeWhichNeighboursAreFound() {
        DataPoint[] a = points(17L);
        DataPoint[] b = points(17L);
        DataPoint[] c = points(17L);
        VpTree<DataPoint> seeded = new VpTree<DataPoint>(new EuclideanDistance(), 1L);
        VpTree<DataPoint> otherSeed = new VpTree<DataPoint>(new EuclideanDistance(), 987654L);
        VpTree<DataPoint> unseeded = new VpTree<DataPoint>(new EuclideanDistance());
        seeded.create(a);
        otherSeed.create(b);
        unseeded.create(c);

        DataPoint[] n1 = new DataPoint[K], n2 = new DataPoint[K], n3 = new DataPoint[K];
        double[] d1 = new double[K], d2 = new double[K], d3 = new double[K];

        for (int i = 0; i < N; i++) {
            seeded.search(a[i], K, n1, d1);
            otherSeed.search(b[i], K, n2, d2);
            unseeded.search(c[i], K, n3, d3);
            for (int m = 0; m < K; m++) {
                assertEquals("neighbour " + m + " of point " + i, n1[m].index(), n2[m].index());
                assertEquals("neighbour " + m + " of point " + i, n1[m].index(), n3[m].index());
                assertEquals("distance " + m + " of point " + i, d1[m], d2[m], 0.0);
                assertEquals("distance " + m + " of point " + i, d1[m], d3[m], 0.0);
            }
        }
    }

    @Test
    public void aSeededRunProducesTheSameEmbeddingAsAnUnseededOne() {
        // the seed is a handle for measurement; it must not move the result
        double[][] x = new double[200][8];
        Random random = new Random(99L);
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) x[i][j] = random.nextGaussian();
        }

        ParallelBHTsne seeded = new ParallelBHTsne();
        seeded.vpTreeSeed = Long.valueOf(12345L);
        double[][] withSeed = seeded.tsne(com.jujutsu.utils.TSneUtils.buildConfig(
                x, 2, x[0].length, 10.0, 40, false, 0.5, true, false));
        double[][] withoutSeed = new ParallelBHTsne().tsne(com.jujutsu.utils.TSneUtils.buildConfig(
                x, 2, x[0].length, 10.0, 40, false, 0.5, true, false));

        for (int i = 0; i < withSeed.length; i++) {
            org.junit.Assert.assertArrayEquals("row " + i, withoutSeed[i], withSeed[i], 0.0);
        }
    }
}

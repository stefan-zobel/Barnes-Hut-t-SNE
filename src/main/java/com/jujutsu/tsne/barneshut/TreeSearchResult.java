package com.jujutsu.tsne.barneshut;

import java.util.AbstractList;
import java.util.Arrays;
import java.util.List;

/**
 * The k nearest neighbours of one search target, nearest first.
 * <p>
 * Neighbours and distances are held in plain arrays. Storing the distances as a {@code List<Double>}
 * used to box every one of them, roughly 1.8 million objects for 20 000 points at perplexity 30, and
 * the perplexity binary search unboxed them again up to 200 times per point.
 */
public class TreeSearchResult {

    private final int n;
    private final DataPoint[] neighbors;
    private final double[] distances;

    /**
     * @param neighbors the neighbours, nearest first, element 0 being the target itself
     * @param distances their distances to the target, in the same order
     * @param n index of the search target
     */
    public TreeSearchResult(DataPoint[] neighbors, double[] distances, int n) {
        this.neighbors = neighbors;
        this.distances = distances;
        this.n = n;
    }

    /**
     * The signature this class was constructed with before the neighbours and their distances moved
     * into arrays. It is kept so that the change stays binary and source compatible, and it delegates:
     * both lists are read out into the arrays this instance stores.
     * <p>
     * Which is the one thing it cannot reproduce exactly. The old constructor kept the caller's list
     * objects, so a later change to either of them showed through {@link #getIndices()} and
     * {@link #getDistances()}. This one copies, so it does not.
     *
     * @param neighbors the neighbours, nearest first, element 0 being the target itself
     * @param distances their distances to the target, in the same order
     * @param n index of the search target
     * @deprecated use {@link #TreeSearchResult(DataPoint[], double[], int)}, which neither boxes the
     *             distances nor copies anything
     */
    @Deprecated
    public TreeSearchResult(List<DataPoint> neighbors, List<Double> distances, int n) {
        this(neighbors.toArray(new DataPoint[neighbors.size()]), unbox(distances), n);
    }

    private static double[] unbox(List<Double> values) {
        double[] unboxed = new double[values.size()];
        int i = 0;
        for (Double value : values) {
            unboxed[i++] = value.doubleValue();
        }
        return unboxed;
    }

    /**
     * @return the neighbours, nearest first; not copied
     */
    public DataPoint[] getNeighbors() {
        return neighbors;
    }

    /**
     * @return the distances of the neighbours, in the same order; not copied
     */
    public double[] getNeighborDistances() {
        return distances;
    }

    /**
     * @return index of the search target
     */
    public int getIndex() {
        return n;
    }

    /**
     * @return the neighbours as a list
     * @deprecated use {@link #getNeighbors()}, which does not wrap the array
     */
    @Deprecated
    public List<DataPoint> getIndices() {
        return Arrays.asList(neighbors);
    }

    /**
     * @return the distances as a list of boxed values
     * @deprecated use {@link #getNeighborDistances()}, which does not box
     */
    @Deprecated
    public List<Double> getDistances() {
        return new AbstractList<Double>() {
            @Override
            public Double get(int index) {
                return Double.valueOf(distances[index]);
            }

            @Override
            public int size() {
                return distances.length;
            }
        };
    }
}

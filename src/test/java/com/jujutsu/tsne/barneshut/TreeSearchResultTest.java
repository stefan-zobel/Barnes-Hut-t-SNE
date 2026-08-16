package com.jujutsu.tsne.barneshut;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.junit.Test;

/**
 * {@link TreeSearchResult} stores the neighbours and their distances in arrays since F4, but keeps the
 * constructor that took two lists so that the change stays binary and source compatible. A
 * compatibility shim nobody exercises is a shim that quietly rots, or gets deleted by the next pass
 * over the dead code, so it is pinned here.
 */
@SuppressWarnings("deprecation")
public class TreeSearchResultTest {

    private static DataPoint point(int index) {
        return new DataPoint(1, index, new double[] { index });
    }

    @Test
    public void theListConstructorProducesTheSameResultAsTheArrayOne() {
        DataPoint[] neighbors = { point(7), point(3), point(11) };
        double[] distances = { 0.0, 1.5, 2.25 };

        TreeSearchResult viaArrays = new TreeSearchResult(neighbors, distances, 7);
        TreeSearchResult viaLists = new TreeSearchResult(Arrays.asList(neighbors),
                Arrays.asList(0.0, 1.5, 2.25), 7);

        assertEquals("target index", viaArrays.getIndex(), viaLists.getIndex());
        assertArrayEquals("neighbours", viaArrays.getNeighbors(), viaLists.getNeighbors());
        assertArrayEquals("distances", viaArrays.getNeighborDistances(),
                viaLists.getNeighborDistances(), 0.0);
    }

    @Test
    public void theListConstructorKeepsTheNeighbourInstances() {
        // the elements are the caller's, only the list holding them is read out
        DataPoint first = point(7);
        DataPoint second = point(3);

        TreeSearchResult result = new TreeSearchResult(Arrays.asList(first, second),
                Arrays.asList(0.0, 1.5), 7);

        assertSame("neighbour 0", first, result.getNeighbors()[0]);
        assertSame("neighbour 1", second, result.getNeighbors()[1]);
    }

    @Test
    public void theListConstructorReadsTheListsOutRatherThanKeepingThem() {
        // the one thing the delegation cannot reproduce: the old constructor stored the caller's
        // lists, so a later change to them showed through. This documents that it no longer does.
        List<DataPoint> neighbors = new ArrayList<DataPoint>(Arrays.asList(point(7), point(3)));
        List<Double> distances = new ArrayList<Double>(Arrays.asList(0.0, 1.5));

        TreeSearchResult result = new TreeSearchResult(neighbors, distances, 7);
        neighbors.set(1, point(99));
        distances.set(1, 42.0);

        assertEquals("neighbour 1 is the one that was passed in", 3, result.getNeighbors()[1].index());
        assertEquals("distance 1 is the one that was passed in", 1.5, result.getNeighborDistances()[1],
                0.0);
    }

    @Test
    public void theDeprecatedListViewsAgreeWithTheArrays() {
        TreeSearchResult result = new TreeSearchResult(new DataPoint[] { point(7), point(3) },
                new double[] { 0.0, 1.5 }, 7);

        assertEquals("size of the neighbour view", 2, result.getIndices().size());
        assertEquals("size of the distance view", 2, result.getDistances().size());
        assertSame("neighbour 0", result.getNeighbors()[0], result.getIndices().get(0));
        assertEquals("distance 1", 1.5, result.getDistances().get(1).doubleValue(), 0.0);
    }
}

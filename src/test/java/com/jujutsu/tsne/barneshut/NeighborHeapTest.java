package com.jujutsu.tsne.barneshut;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.Random;

import org.junit.Test;

/**
 * Tests for the bounded max-heap that collects the nearest neighbours during a search.
 */
public class NeighborHeapTest {

	private static DataPoint[] points(int n) {
		final DataPoint[] items = new DataPoint[n];
		for (int i = 0; i < n; i++) {
			items[i] = new DataPoint(1, i, new double[] {i});
		}
		return items;
	}

	@Test
	public void keepsTheNearestNeighboursAndReturnsThemAscending() {
		final int candidates = 200;
		final int k = 10;
		final Random random = new Random(4711);
		final DataPoint[] items = points(candidates);
		final NeighborHeap heap = new NeighborHeap(k);

		// distinct distances in random order
		final int[] order = new int[candidates];
		for (int i = 0; i < candidates; i++) {
			order[i] = i;
		}
		for (int i = candidates - 1; i > 0; i--) {
			int j = random.nextInt(i + 1);
			int tmp = order[i];
			order[i] = order[j];
			order[j] = tmp;
		}
		for (int i = 0; i < candidates; i++) {
			heap.add(order[i], order[i] * 0.5);
		}

		assertTrue(heap.isFull());
		assertEquals(k, heap.size());

		final DataPoint[] neighbors = new DataPoint[k];
		final double[] distances = new double[k];
		heap.drainAscending(items, neighbors, distances);

		assertEquals(0, heap.size());
		for (int i = 0; i < k; i++) {
			assertEquals("neighbour " + i, i, neighbors[i].index());
			assertEquals("distance " + i, i * 0.5, distances[i], 1e-12);
		}
	}

	@Test
	public void reportsTheFarthestCollectedNeighbour() {
		final NeighborHeap heap = new NeighborHeap(3);
		heap.add(0, 5.0);
		assertEquals(5.0, heap.maxDist(), 1e-12);
		heap.add(1, 2.0);
		assertEquals(5.0, heap.maxDist(), 1e-12);
		heap.add(2, 9.0);
		assertEquals(9.0, heap.maxDist(), 1e-12);
		assertTrue(heap.isFull());
		// the heap is full, so this drops the farthest one
		heap.add(3, 1.0);
		assertEquals(5.0, heap.maxDist(), 1e-12);
		assertEquals(3, heap.size());
	}

	@Test
	public void handlesFewerCandidatesThanCapacity() {
		final DataPoint[] items = points(4);
		final NeighborHeap heap = new NeighborHeap(10);
		heap.add(2, 3.0);
		heap.add(0, 1.0);
		heap.add(3, 2.0);

		assertFalse(heap.isFull());
		assertEquals(3, heap.size());

		final DataPoint[] neighbors = new DataPoint[10];
		final double[] distances = new double[10];
		heap.drainAscending(items, neighbors, distances);

		assertEquals(0, neighbors[0].index());
		assertEquals(3, neighbors[1].index());
		assertEquals(2, neighbors[2].index());
		assertArrayEqualsPrefix(new double[] {1.0, 2.0, 3.0}, distances);
	}

	@Test
	public void clearMakesTheHeapReusable() {
		final NeighborHeap heap = new NeighborHeap(2);
		heap.add(0, 1.0);
		heap.add(1, 2.0);
		heap.clear();
		assertEquals(0, heap.size());
		heap.add(5, 7.0);
		assertEquals(1, heap.size());
		assertEquals(7.0, heap.maxDist(), 1e-12);
	}

	private static void assertArrayEqualsPrefix(double[] expected, double[] actual) {
		assertEquals(Arrays.toString(expected), Arrays.toString(Arrays.copyOf(actual, expected.length)));
	}
}

package com.jujutsu.tsne.barneshut;

import static org.junit.Assert.*;

import java.util.Arrays;
import java.util.Collections;
import java.util.PriorityQueue;

import org.junit.Test;

// The ordering of the intermediate search results is covered by NeighborHeapTest, which replaced the
// PriorityQueue<HeapItem> this class used to exercise.
public class BHUtilsTest {

	@Test
	public void testDistanceComparator() {
		DataPoint [] _items = new DataPoint [3];
		double [] p1 = {1.0,2.0,3.0};
		_items[0] = new DataPoint(3,0,p1);
		double [] p2 = {4.0,5.0,6.0};
		_items[1] = new DataPoint(3,1,p2);
		double [] p3 = {10.0,11.0,12.0};
		_items[2] = new DataPoint(3,2,p3);
		DataPoint [] expectedRes = { _items[1], _items[0], _items[2] };
		Arrays.sort(_items, new DistanceComparator(_items[1]));
		assertArrayEquals(expectedRes, _items);
	}
	
	@Test
	public void testEuclidDistance() {
		double [] p1 = {1.0,2.0,3.0};
		DataPoint d1 = new DataPoint(3,0,p1);
		double [] p2 = {4.0,5.0,6.0};
		DataPoint d2 = new DataPoint(3,1,p2);
		assertEquals(5.196152,d1.euclidean_distance(d2),0.000001);
	}
	
	@Test
	public void testNthElementIntMiddle() {
		int [] array = {5, 6, 4, 3, 2, 6, 7, 9, 3};
		int pivot = 4;
		VpTree.nth_element(array, 0, pivot, 9);
		assertEquals(5,array[pivot]);
	}

	@Test
	public void testNthElementInt1() {
		int [] array = {5, 6, 4, 3, 2, 6, 7, 9, 3};
		int pivot = 1;
		VpTree.nth_element(array, 0, pivot, 7);
		assertEquals(3,array[pivot]);
	}
	
	@Test
	public void testNthElement1() {
		int [] array = {5, 6, 4, 3, 2, 6, 7, 9, 3};
		int pivot = 1;
		VpTree.nth_element(array, 0, pivot, 2);
		assertEquals(6,array[pivot]);
	}
	
	@Test
	public void testNthElementFull() {
		int [] array = {5, 6, 4, 3, 2, 6, 7, 9, 3};
		int pivot = 4;
		VpTree.nth_element(array, 0, pivot, 9);
		assertEquals(5,array[pivot]);
	}
	
	@Test
	public void testNthElementInt2() {
		int [] array = {5, 6, 4, 3, 2, 6, 7, 9, 3};
		int pivot = 2;
		VpTree.nth_element(array, 0, pivot, 3);
		assertEquals(6,array[pivot]);
	}

	@Test
	public void testNthElementInt3() {
		int [] array = {5, 6, 4, 3, 2, 6, 7, 9, 3};
		int pivot = 6;
		VpTree.nth_element(array, 0, pivot, 8);
		assertEquals(7,array[pivot]);
	}
	
	@Test
	public void testNthElementInt4() {
		int [] array = {5, 6, 4, 3, 2, 6, 7, 9, 3};
		int pivot = 6;
		VpTree.nth_element(array, 5, pivot, 8);
		assertEquals(7,array[pivot]);
	}

	@Test
	public void testPrioHeap() {
		int k = 10;
		PriorityQueue<Integer> heap = new PriorityQueue<Integer>(k,Collections.reverseOrder());
		int [] array = {5, 6, 4, 3, 2, 6, 7, 9, 3};
		
		for (int i = 0; i < array.length; i++) {
			heap.add(array[i]);			
		}

		int cnt = 0;		
		int [] result = new int[array.length]; 
		while(!heap.isEmpty()) {
			result[cnt++] = heap.remove();
		}
		int [] expected = {9, 7, 6, 6, 5, 4, 3, 3, 2};
		assertArrayEquals(expected, result);
	}
}

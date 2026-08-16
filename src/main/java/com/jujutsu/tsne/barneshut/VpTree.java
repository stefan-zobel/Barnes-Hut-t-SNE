package com.jujutsu.tsne.barneshut;

import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

public class VpTree<StorageType> {

	DataPoint [] _items;
	Node _root;
	Distance distance;

	// Distance of every item to the vantage point of the node currently being built. The tree is
	// built depth first over disjoint sub ranges, so one array for the whole tree is enough.
	private double [] _distances;

	public VpTree() {
		distance = new EuclideanDistance();
	}

	public VpTree(Distance distance) {
		this.distance = distance;
	}

	/**
	 * Builds the tree over the given points. The points themselves are not copied, only the array
	 * holding them: building permutes it, and the callers keep using their own array to look up the
	 * search target of point {@code n}. That is {@code N} references, not {@code N} rows - the points
	 * are views of one flat matrix, see {@link DataPoint}.
	 *
	 * @param items the points, one per index of the data set
	 */
	public void create(DataPoint [] items) {
		_items = items.clone();
		_distances = new double[_items.length];
		_root = buildFromPoints(0,items.length);
	}

	public void search(DataPoint target, int k, List<DataPoint> results, List<Double> distances) {
		DataPoint [] neighbors = new DataPoint[k];
		double [] neighborDistances = new double[k];
		int found = search(target, k, neighbors, neighborDistances);

		results.clear();
		distances.clear();
		for (int i = 0; i < found; i++) {
			results.add(neighbors[i]);
			distances.add(neighborDistances[i]);
		}
	}

	/**
	 * Searches the k nearest neighbours of {@code target}, nearest first.
	 *
	 * @param target the point to search neighbours for
	 * @param k the number of neighbours to collect
	 * @param outNeighbors receives the neighbours, must hold at least {@code k} elements
	 * @param outDistances receives their distances, must hold at least {@code k} elements
	 * @return the number of neighbours found, which is {@code k} unless the tree holds fewer points
	 */
	public int search(DataPoint target, int k, DataPoint [] outNeighbors, double [] outDistances) {
		NeighborHeap heap = new NeighborHeap(k);

		// Variable that tracks the distance to the farthest point in our results
		double tau = Double.MAX_VALUE;

		// Perform the search
		_root.search(_root, target, k, heap, tau);

		// Gather final results, the heap yields them nearest first
		int found = heap.size();
		heap.drainAscending(_items, outNeighbors, outDistances);
		return found;
	}

	// Function that (recursively) fills the tree
	public Node buildFromPoints( int lower, int upper )
	{
		if (upper == lower) {     // indicates that we're done here!
			return null;
		}

		// Lower index is center of current node
		Node node = createNode();
		node.index = lower;

		if (upper - lower > 1) {      // if we did not arrive at leaf yet
			if (_distances == null || _distances.length < _items.length) {
				_distances = new double[_items.length];
			}

			// Choose an arbitrary point and move it to the start
			int i = (int) (ThreadLocalRandom.current().nextDouble() * (upper - lower - 1)) + lower;
			if(lower != i) swap(_items, lower, i);

			// Distance to the vantage point, evaluated once per item rather than once per comparison
			DataPoint vantagePoint = _items[lower];
			for (int j = lower + 1; j < upper; j++) {
				_distances[j] = distance(vantagePoint, _items[j]);
			}

			// Partition around the median distance
			int median = (upper + lower) / 2;
			select(lower + 1, median, upper);

			// Threshold of the new node will be the distance to the median, which the selection has
			// already computed
			node.threshold = _distances[median];

			// Recursively build tree
			node.index = lower;
			node.left = buildFromPoints(lower + 1, median);
			node.right = buildFromPoints(median, upper);
		}

		// Return result
		return node;
	}

	void print_sorted_items(DataPoint [] items, Distance distance, DataPoint reference) {
		int idx = 0;
		for(DataPoint item : items) {
			System.out.println("[" + (idx++) + "] " + item + " => " + distance(item, reference));
		}
	}
	
	protected VpTree<StorageType>.Node createNode() {
		return new Node();
	}

	public Node getRoot() {
		return _root;
	}

	/**
	 * Partially orders {@code _items} and {@code _distances} in {@code [low, high)} around the
	 * element that belongs at position {@code nth}, so that everything before {@code nth} is not
	 * greater and everything after it is not smaller. This is the selection the tree needs; sorting
	 * the whole range, as the previous implementation did, costs an additional logarithmic factor.
	 *
	 * @param low first index of the range, inclusive
	 * @param nth index whose element has to end up in its sorted position
	 * @param high last index of the range, exclusive
	 */
	private void select(int low, int nth, int high) {
		int left = low;
		int right = high - 1;
		while (left < right) {
			double pivot = _distances[medianOfThree(left, right)];
			int i = left;
			int j = right;
			while (i <= j) {
				while (_distances[i] < pivot) i++;
				while (_distances[j] > pivot) j--;
				if (i <= j) {
					swap(i, j);
					i++;
					j--;
				}
			}
			if (nth <= j) {
				right = j;
			} else if (nth >= i) {
				left = i;
			} else {
				return; // nth ended up between the two partitions and is already in place
			}
		}
	}

	// Index of the median of the first, middle and last distance of the range, a pivot choice that
	// keeps the selection linear for the sorted and reverse sorted inputs a plain pivot degrades on
	private int medianOfThree(int left, int right) {
		int mid = (left + right) >>> 1;
		double a = _distances[left];
		double b = _distances[mid];
		double c = _distances[right];
		if (a < b) {
			if (b < c) return mid;
			return a < c ? right : left;
		}
		if (a < c) return left;
		return b < c ? right : mid;
	}

	// Swaps both the item and its distance to the current vantage point
	private void swap(int idx1, int idx2) {
		DataPoint item = _items[idx1];
		_items[idx1] = _items[idx2];
		_items[idx2] = item;
		double dist = _distances[idx1];
		_distances[idx1] = _distances[idx2];
		_distances[idx2] = dist;
	}

	/**
	 * Moves the element that belongs at position {@code mid} of the sorted range {@code [low, high)}
	 * to that position, and partitions the rest around it.
	 */
	static void nth_element(int [] array, int low, int mid, int high) {
		int left = low;
		int right = high - 1;
		while (left < right) {
			int pivot = array[medianOfThree(array, left, right)];
			int i = left;
			int j = right;
			while (i <= j) {
				while (array[i] < pivot) i++;
				while (array[j] > pivot) j--;
				if (i <= j) {
					int tmp = array[i];
					array[i] = array[j];
					array[j] = tmp;
					i++;
					j--;
				}
			}
			if (mid <= j) {
				right = j;
			} else if (mid >= i) {
				left = i;
			} else {
				return;
			}
		}
	}

	private static int medianOfThree(int [] array, int left, int right) {
		int mid = (left + right) >>> 1;
		int a = array[left];
		int b = array[mid];
		int c = array[right];
		if (a < b) {
			if (b < c) return mid;
			return a < c ? right : left;
		}
		if (a < c) return left;
		return b < c ? right : mid;
	}

	public double distance(DataPoint dataPoint1, DataPoint dataPoint2) {
		return distance.distance(dataPoint1, dataPoint2);
	}
	
	private void swap(DataPoint [] items, int idx1,int idx2) {
		DataPoint dp = items[idx1];
		items[idx1] = items[idx2];
		items[idx2] = dp;
	}

	class Node {
		int index;
		double threshold;
		protected Node left;
		protected Node right;
		
		@Override
		public String toString() {
			return "Node(id=" + index + ")";
		}
		
		public Node getLeft() {
			return left;
		}

		public Node getRight() {
			return right;
		}

		// Helper function that searches the tree
		double search(Node node, DataPoint target, int k, NeighborHeap heap, double _tau)
		{
			if(node == null) return _tau;     // indicates that we're done here

			// Compute distance between target and current node
			double dist = distance(_items[node.index], target);

			// If current node within radius tau
			if(dist < _tau) {
				// drops the farthest node from the result list if we already have k results
				heap.add(node.index, dist);
				if(heap.size() == k) {
					_tau = heap.maxDist(); // update value of tau (farthest point in result list)
				}
			}

			// Return if we arrived at a leaf
			if(node.left == null && node.right == null) {
				return _tau;
			}

			// If the target lies within the radius of ball
			if(dist < node.threshold) {
				if(dist - _tau <= node.threshold) {         // if there can still be neighbors inside the ball, recursively search left child first
					_tau = search(node.left, target, k, heap, _tau);
				}

				if(dist + _tau >= node.threshold) {         // if there can still be neighbors outside the ball, recursively search right child
					_tau = search(node.right, target, k, heap, _tau);
				}

				// If the target lies outside the radius of the ball
			} else {
				if(dist + _tau >= node.threshold) {         // if there can still be neighbors outside the ball, recursively search right child first
					_tau = search(node.right, target, k, heap, _tau);
				}

				if (dist - _tau <= node.threshold) {         // if there can still be neighbors inside the ball, recursively search left child
					_tau = search(node.left, target, k, heap, _tau);
				}
			}
			return _tau;
		}
	}

	// Print out tree
	void print() 
	{
		print_nodetree(_root,0);
	}	

	// Print out tree
	void print_nodetree(Node node, int lvl) 
	{
		String prefix = "";
		for(int i = 0; i < lvl; i++) {
			prefix += "    ";
		}

		if(node == null) {
			System.out.println(prefix + "Empty node");
		} else {
			System.out.println(prefix + "Node; index = [idx=" + node.index + " thr= " + node.threshold + "]");
			print_nodetree(node.left, lvl+1);
			print_nodetree(node.right, lvl+1);
		}
	}
}

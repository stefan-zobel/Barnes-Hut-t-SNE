package com.jujutsu.tsne.barneshut;

/**
 * A bounded max-heap of the k nearest neighbours found so far, keyed by distance.
 * <p>
 * It replaces a {@code PriorityQueue<HeapItem>}, which allocated one object per candidate the search
 * ever considered. The item index and its distance live in two primitive arrays, so a search visits
 * the whole tree without allocating anything.
 * <p>
 * The root is the <em>farthest</em> of the collected neighbours, which is what the search needs to
 * maintain its radius. Not thread-safe; every search uses its own heap.
 */
final class NeighborHeap {

    private final int[] index;
    private final double[] dist;
    private final int capacity;
    private int size;

    /**
     * @param capacity the number of neighbours to keep, must be positive
     */
    NeighborHeap(final int capacity) {
        if (capacity < 1) {
            throw new IllegalArgumentException("capacity must be positive: " + capacity);
        }
        this.capacity = capacity;
        this.index = new int[capacity];
        this.dist = new double[capacity];
    }

    int size() {
        return size;
    }

    boolean isFull() {
        return size == capacity;
    }

    /**
     * @return the distance of the farthest collected neighbour
     * @throws ArrayIndexOutOfBoundsException if the heap is empty
     */
    double maxDist() {
        return dist[0];
    }

    void clear() {
        size = 0;
    }

    /**
     * Offers a candidate. If the heap is already full, the candidate replaces the farthest collected
     * neighbour, or is discarded if it is not closer than that one.
     *
     * @param itemIndex position of the item in the tree
     * @param distance its distance to the search target
     */
    void add(final int itemIndex, final double distance) {
        if (size < capacity) {
            int child = size++;
            index[child] = itemIndex;
            dist[child] = distance;
            siftUp(child);
        } else if (distance < dist[0]) {
            index[0] = itemIndex;
            dist[0] = distance;
            siftDown();
        }
    }

    /**
     * Empties the heap into the given arrays, nearest neighbour first.
     *
     * @param items the items of the tree, indexed by the positions stored in this heap
     * @param outNeighbors receives the neighbours, must hold at least {@link #size()} elements
     * @param outDistances receives their distances, must hold at least {@link #size()} elements
     */
    void drainAscending(final DataPoint[] items, final DataPoint[] outNeighbors, final double[] outDistances) {
        for (int i = size - 1; i >= 0; i--) {
            outNeighbors[i] = items[index[0]];
            outDistances[i] = dist[0];
            --size;
            if (size > 0) {
                index[0] = index[size];
                dist[0] = dist[size];
                siftDown();
            }
        }
    }

    private void siftUp(int child) {
        while (child > 0) {
            int parent = (child - 1) >>> 1;
            if (dist[parent] >= dist[child]) {
                return;
            }
            swap(parent, child);
            child = parent;
        }
    }

    private void siftDown() {
        int parent = 0;
        while (true) {
            int left = 2 * parent + 1;
            if (left >= size) {
                return;
            }
            int largest = left;
            int right = left + 1;
            if (right < size && dist[right] > dist[left]) {
                largest = right;
            }
            if (dist[parent] >= dist[largest]) {
                return;
            }
            swap(parent, largest);
            parent = largest;
        }
    }

    private void swap(final int a, final int b) {
        int i = index[a];
        index[a] = index[b];
        index[b] = i;
        double d = dist[a];
        dist[a] = dist[b];
        dist[b] = d;
    }
}

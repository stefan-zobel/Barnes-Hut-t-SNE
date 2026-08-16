package com.jujutsu.tsne.barneshut;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import com.jujutsu.tsne.progress.TSneProgress;

public class ParallelVpTree<StorageType> extends VpTree<StorageType> {

    /** Name of the progress task reported by {@link #searchMultiple(ParallelVpTree, DataPoint[], int)}. */
    public static final String TASK_PERPLEXITY = "Perplexity";

    public ParallelVpTree(Distance distance) {
        super(distance);
    }

    /**
     * A tree whose vantage point choice is reproducible, see {@link VpTree#VpTree(Distance, long)}.
     * The search is parallel either way; only the build reads the generator, and it is single
     * threaded.
     *
     * @param distance the metric
     * @param seed seed for the vantage point choice
     */
    public ParallelVpTree(Distance distance, long seed) {
        super(distance, seed);
    }

    /**
     * Searches the k nearest neighbours of every target in parallel.
     *
     * @param tree the tree to search, whose root is used
     * @param targets the points to search neighbours for
     * @param k the number of neighbours per target
     * @return one result per target, in the order of {@code targets}
     */
    public List<TreeSearchResult> searchMultiple(ParallelVpTree<StorageType> tree, DataPoint [] targets, int k) {
        VpTree<StorageType>.Node node = tree.getRoot();

        TSneProgress.reset(TASK_PERPLEXITY, targets.length);
        // collecting from the stream keeps the encounter order and needs no lock, the previous
        // synchronized list serialized all worker threads on one monitor
        List<TreeSearchResult> results = IntStream.range(0, targets.length).parallel().mapToObj(n -> {
            DataPoint [] neighbors = new DataPoint[k];
            double [] distances = new double[k];
            NeighborHeap heap = new NeighborHeap(k);

            double tau = Double.MAX_VALUE;
            // Perform the search
            node.search(node, targets[n], k, heap, tau);

            // Gather final results, the heap yields them nearest first
            heap.drainAscending(_items, neighbors, distances);

            TSneProgress.update();
            return new TreeSearchResult(neighbors, distances, n);
        }).collect(Collectors.toList());
        TSneProgress.finished();

        return results;
    }
}

# Optimizations

What was changed in this fork of [T-SNE-Java](https://github.com/lejon/T-SNE-Java), what it is worth,
and what a caller has to know before upgrading.

**It has no dependencies left.** Everything the original took from third party libraries has been
replaced by code in this repository, so the only entry in the POM is JUnit at test scope and a build
that uses this pulls in nothing but this. The largest piece of that is the linear algebra behind the
PCA, which is where the last of them sat - see [The PCA](#the-pca) below.

## The result

One full run over the 60 000 image MNIST training set, two output dimensions, input reduced to 55 by
PCA, perplexity 20, 1000 iterations, on a Ryzen 5 5600H with 12 logical cores:

| | before | after | |
| --- | ---: | ---: | ---: |
| `tsne()` | 345.7 s | 220.9 s | **1.56x** |

The README says how to reproduce it. The algorithm is unchanged: this is the same Barnes-Hut t-SNE,
computed the same way.

## What was changed

### The gradient descent

Half of a long run. Rebuilding the space partitioning tree used to allocate a fresh array per point
*per level of the tree* - some 340 000 throw-away arrays per iteration at 20 000 points - and to find
the target child by testing all of them in turn. The points are now read out of the flat data array
through an offset, and the child follows directly from the signs of the coordinate differences. Leaf
nodes no longer carry a child array they never use.

The edge forces, the more expensive of the two force terms, ran on one thread even in the parallel
implementation. They are parallel over the rows now, which is safe because row `n` only ever writes
its own slice.

The work buffers of the gradient are allocated once per run instead of once per iteration, in both
implementations. The cost function no longer builds a space partitioning tree of its own: it takes the
normalization the gradient step of the same iteration has already computed.

Together this is the largest single gain in the table above: the gradient descent is about two thirds
of a long run, and it is 1.88x faster.

### The k nearest neighbour search

Building the ball tree partitioned each range by **sorting** it, where a linear selection is what the
algorithm needs, and the comparator evaluated the distance up to four times per comparison. Distances
to the vantage point are now computed once into an array and the range is partitioned by quickselect.
The build went from 0.99 s to 0.09 s at 60 000 points.

The search returned its distances as `List<Double>`, roughly 1.8 million boxed objects for 20 000
points, which the perplexity search then unboxed again in its innermost loop. Neighbours and distances
are plain arrays now, and the search itself uses a bounded primitive heap that allocates nothing at
all. The parallel search no longer collects into a synchronized list, so worker threads do not
serialize on one monitor, and the results keep their input order instead of arriving in whatever order
the threads finished.

Building the points of the ball tree used to copy every row twice - 63 MB of throw-away allocation for
5000 points of 784 dimensions, half of it garbage immediately. Each row is copied once now, read
straight out of the input matrix.

Copying none at all was tried, with the points as views into the matrix. It saves the other half and
costs more than it saves: reading the coordinates out of one shared array through an offset made the
search 1.25x slower at 20 000 points, which is far more than the allocation was worth. Half of that
was the shared array and a third the offset in the index, measured by taking them away one at a time.

### The PCA

This is where the last third party dependency sat. The original computes the reduction with
[org.ejml 0.41](https://github.com/lejon/T-SNE-Java/blob/master/tsne-core/src/main/java/com/jujutsu/tsne/PrincipalComponentAnalysis.java);
that library is gone. In its place are three classes in `math.linalg`: `JacobiPCA`, which decomposes
the matrix exactly through `FlatParallelJacobiSVD` - a one-sided Jacobi SVD in flat column-major
storage, ordered Brent-Luk so that the column-disjoint rotations of a sweep run concurrently - and
`TruncatedPCA` for the case below.

For high dimensional input this dominated everything else. Reducing 60 000 x 784 to 55 components with
the exact Jacobi decomposition takes 28.8 s on its own, and it computes all 784 components to keep 55.
A truncated method is used instead above a measured dimension threshold, with the exact one as a
fallback when the truncated one does not converge.

A second, redundant decomposition is gone as well. When the input has already been reduced it is
expressed in its own principal basis, so the leading components needed to initialize the embedding are
simply its leading columns.

### Throughout

The squared Euclidean distance - the innermost operation of both the tree build and the search -
accumulated into a single variable, which ties the loop to the latency of the floating point addition
chain rather than to throughput. It uses four independent accumulators above a threshold, which is
worth up to 1.9x at 200 dimensions and nothing at all at two, hence the threshold.

`Double.isNaN(...)` checks with a `System.out.println` sat in the innermost loops. They could not fix
anything and had no effect on the result, but when they did fire they cost 46x on the serial path and
161x on the parallel one, where every worker thread had to take the print stream's monitor.

About 400 lines of unreachable code were removed, including two methods no caller could reach and a
normalization step that computed a divisor nothing read.

### One phase used to be slower

The perplexity and kNN phase was about 1.1x slower than the original for a while. It is at parity now
- 1.002x over fifteen blocked repetitions, which is to say indistinguishable - and the rest of the
table stands as it is.

Finding that out took the detour worth recording here. The searches of both versions visit the same
number of tree nodes to within 0.0004 %, so the ball tree was as good as it had been and the cost had
to be per visit; it turned out to be the flat shared layout of the points described above.

## Behaviour that changed

Two things return different numbers than they did. Neither is a defect, but a caller comparing against
recorded output will see them.

**Short runs behave differently, and better.** The early exaggeration used to end at a hard coded
iteration 250, against a reference default of 1000 iterations. Taken literally that breaks every
shorter run: at 250 iterations or fewer the exaggeration is never switched off and the embedding is
returned in its inflated state, with no warning. It now ends after a quarter of the run, which
reproduces the reference exactly at 1000 iterations and above and keeps the proportion below.

**Embeddings of high dimensional input differ.** The initialization goes through the truncated PCA,
which is deterministic but approximate. The difference is far below what the stochastic parts of the
pipeline contribute, and the embedding is scaled to 1e-4 and re-centered from the first iteration
onward, but it is not bit for bit what it was.

## Compatibility with the previous API

Both versions were compiled and compared with `javap -protected`, class by class, member set against
member set: 314 public and protected members on each side.

No class declaration changed - nothing narrowed its visibility, swapped a supertype or became final,
and the declared `throws` clauses are identical.

One surviving class had a changed signature, `TreeSearchResult`, whose constructor took two lists
before the neighbours and distances moved into arrays. **It is restored as a delegation** and marked
deprecated, so no signature that outlived the work is broken. What the delegation cannot reproduce is
that the old constructor kept the caller's list objects, so a later change to either showed through;
this one reads them out.

Everything else added to an existing class is additive: `TreeSearchResult.getNeighbors()` and
`getNeighborDistances()`, a `DataPoint` constructor taking a flat matrix and an offset, an array based
`VpTree.search`, seeded `VpTree` and `ParallelVpTree` constructors, and `BHTSne.lastSumQ`.

Three members were removed deliberately: `ParallelVpTree.searchMultipleWOProgress`, `TSneUtils.check`
and `SPTree.buff`. The last was `protected` and therefore counted as API for subclasses.

One contract changed without a signature moving: `TreeSearchResult.getIndices()` and `getDistances()`
return views rather than the stored lists, so writing into the result throws where it used to work.
Both have been marked as deprecated.

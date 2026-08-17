package com.jujutsu.tsne.barneshut;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNotSame;

import java.util.Random;

import org.junit.Test;

import com.jujutsu.utils.MatrixOps;

/**
 * {@link DataPoint} holds its own copy of one row of the flat data matrix, taken straight out of it,
 * so that building the ball tree copies each row once rather than the twice the original did.
 * <p>
 * It was a view into the matrix for a while, copying nothing at all. That is what these tests mostly
 * guard against coming back: it saved the copy and lost more than it saved in the kNN search, which
 * reads these coordinates billions of times. So the row length and the fact that the point is
 * detached from the matrix are pinned deliberately - a view would pass the coordinate tests and fail
 * these.
 * <p>
 * The other requirement is that a point built from an offset sees exactly what a point built from an
 * extracted row sees, down to the last bit of the distances, because the embedding is compared bit for
 * bit against a recorded baseline. The dimensionalities used here are 5 and 37: 5 takes the plain loop
 * of {@link EuclideanDistance#squaredDistance(double[], double[], int)}, 37 takes the unrolled one and
 * leaves a remainder, so both paths are exercised.
 */
public class DataPointRowTest {

    private static final int N = 9;

    private static double[] flatMatrix(int n, int d, long seed) {
        final Random random = new Random(seed);
        final double[] x = new double[n * d];
        for (int i = 0; i < x.length; i++) {
            x[i] = random.nextGaussian();
        }
        return x;
    }

    /** the point built the long way round, through an extracted row */
    private static DataPoint fromExtractedRow(double[] x, int row, int d) {
        return new DataPoint(d, row, MatrixOps.extractRowFromFlatMatrix(x, row, d));
    }

    @Test
    public void aPointReportsTheCoordinatesOfItsOwnRow() {
        for (int d : new int[] { 5, 37 }) {
            final double[] x = flatMatrix(N, d, 20260816L + d);
            for (int row = 0; row < N; row++) {
                final DataPoint point = new DataPoint(x, row * d, d, row);

                assertEquals("index of row " + row, row, point.index());
                assertEquals("dimensionality of row " + row, d, point.dimensionality());
                for (int i = 0; i < d; i++) {
                    assertEquals("row " + row + " component " + i, x[row * d + i], point.x(i), 0.0);
                }
            }
        }
    }

    @Test
    public void theOffsetSelectsTheRowAndIsNotIgnored() {
        final int d = 5;
        final double[] x = flatMatrix(N, d, 4711L);
        final DataPoint point = new DataPoint(x, 3 * d, d, 3);

        assertEquals("first component of row 3", x[3 * d], point.x(0), 0.0);
        // would hold if the offset were ignored, so it is what pins the offset down
        assertNotEquals("row 3 must not report row 0", x[0], point.x(0), 0.0);
    }

    @Test
    public void distancesFromAnOffsetAreBitIdenticalToDistancesFromAnExtractedRow() {
        for (int d : new int[] { 5, 37 }) {
            final double[] x = flatMatrix(N, d, 20260816L + d);
            final EuclideanDistance distance = new EuclideanDistance();

            for (int a = 0; a < N; a++) {
                for (int b = 0; b < N; b++) {
                    final DataPoint offsetA = new DataPoint(x, a * d, d, a);
                    final DataPoint offsetB = new DataPoint(x, b * d, d, b);
                    final DataPoint rowA = fromExtractedRow(x, a, d);
                    final DataPoint rowB = fromExtractedRow(x, b, d);
                    final String where = "D = " + d + ", rows " + a + " and " + b;

                    assertEquals(where, distance.distance(rowA, rowB), distance.distance(offsetA, offsetB), 0.0);
                    assertEquals(where, rowA.euclidean_distance(rowB), offsetA.euclidean_distance(offsetB), 0.0);
                    assertEquals(where, DataPoint.euclidean_distance(rowA, rowB),
                            DataPoint.euclidean_distance(offsetA, offsetB), 0.0);
                }
            }
        }
    }

    @Test
    public void aPointIsDetachedFromTheArrayItWasBuiltFrom() {
        // this is the test a view would fail. Both constructors copy, so the matrix can be changed or
        // dropped afterwards and the point does not notice.
        final int d = 5;
        final double[] x = flatMatrix(N, d, 99L);
        final DataPoint fromMatrix = new DataPoint(x, 2 * d, d, 2);
        final DataPoint fromRow = new DataPoint(d, 0, x);
        final double beforeMatrix = fromMatrix.x(0);
        final double beforeRow = fromRow.x(0);

        x[2 * d] = 12345.0;
        x[0] = -12345.0;

        assertEquals("the point kept its own copy", beforeMatrix, fromMatrix.x(0), 0.0);
        assertEquals("the point kept its own copy", beforeRow, fromRow.x(0), 0.0);
        assertNotSame("and it is not the caller's array", x, fromRow._x);
    }

    @Test
    public void aPointHoldsOneRowAndNotTheWholeMatrix() {
        // a view would hold N * D here, keeping the entire data set reachable through any one point
        final int d = 37;
        final double[] x = flatMatrix(N, d, 1234L);

        for (int row = 0; row < N; row++) {
            assertEquals("row " + row + " holds D coordinates", d, new DataPoint(x, row * d, d, row)._x.length);
        }
    }

    @Test
    public void rowPointsCoversEveryRowInOrder() {
        final int d = 37;
        final double[] x = flatMatrix(N, d, 815L);
        final EuclideanDistance distance = new EuclideanDistance();

        final DataPoint[] points = BHTSne.rowPoints(x, N, d);

        assertEquals("one point per row", N, points.length);
        for (int row = 0; row < N; row++) {
            assertEquals("index of point " + row, row, points[row].index());
            // zero distance to a copy of the same row is the strongest statement about all D components
            assertEquals("point " + row + " is row " + row, 0.0,
                    distance.distance(points[row], fromExtractedRow(x, row, d)), 0.0);
        }
    }
}

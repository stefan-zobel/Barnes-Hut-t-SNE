package com.jujutsu.tsne.barneshut;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

import java.util.Random;

import org.junit.Test;

import com.jujutsu.utils.MatrixOps;

/**
 * {@link DataPoint} is a view of one row of a flat matrix rather than a copy of it, so that building
 * the ball tree does not duplicate the whole data set - twice, as it used to.
 * <p>
 * What has to hold is that a view sees exactly what a copy of the same row sees, down to the last bit
 * of the distances, because the embedding is compared bit for bit against a recorded baseline. The
 * dimensionalities used here are 5 and 37: 5 takes the plain loop of
 * {@link EuclideanDistance#squaredDistance(double[], int, double[], int, int)}, 37 takes the unrolled
 * one and leaves a remainder, so both paths are exercised with a non-zero offset.
 */
public class DataPointViewTest {

	private static final int N = 9;

	private static double[] flatMatrix(int n, int d, long seed) {
		final Random random = new Random(seed);
		final double[] x = new double[n * d];
		for (int i = 0; i < x.length; i++) {
			x[i] = random.nextGaussian();
		}
		return x;
	}

	/** the point as it was built before F8: its own copy of the row */
	private static DataPoint owningCopy(double[] x, int row, int d) {
		return new DataPoint(d, row, MatrixOps.extractRowFromFlatMatrix(x, row, d));
	}

	@Test
	public void aViewReportsTheCoordinatesOfItsOwnRow() {
		for (int d : new int[] { 5, 37 }) {
			final double[] x = flatMatrix(N, d, 20260816L + d);
			for (int row = 0; row < N; row++) {
				final DataPoint view = new DataPoint(x, row * d, d, row);

				assertEquals("index of row " + row, row, view.index());
				assertEquals("dimensionality of row " + row, d, view.dimensionality());
				for (int i = 0; i < d; i++) {
					assertEquals("row " + row + " component " + i, x[row * d + i], view.x(i), 0.0);
				}
			}
		}
	}

	@Test
	public void aViewIsAtItsOffsetAndNotAtTheStartOfTheArray() {
		final int d = 5;
		final double[] x = flatMatrix(N, d, 4711L);
		final DataPoint view = new DataPoint(x, 3 * d, d, 3);

		assertEquals("first component of row 3", x[3 * d], view.x(0), 0.0);
		// would hold if the offset were ignored, so it is what pins the offset down
		assertNotEquals("row 3 must not report row 0", x[0], view.x(0), 0.0);
	}

	@Test
	public void distancesBetweenViewsAreBitIdenticalToDistancesBetweenCopies() {
		for (int d : new int[] { 5, 37 }) {
			final double[] x = flatMatrix(N, d, 20260816L + d);
			final EuclideanDistance distance = new EuclideanDistance();

			for (int a = 0; a < N; a++) {
				for (int b = 0; b < N; b++) {
					final DataPoint viewA = new DataPoint(x, a * d, d, a);
					final DataPoint viewB = new DataPoint(x, b * d, d, b);
					final DataPoint copyA = owningCopy(x, a, d);
					final DataPoint copyB = owningCopy(x, b, d);
					final String where = "D = " + d + ", rows " + a + " and " + b;

					assertEquals(where, distance.distance(copyA, copyB), distance.distance(viewA, viewB), 0.0);
					assertEquals(where, copyA.euclidean_distance(copyB), viewA.euclidean_distance(viewB), 0.0);
					assertEquals(where, DataPoint.euclidean_distance(copyA, copyB),
							DataPoint.euclidean_distance(viewA, viewB), 0.0);
				}
			}
		}
	}

	@Test
	public void aPointDoesNotCopyTheArrayItIsGiven() {
		// deliberate: re-introducing a defensive copy would cost the memory F8 saved, so it has to
		// fail a test rather than pass unnoticed
		final int d = 5;
		final double[] x = flatMatrix(N, d, 99L);
		final DataPoint view = new DataPoint(x, 2 * d, d, 2);
		final DataPoint standalone = new DataPoint(d, 0, x);

		x[2 * d] = 12345.0;
		x[0] = -12345.0;

		assertEquals("the view reads through to the array", 12345.0, view.x(0), 0.0);
		assertEquals("the standalone point reads through as well", -12345.0, standalone.x(0), 0.0);
	}

	@Test
	public void rowViewsCoversEveryRowInOrder() {
		final int d = 37;
		final double[] x = flatMatrix(N, d, 815L);
		final EuclideanDistance distance = new EuclideanDistance();

		final DataPoint[] points = BHTSne.rowViews(x, N, d);

		assertEquals("one point per row", N, points.length);
		for (int row = 0; row < N; row++) {
			assertEquals("index of point " + row, row, points[row].index());
			// zero distance to a copy of the same row is the strongest statement about all D components
			assertEquals("point " + row + " is row " + row, 0.0,
					distance.distance(points[row], owningCopy(x, row, d)), 0.0);
		}
	}
}

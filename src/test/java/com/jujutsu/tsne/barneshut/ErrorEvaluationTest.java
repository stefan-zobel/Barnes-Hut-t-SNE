package com.jujutsu.tsne.barneshut;

import static java.lang.Math.log;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Random;

import org.junit.Before;
import org.junit.Test;

/**
 * Tests for the t-SNE cost function.
 * <p>
 * {@link BHTSne#klDivergence} reuses the normalization term of the gradient step instead of building
 * a space partitioning tree of its own, so it has to agree both with an exact computation and with
 * the tree building implementation it replaces.
 */
public class ErrorEvaluationTest {

	private static final int N = 300;
	private static final int DIMS = 2;
	private static final int K = 15;

	private double[] Y;
	private int[] row_P;
	private int[] col_P;
	private double[] val_P;

	@Before
	public void setUp() {
		final Random random = new Random(19740101L);
		// spread out, so that no two points coincide: the tree drops duplicates but still counts them
		// in cum_size, which would make the tree based and the exact normalization differ
		Y = new double[N * DIMS];
		for (int i = 0; i < Y.length; i++) {
			Y[i] = random.nextGaussian() * 5.0;
		}

		row_P = new int[N + 1];
		col_P = new int[N * K];
		val_P = new double[N * K];
		double sum = 0.0;
		for (int n = 0; n < N; n++) {
			row_P[n + 1] = row_P[n] + K;
			for (int m = 0; m < K; m++) {
				int neighbour = (n + 1 + m * 7) % N;
				col_P[row_P[n] + m] = neighbour;
				double value = 1.0 + ((n + m) % 5);
				val_P[row_P[n] + m] = value;
				sum += value;
			}
		}
		for (int i = 0; i < val_P.length; i++) {
			val_P[i] /= sum;
		}
	}

	/** Normalization of the Q distribution, summed over all pairs. */
	private double exactSumQ() {
		double sumQ = 0.0;
		for (int n = 0; n < N; n++) {
			for (int m = 0; m < N; m++) {
				if (n != m) {
					sumQ += 1.0 / (1.0 + squaredDistance(n, m));
				}
			}
		}
		return sumQ;
	}

	private double exactKlDivergence() {
		final double sumQ = exactSumQ();
		double C = 0.0;
		for (int n = 0; n < N; n++) {
			for (int i = row_P[n]; i < row_P[n + 1]; i++) {
				double q = (1.0 / (1.0 + squaredDistance(n, col_P[i]))) / sumQ;
				C += val_P[i] * log((val_P[i] + Double.MIN_VALUE) / (q + Double.MIN_VALUE));
			}
		}
		return C;
	}

	private double squaredDistance(int a, int b) {
		double d2 = 0.0;
		for (int d = 0; d < DIMS; d++) {
			double diff = Y[a * DIMS + d] - Y[b * DIMS + d];
			d2 += diff * diff;
		}
		return d2;
	}

	/** Runs the gradient step, which is what supplies {@link BHTSne#lastSumQ}. */
	private double costOf(BHTSne tsne, double theta) {
		double[] dC = new double[N * DIMS];
		tsne.computeGradient(row_P, col_P, val_P, Y, N, DIMS, dC, theta, 0);
		return tsne.klDivergence(row_P, col_P, val_P, Y, N, DIMS, tsne.lastSumQ);
	}

	@Test
	public void isExactWhenTheApproximationIsTurnedOff() {
		// with theta = 0 the Barnes-Hut criterion never applies and the traversal descends to the
		// leaves, so the normalization is the exact sum over all pairs
		assertEquals(exactKlDivergence(), costOf(new BHTSne(), 0.0), 1e-9);
	}

	@Test
	public void isExactWhenTheApproximationIsTurnedOffInParallel() {
		assertEquals(exactKlDivergence(), costOf(new ParallelBHTsne(), 0.0), 1e-9);
	}

	@Test
	public void serialAndParallelAgree() {
		for (double theta : new double[] {0.2, 0.5, 0.8}) {
			assertEquals("theta " + theta, costOf(new BHTSne(), theta), costOf(new ParallelBHTsne(), theta),
					1e-12 * Math.abs(costOf(new BHTSne(), theta)));
		}
	}

	@Test
	public void theApproximationStaysCloseToTheExactCost() {
		double exact = exactKlDivergence();
		double approximate = costOf(new BHTSne(), 0.5);
		assertTrue("approximate cost " + approximate + " too far from " + exact,
				Math.abs(approximate - exact) < 0.05 * Math.abs(exact));
	}
}

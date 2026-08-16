package com.jujutsu.tsne.barneshut;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;

import java.util.Random;

import org.junit.Test;

import math.linalg.JacobiPCA;

/**
 * Tests for the input reduction that {@code usePca()} performs, {@link BHTSne#reduceInput}.
 * <p>
 * Everything downstream of the reduction - the vantage point tree, the kNN search, the perplexity
 * search - sees only <em>distances</em> between the reduced samples. Those are invariant under a
 * rotation inside the retained subspace, so the truncated method is not required to find the same
 * individual directions as the exact one; it is required to find the same subspace, keep the same
 * amount of variance, and leave the distances alone. These tests assert that, not per-component
 * agreement, which would be asserting something the pipeline does not care about.
 */
public class InputReductionTest {

	private static double[][] decayingSpectrum(int m, int n, long seed) {
		Random random = new Random(seed);
		int factors = Math.min(40, n);
		double[][] directions = new double[factors][n];
		for (int k = 0; k < factors; k++) {
			for (int j = 0; j < n; j++) directions[k][j] = random.nextGaussian();
		}
		double[][] data = new double[m][n];
		for (int i = 0; i < m; i++) {
			double[] coefficients = new double[factors];
			for (int k = 0; k < factors; k++) coefficients[k] = random.nextGaussian() / (k + 1.0);
			for (int j = 0; j < n; j++) {
				double value = random.nextGaussian() * 0.3;
				for (int k = 0; k < factors; k++) value += coefficients[k] * directions[k][j];
				data[i][j] = value;
			}
		}
		return data;
	}

	private static double distance(double[] u, double[] v) {
		double sum = 0.0;
		for (int j = 0; j < u.length; j++) {
			double d = u[j] - v[j];
			sum += d * d;
		}
		return Math.sqrt(sum);
	}

	/** total variance retained by a reduction, i.e. the sum of the column variances */
	private static double capturedVariance(double[][] reduced) {
		double total = 0.0;
		for (int c = 0; c < reduced[0].length; c++) {
			double mean = 0.0;
			for (double[] row : reduced) mean += row[c];
			mean /= reduced.length;
			double variance = 0.0;
			for (double[] row : reduced) {
				double d = row[c] - mean;
				variance += d * d;
			}
			total += variance;
		}
		return total;
	}

	/** largest relative deviation of the pairwise distances over a deterministic sample of pairs */
	private static double worstDistanceDeviation(double[][] expected, double[][] actual) {
		Random random = new Random(99);
		double worst = 0.0;
		for (int t = 0; t < 20_000; t++) {
			int i = random.nextInt(expected.length);
			int j = random.nextInt(expected.length);
			if (i == j) continue;
			double reference = distance(expected[i], expected[j]);
			if (reference <= 0.0) continue;
			worst = Math.max(worst, Math.abs(distance(actual[i], actual[j]) - reference) / reference);
		}
		return worst;
	}

	@Test
	public void usesTheExactDecompositionForLowDimensionalInput() {
		// below the feature threshold the exact path costs almost nothing, so it must still be taken
		// and the result must be identical to it, bit for bit
		double[][] data = decayingSpectrum(200, 60, 13L);

		double[][] actual = BHTSne.reduceInput(data, 10);
		double[][] expected = new JacobiPCA().pca(data, 10);

		for (int i = 0; i < data.length; i++) {
			assertArrayEquals("row " + i, expected[i], actual[i], 0.0);
		}
	}

	@Test
	public void usesTheExactDecompositionWhenMostComponentsAreKept() {
		// keeping a large share of the features leaves nothing for the truncated method to save,
		// so the exact one must run - again bit for bit
		double[][] data = decayingSpectrum(200, 100, 17L);

		double[][] actual = BHTSne.reduceInput(data, 80);
		double[][] expected = new JacobiPCA().pca(data, 80);

		for (int i = 0; i < data.length; i++) {
			assertArrayEquals("row " + i, expected[i], actual[i], 0.0);
		}
	}

	@Test
	public void keepsTheVarianceTheExactReductionKeeps() {
		double[][] data = decayingSpectrum(400, 400, 23L);

		double[][] actual = BHTSne.reduceInput(data, 55);
		double[][] expected = new JacobiPCA().pca(data, 55);

		double ratio = capturedVariance(actual) / capturedVariance(expected);
		assertTrue("captured variance ratio was " + ratio, ratio > 0.995 && ratio <= 1.0 + 1e-9);
	}

	@Test
	public void preservesThePairwiseDistancesTheKnnSearchSees() {
		double[][] data = decayingSpectrum(400, 400, 29L);

		double[][] actual = BHTSne.reduceInput(data, 55);
		double[][] expected = new JacobiPCA().pca(data, 55);

		double worst = worstDistanceDeviation(expected, actual);
		assertTrue("worst relative distance deviation was " + worst, worst < 0.05);
	}

	@Test
	public void agreesWithTheExactReductionOnTheLeadingComponents() {
		// the leading directions are the accurate end of the truncated output, and BHTSne relies on
		// exactly that when it takes the leading columns to initialize the embedding
		double[][] data = decayingSpectrum(400, 400, 31L);

		double[][] actual = BHTSne.reduceInput(data, 55);
		double[][] expected = new JacobiPCA().pca(data, 55);

		for (int c = 0; c < 2; c++) {
			double scale = 0.0, deviation = 0.0;
			for (int i = 0; i < data.length; i++) scale = Math.max(scale, Math.abs(expected[i][c]));
			for (int i = 0; i < data.length; i++) {
				deviation = Math.max(deviation, Math.abs(expected[i][c] - actual[i][c]));
			}
			assertTrue("component " + c + " deviated by " + (deviation / scale), deviation / scale < 1e-3);
		}
	}

	@Test
	public void reducesDeterministically() {
		double[][] data = decayingSpectrum(300, 300, 37L);

		double[][] first = BHTSne.reduceInput(data, 40);
		double[][] second = BHTSne.reduceInput(data, 40);

		for (int i = 0; i < data.length; i++) {
			assertArrayEquals("row " + i, first[i], second[i], 0.0);
		}
	}

	@Test
	public void leadingColumnsStillReproduceAPcaOverTruncatedlyReducedData() {
		// option 1 takes the leading columns of the reduction instead of decomposing it again. That
		// shortcut must stay valid now that the reduction itself is truncated.
		double[][] reduced = BHTSne.reduceInput(decayingSpectrum(300, 300, 41L), 40);

		double[][] shortcut = BHTSne.leadingColumns(reduced, 2);
		double[][] full = new JacobiPCA().pca(reduced, 2);

		double scale = 0.0;
		for (double[] row : full) scale = Math.max(scale, Math.abs(row[0]));
		for (int i = 0; i < reduced.length; i++) {
			assertArrayEquals("row " + i, full[i], shortcut[i], 1e-10 * scale);
		}
	}
}

package com.jujutsu.tsne.barneshut;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;

import java.util.Random;

import org.junit.Test;

import math.linalg.JacobiPCA;

/**
 * Tests for how {@link BHTSne} obtains the principal components that initialize the embedding.
 * <p>
 * Two shortcuts are involved. When the input has already been reduced by a PCA, the leading columns
 * of that reduction <em>are</em> the leading components, so no second decomposition is needed. When
 * it has not, and the input is high dimensional, a truncated method computes the two components
 * that are wanted instead of all of them - with a fallback to the exact decomposition where it does
 * not settle on an answer.
 */
public class EmbeddingInitializationTest {

	private static double[][] decayingSpectrum(int m, int n, long seed) {
		Random random = new Random(seed);
		int factors = Math.min(20, n);
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

	private static double deviation(double[][] expected, double[][] actual, int c) {
		double scale = 0.0, plus = 0.0, minus = 0.0;
		for (int i = 0; i < expected.length; i++) scale = Math.max(scale, Math.abs(expected[i][c]));
		for (int i = 0; i < expected.length; i++) {
			plus = Math.max(plus, Math.abs(expected[i][c] - actual[i][c]));
			minus = Math.max(minus, Math.abs(expected[i][c] + actual[i][c]));
		}
		return Math.min(plus, minus) / scale;
	}

	@Test
	public void leadingColumnsReproduceAPcaOverAlreadyReducedData() {
		// after a reduction the data sits in its own principal basis, so a second PCA over it can
		// only return its leading columns - which is exactly what the shortcut takes
		double[][] reduced = new JacobiPCA().pca(decayingSpectrum(300, 90, 11L), 40);

		double[][] shortcut = BHTSne.leadingColumns(reduced, 2);
		double[][] full = new JacobiPCA().pca(reduced, 2);

		double scale = 0.0;
		for (double[] row : full) scale = Math.max(scale, Math.abs(row[0]));
		for (int i = 0; i < reduced.length; i++) {
			assertArrayEquals("row " + i, full[i], shortcut[i], 1e-10 * scale);
		}
	}

	@Test
	public void usesTheExactDecompositionForLowDimensionalInput() {
		// below the threshold the exact path costs almost nothing, so it must still be taken and
		// the result must be identical to it, bit for bit
		double[][] data = decayingSpectrum(300, 40, 13L);

		double[][] actual = BHTSne.initialComponents(data, 2);
		double[][] expected = new JacobiPCA().pca(data, 2);

		for (int i = 0; i < data.length; i++) {
			assertArrayEquals("row " + i, expected[i], actual[i], 0.0);
		}
	}

	@Test
	public void agreesWithTheExactDecompositionForHighDimensionalInput() {
		double[][] data = decayingSpectrum(400, 200, 17L);

		double[][] actual = BHTSne.initialComponents(data, 2);
		double[][] expected = new JacobiPCA().pca(data, 2);

		assertTrue("component 1: " + deviation(expected, actual, 0), deviation(expected, actual, 0) < 1e-4);
		assertTrue("component 2: " + deviation(expected, actual, 1), deviation(expected, actual, 1) < 1e-4);
	}

	@Test
	public void fallsBackToTheExactDecompositionOnStructurelessInput() {
		// structureless data neither settles nor leaves the trailing components negligible, so the
		// exact decomposition has to run - the result must be identical to it, bit for bit
		Random random = new Random(5);
		double[][] noise = new double[300][200];
		for (int i = 0; i < noise.length; i++) {
			for (int j = 0; j < noise[i].length; j++) noise[i][j] = random.nextGaussian();
		}

		double[][] actual = BHTSne.initialComponents(noise, 2);
		double[][] expected = new JacobiPCA().pca(noise, 2);

		for (int i = 0; i < noise.length; i++) {
			assertArrayEquals("row " + i, expected[i], actual[i], 0.0);
		}
	}

	@Test
	public void acceptsATruncatedResultWhoseTrailingComponentCarriesNoVariance() {
		// one dominant direction, noise behind it: the second direction is not determined by the
		// data, so the truncated result is accepted although its Ritz values never fully settle
		Random random = new Random(7);
		double[][] data = new double[400][200];
		for (int i = 0; i < data.length; i++) {
			for (int j = 0; j < data[i].length; j++) data[i][j] = random.nextGaussian() + (i % 10) * 3.0;
		}

		double[][] actual = BHTSne.initialComponents(data, 2);
		double[][] expected = new JacobiPCA().pca(data, 2);

		assertTrue("the leading component must still be right: " + deviation(expected, actual, 0),
				deviation(expected, actual, 0) < 1e-8);
	}
}

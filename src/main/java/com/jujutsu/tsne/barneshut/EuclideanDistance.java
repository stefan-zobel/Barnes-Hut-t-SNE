package com.jujutsu.tsne.barneshut;

import static java.lang.Math.sqrt;

public class EuclideanDistance implements Distance{

	// Below this dimensionality the remainder handling of the unrolled loop costs more than the
	// additional accumulators save
	private static final int UNROLL_THRESHOLD = 16;

	public EuclideanDistance() {
	}

	@Override
	public double distance(DataPoint d1, DataPoint d2) {
		return sqrt(squaredDistance(d1._x, d2._x, d1._D));
	}

	/**
	 * Squared Euclidean distance of the first {@code D} components of two vectors.
	 * <p>
	 * For higher dimensionalities the sum is accumulated in four independent accumulators. A single
	 * accumulator makes the loop dependent on the latency of the floating point add chain, four of
	 * them keep the pipeline busy. Measured on JDK 25 this is about 1.1x faster for {@code D = 50}
	 * and 1.9x for {@code D = 200}, while for small {@code D} the plain loop wins, hence the
	 * threshold. Note that the summation order differs between the two paths, so the results can
	 * differ in the last bits.
	 *
	 * @param x1 first vector, at least {@code D} elements
	 * @param x2 second vector, at least {@code D} elements
	 * @param D the number of components to use
	 * @return the squared Euclidean distance
	 */
	static double squaredDistance(double[] x1, double[] x2, int D) {
		if (D < UNROLL_THRESHOLD) {
			double dd = .0;
			for (int d = 0; d < D; d++) {
				double diff = x1[d] - x2[d];
				dd += diff * diff;
			}
			return dd;
		}
		double s0 = .0, s1 = .0, s2 = .0, s3 = .0;
		int d = 0;
		int limit = D - 3;
		for (; d < limit; d += 4) {
			double d0 = x1[d]     - x2[d];
			double d1 = x1[d + 1] - x2[d + 1];
			double d2 = x1[d + 2] - x2[d + 2];
			double d3 = x1[d + 3] - x2[d + 3];
			s0 += d0 * d0;
			s1 += d1 * d1;
			s2 += d2 * d2;
			s3 += d3 * d3;
		}
		double dd = (s0 + s1) + (s2 + s3);
		for (; d < D; d++) {
			double diff = x1[d] - x2[d];
			dd += diff * diff;
		}
		return dd;
	}
}

package math.linalg;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/**
 * Characterization test for {@link JacobiPCA}.
 * <p>
 * {@link TruncatedPCA} was added as an alternative for the case where only the leading components
 * are wanted, on the explicit condition that {@code JacobiPCA} itself keeps behaving exactly as
 * before. The expected values below were taken from the implementation at that point, so any change
 * in its numerical behaviour - not only in its API - fails the build.
 */
public class JacobiPCATest {

	private static final double TOLERANCE = 1e-12;

	/** tall input, m > n, which takes the QR preconditioned path */
	private static final double[][] TALL = {
			{ 2.5, 2.4, 0.5, 1.1 },
			{ 0.5, 0.7, 1.5, 2.2 },
			{ 2.2, 2.9, 0.2, 1.4 },
			{ 1.9, 2.2, 0.7, 0.9 },
			{ 3.1, 3.0, 1.1, 2.5 },
			{ 2.3, 2.7, 0.4, 1.8 },
	};

	/** wide input, m < n, which takes the transposed path */
	private static final double[][] WIDE = {
			{ 1.0, 2.0, 3.0, 4.0, 5.0 },
			{ 2.0, 1.0, 4.0, 3.0, 6.0 },
			{ 5.0, 4.0, 1.0, 2.0, 0.5 },
	};

	@Test
	public void reproducesTheKnownDecompositionOfATallMatrix() {
		JacobiPCA pca = new JacobiPCA();
		double[][] projected = pca.pca(TALL, 2);

		assertArrayEquals(new double[] { 0.4574185511894293, -0.4709213830305949 }, projected[0], TOLERANCE);
		assertArrayEquals(new double[] { -2.4293158992819492, 0.327117711296727 }, projected[1], TOLERANCE);
		assertArrayEquals(new double[] { 0.6441544258344503, -0.38209579128203747 }, projected[2], TOLERANCE);
		assertArrayEquals(new double[] { -0.11912223827661744, -0.7198173160087022 }, projected[3], TOLERANCE);
		assertArrayEquals(new double[] { 0.964711184062765, 1.1808663748574804 }, projected[4], TOLERANCE);
		assertArrayEquals(new double[] { 0.4821539764719231, 0.06485040416712537 }, projected[5], TOLERANCE);

		assertArrayEquals(new double[] { 2.7754409788597387, 1.5464663198306525 }, pca.getSingularValues(), TOLERANCE);
		assertArrayEquals(new double[] { 0.6775549162730331, 0.6749529498679511, -0.27466641745585624,
				-0.09958016881267225 }, pca.getComponents()[0], TOLERANCE);
		assertArrayEquals(new double[] { 0.24579518005180506, 0.0521795417178138, 0.41806125430103686,
				0.8729758373183442 }, pca.getComponents()[1], TOLERANCE);
		assertArrayEquals(new double[] { 2.0833333333333335, 2.3166666666666664, 0.7333333333333334,
				1.6500000000000004 }, pca.getMean(), TOLERANCE);
	}

	@Test
	public void reproducesTheKnownDecompositionOfAWideMatrix() {
		JacobiPCA pca = new JacobiPCA();
		double[][] projected = pca.pca(WIDE, 2);

		assertArrayEquals(new double[] { 2.0325129687346717, -1.109054616795982 }, projected[0], TOLERANCE);
		assertArrayEquals(new double[] { 2.773665481501946, 1.000611933353296 }, projected[1], TOLERANCE);
		assertArrayEquals(new double[] { -4.806178450236619, 0.10844268344268637 }, projected[2], TOLERANCE);

		assertArrayEquals(new double[] { 5.909626085199378, 1.497660241776403 }, pca.getSingularValues(), TOLERANCE);
	}

	@Test
	public void canonicalizesTheSignOfEveryComponent() {
		JacobiPCA pca = new JacobiPCA();
		pca.pca(TALL, 3);
		for (double[] component : pca.getComponents()) {
			int argmax = 0;
			for (int j = 0; j < component.length; j++) {
				if (Math.abs(component[j]) > Math.abs(component[argmax])) argmax = j;
			}
			assertTrue("largest magnitude entry must be positive", component[argmax] > 0.0);
		}
	}

	@Test
	public void componentsAreOrthonormalAndSingularValuesDescend() {
		JacobiPCA pca = new JacobiPCA();
		pca.pca(TALL, 3);
		double[][] c = pca.getComponents();
		for (int p = 0; p < c.length; p++) {
			for (int q = 0; q < c.length; q++) {
				double dot = 0.0;
				for (int j = 0; j < c[p].length; j++) dot += c[p][j] * c[q][j];
				assertEquals(p == q ? 1.0 : 0.0, dot, 1e-12);
			}
		}
		double[] sv = pca.getSingularValues();
		for (int k = 0; k + 1 < sv.length; k++) {
			assertTrue("singular values must descend", sv[k] >= sv[k + 1]);
		}
	}

	@Test
	public void isDeterministic() {
		double[][] first = new JacobiPCA().pca(TALL, 2);
		double[][] second = new JacobiPCA().pca(TALL, 2);
		for (int i = 0; i < first.length; i++) {
			assertArrayEquals(first[i], second[i], 0.0);
		}
	}

	@Test(expected = IllegalArgumentException.class)
	public void rejectsAnEmptyMatrix() {
		new JacobiPCA().pca(new double[0][0], 1);
	}

	@Test(expected = IllegalArgumentException.class)
	public void rejectsARaggedMatrix() {
		new JacobiPCA().pca(new double[][] { { 1.0, 2.0 }, { 3.0 } }, 1);
	}

	@Test(expected = IllegalArgumentException.class)
	public void rejectsTooManyComponents() {
		new JacobiPCA().pca(WIDE, 4);
	}
}

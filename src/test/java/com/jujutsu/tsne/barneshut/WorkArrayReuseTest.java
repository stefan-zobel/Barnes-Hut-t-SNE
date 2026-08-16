package com.jujutsu.tsne.barneshut;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;

import java.util.Random;
import java.util.function.Supplier;

import org.junit.Test;

import com.jujutsu.tsne.TSneConfiguration;
import com.jujutsu.utils.TSneUtils;

/**
 * {@link BHTSne} keeps the work arrays of its gradient in fields so that they are allocated once per
 * run rather than once per iteration. That makes the instance stateful, and the state has to survive
 * being reused for a second run of a different shape.
 * <p>
 * The parallel implementation used to allocate on {@code pos_f == null} alone, so the first run's
 * shape was kept forever: a larger second run threw an {@code ArrayIndexOutOfBoundsException}, a
 * smaller one kept the larger allocation. Every test here therefore compares a reused instance
 * against a fresh one, which is the behaviour a caller can reasonably expect and the only one that is
 * obviously right.
 * <p>
 * Every case runs against both implementations. The serial one allocated its buffers per iteration
 * until F10 and had no such state at all; now it shares the fields and the reallocation rule with the
 * parallel one, so it is exposed to exactly the same hazard and is checked for it here.
 */
public class WorkArrayReuseTest {

	private static final Supplier<BHTSne> SERIAL = new Supplier<BHTSne>() {
		@Override
		public BHTSne get() {
			return new BHTSne();
		}
	};

	private static final Supplier<BHTSne> PARALLEL = new Supplier<BHTSne>() {
		@Override
		public BHTSne get() {
			return new ParallelBHTsne();
		}
	};

	@Test
	public void aLargerSecondRunIsUnaffectedByTheFirst() {
		// the case that used to throw
		assertReuseMatchesAFreshInstance(SERIAL, 100, 2, 200, 2);
		assertReuseMatchesAFreshInstance(PARALLEL, 100, 2, 200, 2);
	}

	@Test
	public void aSecondRunWithMoreOutputDimensionsIsUnaffectedByTheFirst() {
		// the same defect along the other axis, and it used to throw as well
		assertReuseMatchesAFreshInstance(SERIAL, 100, 2, 100, 3);
		assertReuseMatchesAFreshInstance(PARALLEL, 100, 2, 100, 3);
	}

	@Test
	public void aSmallerSecondRunIsUnaffectedByTheFirst() {
		// this one was already correct - it is here so that the fix cannot break it
		assertReuseMatchesAFreshInstance(SERIAL, 200, 2, 100, 2);
		assertReuseMatchesAFreshInstance(PARALLEL, 200, 2, 100, 2);
	}

	@Test
	public void aSecondRunOfTheSameShapeReusesTheArrays() {
		// the point of holding them in fields at all: no reallocation when nothing changed
		assertSameShapeKeepsTheArrays(SERIAL);
		assertSameShapeKeepsTheArrays(PARALLEL);
	}

	@Test
	public void aSecondRunOfADifferentShapeReplacesTheArrays() {
		assertDifferentShapeReplacesTheArrays(SERIAL);
		assertDifferentShapeReplacesTheArrays(PARALLEL);
	}

	private static void assertSameShapeKeepsTheArrays(Supplier<BHTSne> implementation) {
		BHTSne tsne = implementation.get();
		tsne.tsne(config(data(100, 17), 2));
		double[] afterFirst = tsne.pos_f;

		tsne.tsne(config(data(100, 4711), 2));

		assertSame(name(tsne) + ": the same shape must not reallocate", afterFirst, tsne.pos_f);
	}

	private static void assertDifferentShapeReplacesTheArrays(Supplier<BHTSne> implementation) {
		BHTSne tsne = implementation.get();
		tsne.tsne(config(data(100, 17), 2));
		double[] afterFirst = tsne.pos_f;

		tsne.tsne(config(data(200, 4711), 2));

		assertNotSame(name(tsne) + ": a different shape must reallocate", afterFirst, tsne.pos_f);
	}

	private static void assertReuseMatchesAFreshInstance(Supplier<BHTSne> implementation, int firstN,
			int firstDims, int secondN, int secondDims) {
		double[][] first = data(firstN, 17);
		double[][] second = data(secondN, 4711);

		BHTSne reused = implementation.get();
		reused.tsne(config(first, firstDims));
		double[][] viaReuse = reused.tsne(config(second, secondDims));

		double[][] viaFresh = implementation.get().tsne(config(second, secondDims));

		for (int i = 0; i < viaFresh.length; i++) {
			assertArrayEquals(name(reused) + ": row " + i + " differs from a fresh instance",
					viaFresh[i], viaReuse[i], 0.0);
		}
	}

	private static String name(BHTSne tsne) {
		return tsne.getClass().getSimpleName();
	}

	private static TSneConfiguration config(double[][] x, int outputDims) {
		return TSneUtils.buildConfig(x, outputDims, x[0].length, 10.0, 40, false, 0.5, true, false);
	}

	private static double[][] data(int n, long seed) {
		Random random = new Random(seed);
		double[][] x = new double[n][8];
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < x[i].length; j++) x[i][j] = random.nextGaussian();
		}
		return x;
	}
}

package com.jujutsu.tsne.barneshut;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.OutputStream;
import java.io.PrintStream;
import java.util.Random;
import java.util.concurrent.atomic.AtomicReference;

import org.junit.Test;

import com.jujutsu.tsne.TSneConfiguration;
import com.jujutsu.tsne.progress.ProgressListener;
import com.jujutsu.tsne.progress.ProgressState;
import com.jujutsu.tsne.progress.TSneProgress;
import com.jujutsu.utils.TSneUtils;

/**
 * Tests for the early phase of the gradient descent - the stretch in which the P values are
 * exaggerated by a factor of 12 and the momentum is held at its low value.
 * <p>
 * The reference implementation ends that phase at a hard coded iteration 250, which silently breaks
 * every run of 250 iterations or fewer: the exaggeration is never lifted and the embedding is
 * returned in its inflated state.
 */
public class EarlyPhaseScheduleTest {

	@Test
	public void theEarlyPhaseAlwaysEndsBeforeTheRunDoes() {
		// the invariant the fix is about: whatever the iteration budget, there is at least one
		// iteration left that sees the un-exaggerated P values
		for (int maxIter = 1; maxIter <= 3000; maxIter++) {
			int end = BHTSne.earlyPhaseEnd(maxIter);
			assertTrue("maxIter " + maxIter + " ended the early phase at " + end, end < maxIter);
			assertTrue("maxIter " + maxIter + " ended the early phase at " + end, end >= 0);
		}
	}

	@Test
	public void reproducesTheReferenceScheduleAtTheReferenceLength() {
		// the C++ implementation uses 250 at its default of 1000 iterations, and anything longer
		// must keep that value rather than exaggerating proportionally longer
		assertEquals(250, BHTSne.earlyPhaseEnd(1000));
		assertEquals(250, BHTSne.earlyPhaseEnd(2000));
		assertEquals(250, BHTSne.earlyPhaseEnd(10000));
	}

	@Test
	public void keepsTheSameProportionForShorterRuns() {
		assertEquals(100, BHTSne.earlyPhaseEnd(400));
		assertEquals(50, BHTSne.earlyPhaseEnd(200));
		assertEquals(25, BHTSne.earlyPhaseEnd(100));
		assertEquals(2, BHTSne.earlyPhaseEnd(10));
	}

	/**
	 * A short run must return an embedding of the actual objective, not of the exaggerated one.
	 * <p>
	 * The cost is what this asserts on, deliberately. The obvious alternative - how cleanly the
	 * embedding separates known clusters - measures the wrong thing here: pulling clusters apart is
	 * exactly what the exaggeration does, so on well separated data a run that never switches it off
	 * scores <em>better</em> (measured: 0.96 against 0.83 at 200 iterations). The cost against the
	 * true P values has no such blind spot.
	 */
	@Test
	public void aShortRunReturnsTheEmbeddingOfTheTrueObjective() {
		double[][] data = clusteredData();

		double shortRun = finalCost(data, 200);
		double longRun = finalCost(data, 1000);

		// measured: 2.09 against 0.71 after the fix, 59.40 against 0.71 before it
		assertTrue("short run cost " + shortRun + ", long run cost " + longRun,
				shortRun < 5.0 * longRun);
	}

	private static double[][] clusteredData() {
		int clusters = 5, perCluster = 60, dims = 10;
		Random random = new Random(4711);
		double[][] centers = new double[clusters][dims];
		for (int c = 0; c < clusters; c++) {
			for (int j = 0; j < dims; j++) centers[c][j] = random.nextGaussian() * 12.0;
		}
		double[][] data = new double[clusters * perCluster][dims];
		for (int i = 0; i < data.length; i++) {
			for (int j = 0; j < dims; j++) {
				data[i][j] = centers[i % clusters][j] + random.nextGaussian();
			}
		}
		return data;
	}

	/** the cost the run reports for its last iteration */
	private static double finalCost(double[][] data, int maxIter) {
		final AtomicReference<Double> last = new AtomicReference<>(Double.NaN);
		ProgressListener listener = new ProgressListener() {
			@Override
			public void updated(ProgressState state) {
				String message = state.getMessage();
				if (message != null && message.startsWith("Err: ") && !message.endsWith("not_calculated")) {
					last.set(Double.valueOf(message.substring(5)));
				}
			}
		};
		TSneProgress.addProgressListener(listener);

		// printError requires a non silent run, whose console output is swallowed here so that the
		// test does not print a progress bar
		TSneConfiguration config =
				TSneUtils.buildConfig(data, 2, data[0].length, 15.0, maxIter, false, 0.5, false, true);
		PrintStream original = System.out;
		System.setOut(new PrintStream(discardingStream()));
		try {
			new ParallelBHTsne().tsne(config);
		} finally {
			System.setOut(original);
			TSneProgress.removeProgressListener(listener);
		}

		double cost = last.get().doubleValue();
		assertTrue("no cost was reported for maxIter " + maxIter, cost > 0.0);
		return cost;
	}

	private static OutputStream discardingStream() {
		return new OutputStream() {
			@Override
			public void write(int b) {
				// discarded
			}

			@Override
			public void write(byte[] b, int off, int len) {
				// discarded - overridden so that the default byte at a time loop is skipped
			}
		};
	}
}

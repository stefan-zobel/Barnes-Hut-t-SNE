package com.jujutsu.tsne.progress;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

import org.junit.After;
import org.junit.Test;

import com.jujutsu.tsne.TSneConfiguration;
import com.jujutsu.tsne.barneshut.BHTSne;
import com.jujutsu.tsne.barneshut.ParallelBHTsne;
import com.jujutsu.tsne.barneshut.ParallelVpTree;
import com.jujutsu.utils.TSneUtils;

/**
 * Tests for the progress notification mechanism and for its use in the Barnes-Hut t-SNE
 * implementation.
 */
public class TSneProgressTest {

    private final List<ProgressListener> registered = new ArrayList<>();

    /** Registers a listener and remembers it for removal after the test. */
    private <T extends ProgressListener> T register(final T listener) {
        TSneProgress.addProgressListener(listener);
        registered.add(listener);
        return listener;
    }

    @After
    public void removeListeners() {
        for (final ProgressListener listener : registered) {
            TSneProgress.removeProgressListener(listener);
        }
        registered.clear();
    }

    // ------------------------------------------------------------------
    // TSneProgress
    // ------------------------------------------------------------------

    @Test
    public void resetStartsANewTask() {
        TSneProgress.reset("Task A", 10);
        final ProgressState state = TSneProgress.getProgress();
        assertEquals("Task A", state.getTaskName());
        assertEquals(10, state.getTotal());
        assertEquals(0, state.getCount());
        assertNull(state.getMessage());
        assertFalse(state.isFinished());
        assertEquals(0.0, state.getFraction(), 1e-12);
    }

    @Test
    public void updateAdvancesTheCounter() {
        TSneProgress.reset("Task A", 10);
        TSneProgress.update();
        TSneProgress.update(3);
        final ProgressState state = TSneProgress.getProgress();
        assertEquals(4, state.getCount());
        assertEquals(0.4, state.getFraction(), 1e-12);
    }

    @Test
    public void updateToNeverDecreasesTheCounter() {
        TSneProgress.reset("Task A", 10);
        TSneProgress.updateTo(5);
        TSneProgress.updateTo(3);
        assertEquals(5, TSneProgress.getProgress().getCount());
        TSneProgress.updateTo(6);
        assertEquals(6, TSneProgress.getProgress().getCount());
    }

    @Test
    public void countIsClampedToTheTotal() {
        TSneProgress.reset("Task A", 10);
        TSneProgress.update(25);
        final ProgressState state = TSneProgress.getProgress();
        assertEquals(10, state.getCount());
        assertEquals(1.0, state.getFraction(), 1e-12);
        assertTrue(state.isFinished());
    }

    @Test
    public void incTotalExtendsTheTask() {
        TSneProgress.reset("Task A", 10);
        TSneProgress.update(10);
        TSneProgress.incTotal(5);
        final ProgressState state = TSneProgress.getProgress();
        assertEquals(15, state.getTotal());
        assertEquals(10, state.getCount());
        assertFalse(state.isFinished());
    }

    @Test
    public void finishedCompletesTheTask() {
        TSneProgress.reset("Task A", 10);
        TSneProgress.update(2);
        TSneProgress.finished();
        final ProgressState state = TSneProgress.getProgress();
        assertEquals(10, state.getCount());
        assertTrue(state.isFinished());
    }

    @Test
    public void messageIsPropagatedAndClearedByReset() {
        TSneProgress.reset("Task A", 10);
        TSneProgress.setMessage("Err: 1.5");
        assertEquals("Err: 1.5", TSneProgress.getProgress().getMessage());
        TSneProgress.reset("Task B", 10);
        assertNull(TSneProgress.getProgress().getMessage());
    }

    @Test
    public void unknownTotalYieldsAZeroFraction() {
        TSneProgress.reset("Task A", 0);
        TSneProgress.update(7);
        final ProgressState state = TSneProgress.getProgress();
        assertEquals(0, state.getTotal());
        assertEquals(7, state.getCount());
        assertEquals(0.0, state.getFraction(), 1e-12);
        assertFalse(state.isFinished());
    }

    // ------------------------------------------------------------------
    // Listener registration and notification
    // ------------------------------------------------------------------

    @Test
    public void listenersAreRegisteredOnlyOnceAndCanBeRemoved() {
        final AtomicInteger notifications = new AtomicInteger();
        final ProgressListener listener = state -> notifications.incrementAndGet();

        register(listener);
        TSneProgress.addProgressListener(listener); // must not register the listener twice
        TSneProgress.reset("Task A", 10);
        assertEquals(1, notifications.get());

        assertTrue(TSneProgress.removeProgressListener(listener));
        assertFalse(TSneProgress.removeProgressListener(listener));
        TSneProgress.reset("Task A", 10);
        assertEquals(1, notifications.get());
    }

    @Test
    public void resetAndFinishedAreAlwaysNotified() {
        final List<ProgressState> states = Collections.synchronizedList(new ArrayList<>());
        register(states::add);

        TSneProgress.reset("Task A", 1000);
        for (int i = 1; i <= 1000; ++i) {
            TSneProgress.updateTo(i); // most of these are swallowed by the notification throttle
        }
        TSneProgress.finished();

        assertTrue("expected at least the reset and the finished notification", states.size() >= 2);
        final ProgressState first = states.get(0);
        assertEquals("Task A", first.getTaskName());
        assertEquals(0, first.getCount());
        final ProgressState last = states.get(states.size() - 1);
        assertEquals(1000, last.getCount());
        assertTrue(last.isFinished());
    }

    @Test
    public void concurrentUpdatesAreCountedExactly() {
        final int steps = 20_000;
        final AtomicInteger decreases = new AtomicInteger();
        final AtomicInteger overflows = new AtomicInteger();
        register(new ProgressListener() {
            private int lastCount = 0;

            @Override
            public synchronized void updated(final ProgressState state) {
                if (state.getCount() < lastCount) {
                    decreases.incrementAndGet();
                }
                if (state.getCount() > state.getTotal()) {
                    overflows.incrementAndGet();
                }
                lastCount = state.getCount();
            }
        });

        TSneProgress.reset("Task A", steps);
        IntStream.range(0, steps).parallel().forEach(i -> TSneProgress.update());
        TSneProgress.finished();

        assertEquals(steps, TSneProgress.getProgress().getCount());
        assertEquals("the reported count must never decrease within a task", 0, decreases.get());
        assertEquals("the reported count must never exceed the total", 0, overflows.get());
    }

    // ------------------------------------------------------------------
    // ConsoleProgressListener
    // ------------------------------------------------------------------

    @Test
    public void consoleListenerRendersTheBarInPlace() throws UnsupportedEncodingException {
        final ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        register(new ConsoleProgressListener(new PrintStream(buffer, true, "UTF-8"), true));

        TSneProgress.reset("Task A", 10);
        TSneProgress.setMessage("Err: 1.5");
        TSneProgress.finished();

        final String rendered = buffer.toString("UTF-8");
        assertTrue("expected an in place redraw: " + rendered, rendered.contains("\r"));
        assertTrue("expected the task name: " + rendered, rendered.contains("Task A"));
        assertTrue("expected the completed bar: " + rendered, rendered.contains("100%"));
        assertTrue("expected the counts: " + rendered, rendered.contains("10/10"));
        assertTrue("expected the message: " + rendered, rendered.contains("Err: 1.5"));
        assertTrue("expected a line break after the finished task", rendered.endsWith(System.lineSeparator()));
    }

    @Test
    public void consoleListenerRendersOneCompletedBarPerTask() throws UnsupportedEncodingException {
        final ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        register(new ConsoleProgressListener(new PrintStream(buffer, true, "UTF-8"), true));

        // the same task name twice, as it happens when two t-SNE runs are performed in a row
        for (int run = 0; run < 2; ++run) {
            TSneProgress.reset("Task A", 10);
            for (int i = 1; i <= 10; ++i) {
                TSneProgress.updateTo(i);
            }
            TSneProgress.finished();
        }

        final String rendered = buffer.toString("UTF-8");
        assertEquals("each task must be completed exactly once", 2, countOccurrences(rendered, "100%"));
    }

    @Test
    public void consoleListenerWritesPlainLinesWhenNotInteractive() throws UnsupportedEncodingException {
        final ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        register(new ConsoleProgressListener(new PrintStream(buffer, true, "UTF-8"), false));

        TSneProgress.reset("Task A", 10);
        for (int i = 1; i <= 10; ++i) {
            TSneProgress.updateTo(i);
        }
        TSneProgress.finished();

        final String rendered = buffer.toString("UTF-8");
        assertFalse("must not redraw in place: " + rendered, rendered.startsWith("\r"));
        assertTrue("expected several lines: " + rendered, rendered.split("\\R").length > 1);
        assertTrue("expected the completed bar: " + rendered, rendered.contains("100%"));
        assertEquals("the completed task must be rendered once", 1, countOccurrences(rendered, "100%"));
    }

    private static int countOccurrences(final String text, final String needle) {
        int count = 0;
        for (int i = text.indexOf(needle); i >= 0; i = text.indexOf(needle, i + needle.length())) {
            ++count;
        }
        return count;
    }

    // ------------------------------------------------------------------
    // Integration with the Barnes-Hut t-SNE implementation
    // ------------------------------------------------------------------

    @Test
    public void barnesHutRunReportsBothPhases() {
        final int rows = 200;
        final int maxIter = 100;
        final Map<String, ProgressState> lastStates = new ConcurrentHashMap<>();
        final AtomicInteger decreases = new AtomicInteger();
        register(new ProgressListener() {
            @Override
            public void updated(final ProgressState state) {
                final ProgressState previous = lastStates.put(state.getTaskName(), state);
                if (previous != null && state.getCount() < previous.getCount()) {
                    decreases.incrementAndGet();
                }
            }
        });

        // silent, so that the run does not install a ConsoleProgressListener of its own
        final TSneConfiguration config = TSneUtils.buildConfig(randomData(rows, 8), 2, 8, 15.0, maxIter, true, 0.5, true, false);
        final double[][] embedding = new ParallelBHTsne().tsne(config);

        assertEquals(rows, embedding.length);
        assertEquals(2, embedding[0].length);

        final ProgressState perplexity = lastStates.get(ParallelVpTree.TASK_PERPLEXITY);
        assertNotNull("the perplexity phase must be reported", perplexity);
        assertEquals(rows, perplexity.getTotal());
        assertTrue("the perplexity phase must be completed", perplexity.isFinished());

        final ProgressState gradientDescent = lastStates.get(BHTSne.TASK_GRADIENT_DESCENT);
        assertNotNull("the gradient descent phase must be reported", gradientDescent);
        assertEquals(maxIter, gradientDescent.getTotal());
        assertTrue("the gradient descent phase must be completed", gradientDescent.isFinished());

        assertEquals("the reported count must never decrease within a phase", 0, decreases.get());
    }

    private static double[][] randomData(final int rows, final int columns) {
        final Random random = new Random(42);
        final double[][] data = new double[rows][columns];
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < columns; ++j) {
                data[i][j] = random.nextGaussian() + (i % 3) * 4.0;
            }
        }
        return data;
    }
}

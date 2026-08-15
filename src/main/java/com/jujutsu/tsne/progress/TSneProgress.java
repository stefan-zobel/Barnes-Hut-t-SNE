package com.jujutsu.tsne.progress;

import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Static facade for reporting and observing the progress of t-SNE computations.
 * <p>
 * A t-SNE run consists of several consecutive tasks (or phases), each of which is started with
 * {@link #reset(String, int)} and terminated with {@link #finished()}. Only one task is tracked at
 * a time; the task name is part of every {@link ProgressState} so that listeners can distinguish
 * the phases of a run.
 * <p>
 * Listeners can be registered by the algorithm itself (see
 * {@code com.jujutsu.tsne.barneshut.BHTSne}, which installs a {@link ConsoleProgressListener}
 * unless the configuration is silent) as well as by client code:
 *
 * <pre>{@code
 * ProgressListener listener = state -> System.out.println(state);
 * TSneProgress.addProgressListener(listener);
 * try {
 *     tsne.tsne(config);
 * } finally {
 *     TSneProgress.removeProgressListener(listener);
 * }
 * }</pre>
 *
 * All methods are thread-safe. Progress updates may be reported from many threads concurrently;
 * notifications are throttled to at most one every {@value #MIN_UPDATE_PERIOD} milliseconds, except
 * for the notifications triggered by {@link #reset(String, int)} and {@link #finished()}, which are
 * always delivered. Notifications are serialized, so a listener never sees the count of a task go
 * backwards.
 */
public final class TSneProgress {

    /** Minimum period between two throttled notifications in milliseconds. */
    private static final long MIN_UPDATE_PERIOD = 100L;

    private static final TSneProgress PROGRESS = new TSneProgress();

    private final CopyOnWriteArrayList<ProgressListener> progressListeners = new CopyOnWriteArrayList<>();
    private final Object notificationLock = new Object();
    private final AtomicInteger counter = new AtomicInteger();
    private final AtomicInteger total = new AtomicInteger();
    private final AtomicLong lastNotificationTime = new AtomicLong();
    private volatile String taskName = "";
    private volatile String message = null;

    private TSneProgress() {
    }

    /**
     * Registers a listener. Listeners are notified in registration order. Registering the same
     * listener twice has no effect.
     *
     * @param listener the listener to add, ignored if {@code null}
     */
    public static void addProgressListener(final ProgressListener listener) {
        if (listener != null) {
            PROGRESS.progressListeners.addIfAbsent(listener);
        }
    }

    /**
     * Unregisters a listener.
     *
     * @param listener the listener to remove
     * @return {@code true} if the listener was registered
     */
    public static boolean removeProgressListener(final ProgressListener listener) {
        return PROGRESS.progressListeners.remove(listener);
    }

    /**
     * Starts a new task, resetting the step counter and the message. Listeners are notified
     * unconditionally.
     *
     * @param taskName the name of the task, {@code null} is treated as the empty name
     * @param total the number of steps the task consists of
     */
    public static void reset(final String taskName, final int total) {
        PROGRESS.taskName = taskName == null ? "" : taskName;
        PROGRESS.message = null;
        PROGRESS.total.set(Math.max(0, total));
        PROGRESS.counter.set(0);
        PROGRESS.notifyListeners(true);
    }

    /**
     * Increases the number of steps of the current task, for cases where the total only becomes
     * known while the task is already running.
     *
     * @param inc the number of steps to add
     */
    public static void incTotal(final int inc) {
        PROGRESS.total.addAndGet(inc);
        PROGRESS.notifyListeners(false);
    }

    /**
     * Sets an additional message describing the current state of the task, for example the current
     * error of the gradient descent.
     *
     * @param message the message, or {@code null} to clear it
     */
    public static void setMessage(final String message) {
        PROGRESS.message = message;
        PROGRESS.notifyListeners(false);
    }

    /**
     * Completes one step of the current task.
     */
    public static void update() {
        update(1);
    }

    /**
     * Completes {@code n} steps of the current task.
     *
     * @param n the number of steps completed
     */
    public static void update(final int n) {
        PROGRESS.counter.addAndGet(n);
        PROGRESS.notifyListeners(false);
    }

    /**
     * Sets the absolute number of completed steps of the current task. The counter never decreases,
     * i.e. a {@code count} smaller than the current one is ignored.
     *
     * @param count the number of steps completed so far
     */
    public static void updateTo(final int count) {
        PROGRESS.counter.accumulateAndGet(count, Math::max);
        PROGRESS.notifyListeners(false);
    }

    /**
     * Marks the current task as completed and notifies the listeners unconditionally. This is also
     * the correct call after an aborted run: it allows listeners to release resources and to
     * terminate their output.
     */
    public static void finished() {
        PROGRESS.counter.set(PROGRESS.total.get());
        PROGRESS.notifyListeners(true);
    }

    /**
     * @return a snapshot of the current progress
     */
    public static ProgressState getProgress() {
        final int total = PROGRESS.total.get();
        int count = PROGRESS.counter.get();
        if (total > 0 && count > total) {
            count = total;
        }
        return new ProgressState(PROGRESS.taskName, PROGRESS.message, total, count);
    }

    /**
     * Notifies the registered listeners.
     *
     * @param force if {@code true} the notification is delivered regardless of the throttle
     */
    private void notifyListeners(final boolean force) {
        if (progressListeners.isEmpty()) {
            return;
        }
        final long now = System.currentTimeMillis();
        if (force) {
            lastNotificationTime.set(now);
        } else {
            // limit the notification rate; the CAS makes sure that only one of possibly many
            // concurrent updaters notifies for a given time slot
            final long last = lastNotificationTime.get();
            if (now - last < MIN_UPDATE_PERIOD || !lastNotificationTime.compareAndSet(last, now)) {
                return;
            }
        }
        // taking the snapshot under the lock guarantees that listeners never see the progress of a
        // task go backwards, even if the notifying threads are descheduled between the two steps
        synchronized (notificationLock) {
            final ProgressState state = getProgress();
            for (final ProgressListener listener : progressListeners) {
                listener.updated(state);
            }
        }
    }
}

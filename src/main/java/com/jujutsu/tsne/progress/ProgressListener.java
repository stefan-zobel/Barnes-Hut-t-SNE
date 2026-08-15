package com.jujutsu.tsne.progress;

/**
 * Receives progress notifications from long running t-SNE computations.
 * <p>
 * Listeners are registered with {@link TSneProgress#addProgressListener(ProgressListener)}, either
 * by the algorithm itself or by client code. They may be called from arbitrary threads, including
 * worker threads of the common {@link java.util.concurrent.ForkJoinPool}, and must therefore be
 * cheap and thread-safe. Implementations that update a UI have to dispatch to their own UI thread.
 */
@FunctionalInterface
public interface ProgressListener {

    /**
     * Called when the progress of the currently running task has changed.
     *
     * @param progressState an immutable snapshot of the current progress
     */
    void updated(ProgressState progressState);
}

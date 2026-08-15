package com.jujutsu.tsne.progress;

/**
 * An immutable snapshot of the progress of a single task.
 */
public final class ProgressState {

    private final String taskName;
    private final String message;
    private final int total;
    private final int count;

    ProgressState(final String taskName, final String message, final int total, final int count) {
        this.taskName = taskName;
        this.message = message;
        this.total = total;
        this.count = count;
    }

    /**
     * @return the name of the task this snapshot belongs to, never {@code null}
     */
    public String getTaskName() {
        return taskName;
    }

    /**
     * @return an additional message describing the current state, or {@code null} if there is none
     */
    public String getMessage() {
        return message;
    }

    /**
     * @return the number of steps the task consists of, or {@code 0} if the total is unknown
     */
    public int getTotal() {
        return total;
    }

    /**
     * @return the number of steps completed so far, never greater than {@link #getTotal()}
     */
    public int getCount() {
        return count;
    }

    /**
     * @return the completed fraction in {@code [0.0, 1.0]}, or {@code 0.0} if the total is unknown
     */
    public double getFraction() {
        return total <= 0 ? 0.0 : (double) count / (double) total;
    }

    /**
     * @return {@code true} if all steps of the task have been completed
     */
    public boolean isFinished() {
        return total > 0 && count >= total;
    }

    @Override
    public String toString() {
        return taskName + " " + count + "/" + total + (message == null ? "" : " " + message);
    }
}

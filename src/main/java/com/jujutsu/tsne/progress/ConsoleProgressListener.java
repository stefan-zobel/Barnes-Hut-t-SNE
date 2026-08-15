package com.jujutsu.tsne.progress;

import java.io.PrintStream;

/**
 * A {@link ProgressListener} that renders the progress as a textual progress bar, for example
 *
 * <pre>
 * Calc T-Sne  63% [========================&gt;              ] 630/1000 (0:00:12) Err: 1.234
 * </pre>
 *
 * On an interactive console the bar is redrawn in place using a carriage return, and a line break is
 * written when the task has finished. If the output is not a console (redirected output, IDE
 * console, log file), the bar is instead written as separate lines whenever the progress advances by
 * at least {@value #NON_INTERACTIVE_STEP_PERCENT} percent, so that the output stays readable.
 */
public final class ConsoleProgressListener implements ProgressListener {

    /** Progress increment in percent that triggers a line in non-interactive mode. */
    private static final int NON_INTERACTIVE_STEP_PERCENT = 5;

    private static final int BAR_WIDTH = 40;

    private final PrintStream out;
    private final boolean interactive;

    private String currentTask = null;
    private long startTime = 0L;
    private int lastLineLength = 0;
    private int lastPercent = -1;
    private int lastCount = 0;
    private boolean finishedRendered = false;

    /**
     * Creates a listener that writes to {@link System#out}, rendering in place if the JVM is
     * attached to a console.
     */
    public ConsoleProgressListener() {
        this(System.out, System.console() != null);
    }

    /**
     * @param out the stream to write to
     * @param interactive {@code true} to redraw the bar in place, {@code false} to write separate
     *            lines
     */
    public ConsoleProgressListener(final PrintStream out, final boolean interactive) {
        this.out = out;
        this.interactive = interactive;
    }

    @Override
    public synchronized void updated(final ProgressState state) {
        final long now = System.currentTimeMillis();
        // a different name or a counter that went backwards means a new task has been started
        if (!state.getTaskName().equals(currentTask) || state.getCount() < lastCount) {
            currentTask = state.getTaskName();
            startTime = now;
            lastLineLength = 0;
            lastPercent = -1;
            finishedRendered = false;
        } else if (finishedRendered) {
            // the completed task has already been rendered, don't append further lines
            return;
        }
        final int percent = (int) Math.round(100.0 * state.getFraction());
        final boolean finished = state.isFinished();
        lastCount = state.getCount();
        // in non-interactive mode the start and the end of a task are always written, everything in
        // between only if the progress has advanced far enough
        if (!interactive && !finished && lastPercent >= 0 && percent - lastPercent < NON_INTERACTIVE_STEP_PERCENT) {
            return;
        }
        finishedRendered = finished;
        lastPercent = percent;

        final String line = render(state, percent, now - startTime);
        if (interactive) {
            out.print('\r');
            out.print(line);
            // erase the remainder of a previously longer line
            for (int i = line.length(); i < lastLineLength; ++i) {
                out.print(' ');
            }
            lastLineLength = line.length();
            if (finished) {
                out.println();
                lastLineLength = 0;
            }
        } else {
            out.println(line);
        }
        out.flush();
    }

    private String render(final ProgressState state, final int percent, final long elapsedMillis) {
        final StringBuilder sb = new StringBuilder(128);
        sb.append(state.getTaskName());
        sb.append(String.format(" %3d%% [", percent));
        final int filled = (int) Math.round(BAR_WIDTH * state.getFraction());
        for (int i = 0; i < BAR_WIDTH; ++i) {
            sb.append(i < filled - 1 ? '=' : (i == filled - 1 ? (percent >= 100 ? '=' : '>') : ' '));
        }
        sb.append("] ").append(state.getCount()).append('/').append(state.getTotal());
        sb.append(" (").append(formatDuration(elapsedMillis)).append(')');
        final String message = state.getMessage();
        if (message != null && !message.isEmpty()) {
            sb.append(' ').append(message);
        }
        return sb.toString();
    }

    private static String formatDuration(final long millis) {
        final long totalSeconds = Math.max(0L, millis) / 1000L;
        return String.format("%d:%02d:%02d", totalSeconds / 3600L, (totalSeconds / 60L) % 60L, totalSeconds % 60L);
    }
}

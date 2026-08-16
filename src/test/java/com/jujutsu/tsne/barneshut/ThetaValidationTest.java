package com.jujutsu.tsne.barneshut;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.Random;

import org.junit.Test;

import com.jujutsu.tsne.TSneConfiguration;
import com.jujutsu.utils.TSneUtils;

/**
 * {@code theta == 0.0} asks for the exact O(N^2) formulation of t-SNE, which this package does not
 * implement - the exact code paths it used to contain were unreachable behind this very check and
 * carried a distance function that was missing its cross term.
 * <p>
 * The rejection is therefore the only surviving trace of that feature, and it has to keep rejecting:
 * a caller who passes 0.0 must be told, not silently given a Barnes-Hut approximation under a
 * parameter that promised exactness.
 */
public class ThetaValidationTest {

    @Test
    public void theSerialImplementationRejectsExactMode() {
        assertRejects(new BHTSne());
    }

    @Test
    public void theParallelImplementationRejectsExactMode() {
        assertRejects(new ParallelBHTsne());
    }

    @Test
    public void aPositiveThetaIsAccepted() {
        // the counterpart, so that the check above cannot pass by rejecting everything
        double[][] data = data();
        double[][] Y = new ParallelBHTsne().tsne(config(data, 0.5));

        assertTrue("expected one row per point", Y.length == data.length);
        assertTrue("expected two output dimensions", Y[0].length == 2);
    }

    private static void assertRejects(BarnesHutTSne tsne) {
        try {
            tsne.tsne(config(data(), 0.0));
            fail("theta == 0.0 was accepted");
        } catch (IllegalArgumentException expected) {
            // the message has to name the parameter, otherwise the caller cannot act on it
            assertTrue("unhelpful message: " + expected.getMessage(),
                    expected.getMessage().contains("theta"));
        }
    }

    private static TSneConfiguration config(double[][] data, double theta) {
        return TSneUtils.buildConfig(data, 2, data[0].length, 5.0, 20, false, theta, true, false);
    }

    private static double[][] data() {
        Random random = new Random(17);
        double[][] data = new double[40][4];
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                data[i][j] = random.nextGaussian();
            }
        }
        return data;
    }
}

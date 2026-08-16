package com.jujutsu.utils;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.io.File;

import org.junit.Test;

import com.jujutsu.utils.MnistData.Encoding;

/**
 * Pins down how {@code src/test/resources/datasets/mnist2500_X.txt} relates to the original MNIST
 * training set, so that a run over the full 60 000 images is the same preprocessing as a run over the
 * 2500 checked in here, and not a different data set that merely looks similar.
 * <p>
 * The relation was not documented anywhere and is not what anyone would guess. It was established by
 * measurement, in four steps:
 * <ol>
 * <li>the file holds exactly two distinct values, {@code 0.0000000e+00} and {@code 1.0000000e+00},
 * over all 1 960 000 of them, so it is thresholded;</li>
 * <li>row 0 has 618 of 784 entries set while training image 0 has 166 non-zero pixels, and
 * {@code 784 - 166 = 618} - so the <em>ones mark the background</em>, not the ink;</li>
 * <li>the count of zeros in row {@code r} equals the number of non-zero pixels of training image
 * {@code r} for all 2500 rows, at threshold 1 and at no higher one, which fixes both the pairing by
 * index and the threshold: any non-zero pixel counts as ink;</li>
 * <li>the positions still did not line up, so of the eight combinations of the four orientations with
 * the two polarities exactly one reproduces the file: the image is <em>transposed</em>.</li>
 * </ol>
 * So {@code file[r][row * 28 + col]} is {@code 0.0} where pixel {@code (col, row)} - the other way
 * round - of training image {@code r} is non-zero, and {@code 1.0} where it is zero. That is
 * {@link Encoding#LEGACY}, and the first test below is what keeps this description honest.
 */
public class MnistLegacyEncodingTest {

    private static final File SUBSET = new File("src/test/resources/datasets/mnist2500_X.txt");
    private static final int SUBSET_ROWS = 2500;

    @Test
    public void theLegacyEncodingReproducesTheCheckedInSubset() {
        double[][] expected = MatrixUtils.simpleRead2DMatrix(SUBSET, "   ");
        assertEquals("rows of the checked in subset", SUBSET_ROWS, expected.length);
        assertEquals("columns of the checked in subset", MnistData.PIXELS, expected[0].length);

        double[][] actual = MnistData.load(SUBSET_ROWS, Encoding.LEGACY);

        for (int r = 0; r < SUBSET_ROWS; r++) {
            assertArrayEquals("row " + r, expected[r], actual[r], 0.0);
        }
    }

    @Test
    public void theLegacyEncodingIsNotSimplyTheBinaryOne() {
        // both are 0.0 and 1.0 only, which makes it tempting to fold them together. They differ by a
        // transposition and by which of the two values means ink, so this is here to say no.
        double[][] legacy = MnistData.load(1, Encoding.LEGACY);
        double[][] binary = MnistData.load(1, Encoding.BINARY);

        boolean identical = true;
        for (int i = 0; i < MnistData.PIXELS && identical; i++) {
            identical = legacy[0][i] == binary[0][i];
        }
        assertFalse("LEGACY and BINARY must not agree", identical);
    }

    @Test
    public void binaryMarksExactlyTheNonZeroPixels() {
        double[][] gray = MnistData.load(64, Encoding.GRAY);
        double[][] binary = MnistData.load(64, Encoding.BINARY);

        for (int r = 0; r < gray.length; r++) {
            for (int i = 0; i < MnistData.PIXELS; i++) {
                assertTrue("grey value out of range at " + r + "/" + i,
                        gray[r][i] >= 0.0 && gray[r][i] <= 1.0);
                assertEquals("pixel " + r + "/" + i, gray[r][i] > 0.0 ? 1.0 : 0.0, binary[r][i], 0.0);
            }
        }
    }

    @Test
    public void theFirstLabelsAreTheOnesMnistIsKnownFor() {
        assertArrayEquals(new int[] { 5, 0, 4, 1, 9, 2, 1, 3, 1, 4 }, MnistData.labels(10));
    }

    @Test
    public void countsOutsideTheDataSetAreRejected() {
        for (int n : new int[] { 0, -1, MnistData.SIZE + 1 }) {
            try {
                MnistData.load(n);
                org.junit.Assert.fail("n = " + n + " should have been rejected");
            } catch (IllegalArgumentException expected) {
                // the message names the bound
                assertTrue(expected.getMessage().contains(String.valueOf(MnistData.SIZE)));
            }
        }
    }
}

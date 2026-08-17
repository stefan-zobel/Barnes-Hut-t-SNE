package com.jujutsu.utils;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.zip.GZIPInputStream;

/**
 * The full MNIST training set, 60 000 images of 28 x 28, for measurements that need more than the
 * 2500 rows of {@code mnist2500_X.txt}.
 * <p>
 * The data sits next to that file, in the original IDX form as it is distributed, still compressed.
 * That is 9.9 MB for all 60 000 images against roughly 750 MB for the same rows in the format of
 * {@code mnist2500_X.txt}, and reading it skips parsing 47 million decimal numbers, which on its own
 * takes longer than a t-SNE prologue. Reading all of it costs 0.42 s.
 * <p>
 * <b>Memory.</b> {@code load(60000)} returns a {@code double[60000][784]}, which is 376 MB, and
 * {@code BHTSne} flattens it into another 376 MB before it starts. Plan for {@code -Xmx8g}.
 */
public final class MnistData {

    /** How a pixel becomes a coordinate. */
    public enum Encoding {
        /** {@code pixel / 255.0}, so 0.0 is background and 1.0 is full ink. */
        GRAY,
        /** 1.0 where the pixel is not zero, 0.0 where it is. */
        BINARY,
        /**
         * What {@code mnist2500_X.txt} contains, so that the large data set is the same
         * preprocessing as the small one. See the class comment of {@link MnistLegacyEncodingTest}
         * for how this was established - it is not what anyone would guess.
         */
        LEGACY
    }

    public static final int SIZE = 60000;
    public static final int PIXELS = 784;

    private static final int EDGE = 28;
    private static final File DIRECTORY = new File("src/test/resources/datasets");
    private static final String IMAGE_FILE = "train-images-idx3-ubyte.gz";
    private static final String LABEL_FILE = "train-labels-idx1-ubyte.gz";
    private static final int IMAGE_MAGIC = 2051;
    private static final int LABEL_MAGIC = 2049;

    private MnistData() {
    }

    /**
     * The first {@code n} images, grey scaled to {@code [0, 1]}.
     *
     * @param n how many images to read, at most {@link #SIZE}
     * @return {@code n x 784}, row major within the image
     */
    public static double[][] load(int n) {
        return load(n, Encoding.GRAY);
    }

    /**
     * The first {@code n} images.
     *
     * @param n how many images to read, at most {@link #SIZE}
     * @param encoding how a pixel becomes a coordinate
     * @return {@code n x 784}
     */
    public static double[][] load(int n, Encoding encoding) {
        checkCount(n);
        byte[] pixels = readPayload(IMAGE_FILE, IMAGE_MAGIC, n, PIXELS);
        double[][] images = new double[n][PIXELS];
        for (int image = 0; image < n; image++) {
            int base = image * PIXELS;
            double[] out = images[image];
            for (int i = 0; i < PIXELS; i++) {
                out[i] = encode(pixels, base, i, encoding);
            }
        }
        return images;
    }

    /**
     * The labels of the first {@code n} images.
     *
     * @param n how many labels to read, at most {@link #SIZE}
     * @return the digit each image shows, 0 to 9
     */
    public static int[] labels(int n) {
        checkCount(n);
        byte[] raw = readPayload(LABEL_FILE, LABEL_MAGIC, n, 1);
        int[] labels = new int[n];
        for (int i = 0; i < n; i++) {
            labels[i] = raw[i] & 0xff;
        }
        return labels;
    }

    private static double encode(byte[] pixels, int base, int i, Encoding encoding) {
        switch (encoding) {
            case GRAY:
                return (pixels[base + i] & 0xff) / 255.0;
            case BINARY:
                return (pixels[base + i] & 0xff) != 0 ? 1.0 : 0.0;
            default:
                // transposed within the image, and 1.0 marks the background rather than the ink
                int transposed = (i % EDGE) * EDGE + (i / EDGE);
                return (pixels[base + transposed] & 0xff) != 0 ? 0.0 : 1.0;
        }
    }

    private static void checkCount(int n) {
        if (n < 1 || n > SIZE) {
            throw new IllegalArgumentException("n must be between 1 and " + SIZE + ": " + n);
        }
    }

    /** Reads the first {@code count} records of {@code recordSize} bytes each, past the header. */
    private static byte[] readPayload(String name, int magic, int count, int recordSize) {
        File file = new File(DIRECTORY, name);
        try {
            DataInputStream in = new DataInputStream(new GZIPInputStream(
                    new BufferedInputStream(new FileInputStream(file), 1 << 16), 1 << 16));
            try {
                int actualMagic = in.readInt();
                if (actualMagic != magic) {
                    throw new IOException("expected magic " + magic + ", found " + actualMagic);
                }
                int available = in.readInt();
                if (available < count) {
                    throw new IOException("holds only " + available + " records");
                }
                // the image file states its two dimensions here, the label file states nothing more
                if (magic == IMAGE_MAGIC) {
                    int rows = in.readInt();
                    int cols = in.readInt();
                    if (rows * cols != recordSize) {
                        throw new IOException("images are " + rows + " x " + cols);
                    }
                }
                byte[] payload = new byte[count * recordSize];
                in.readFully(payload);
                return payload;
            } finally {
                in.close();
            }
        } catch (IOException e) {
            throw new IllegalStateException("cannot read " + file.getAbsolutePath(), e);
        }
    }
}

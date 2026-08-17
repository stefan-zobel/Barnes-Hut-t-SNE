package com.jujutsu.tsne;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.util.Random;

import org.junit.Test;

import com.jujutsu.utils.MatrixOps;
import com.jujutsu.utils.MatrixUtils;

import math.linalg.JacobiPCA;

public class MatrixOpTest {

    @Test
    public void thePcaProjectionIsCenteredAndOrderedByVariance() {
        // This used to draw from MatrixOps.rnorm, which is seeded from nothing, print both the input
        // and the projection, and assert not one thing - it could only fail by throwing. The input is
        // deterministic now and the two properties every PCA projection has are checked. What
        // JacobiPCA computes is pinned in JacobiPCATest; this is the call from here.
        double [][] matrix = twoLatentDirections(200, 7);

        double [][] projected = new JacobiPCA().pca(matrix, 2);

        assertEquals("rows", 200, projected.length);
        assertEquals("kept components", 2, projected[0].length);
        assertEquals("component 0 is centered", 0.0, mean(projected, 0), 1e-9);
        assertEquals("component 1 is centered", 0.0, mean(projected, 1), 1e-9);
        assertTrue("components must come out in descending order of variance",
                variance(projected, 0) > variance(projected, 1));
    }

    /** Rows spanned by one strong and one weak direction plus noise, so the order of the first two
     *  components is not a matter of chance. */
    private static double [][] twoLatentDirections(int rows, int cols) {
        Random random = new Random(20260816L);
        double [][] x = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            double strong = 10.0 * random.nextGaussian();
            double weak = 2.0 * random.nextGaussian();
            for (int j = 0; j < cols; j++) {
                x[i][j] = (j % 2 == 0 ? strong : weak) + 0.1 * random.nextGaussian();
            }
        }
        return x;
    }

    private static double mean(double [][] m, int column) {
        double sum = 0.0;
        for (double[] row : m) sum += row[column];
        return sum / m.length;
    }

    private static double variance(double [][] m, int column) {
        double mean = mean(m, column);
        double sum = 0.0;
        for (double[] row : m) {
            double d = row[column] - mean;
            sum += d * d;
        }
        return sum / (m.length - 1);
    }

    /*
    @Test
    public void testTimeManyTransposes() {
        MatrixOps mo = new MatrixOps();
        int rows = 15302;
        int cols = 1143;
        double [][] matrix = new double [rows][cols];
        double [][] trmatrix = new double [cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = (double) j + (i*cols);
                trmatrix[j][i] = (double) j + (i*cols);
            }
        }
        int noLaps = 10;
        long trtime = 0;
        long partrtime = 0;
        long time = 0;
        for (int laps = 0; laps < noLaps; laps++) {
            time = System.currentTimeMillis();
            double [][] tr1 = MatrixOps.transposeSerial(matrix);
            trtime += (System.currentTimeMillis()-time);
            assertEquals(tr1.length,cols);
            assertEquals(tr1[0].length,rows);
            time = System.currentTimeMillis();
            double [][] tr2 = mo.transpose(matrix,20);
            partrtime += (System.currentTimeMillis()-time);
            assertEquals(tr2.length,cols);
            assertEquals(tr2[0].length,rows);
            for (int i = 0; i < tr1.length; i++) {
                for (int j = 0; j < tr1[0].length; j++) {
                    assertEquals(trmatrix[i][j],tr1[i][j],0.0000001);
                    assertEquals("I: " + i + " J:" + j, trmatrix[i][j],tr2[i][j],0.0000001);
                }
            }
        }
        System.out.println("    Tr time: " + trtime);
        System.out.println("Par Tr time: " + partrtime);
    }
    */

    /*
    @Test
    public void timeTransposesNist() {
        MatrixOps mo = new MatrixOps();
        double [][] matrix = MatrixUtils.simpleRead2DMatrix(new File("src/test/resources/datasets/mnist2500_X.txt"), "   ");
        int rows = matrix.length;
        int cols = matrix[0].length;
        double [][] trmatrix = new double [cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                trmatrix[j][i] = matrix[i][j];
            }
        }
        int noLaps = 300;
        long trtime = 0;
        long partrtime = 0;
        long time = 0;
        System.out.println("Size is: " + rows + " x" + cols + "...");
        for (int laps = 0; laps < noLaps; laps++) {
            if((laps%100)==0) System.out.println("Iter " + laps + "...");
            time = System.currentTimeMillis();
            double [][] tr1 = MatrixOps.transposeSerial(matrix);
            trtime += (System.currentTimeMillis()-time);
            assertEquals(tr1.length,cols);
            assertEquals(tr1[0].length,rows);
            time = System.currentTimeMillis();
            double [][] tr2 = mo.transpose(matrix,20);
            partrtime += (System.currentTimeMillis()-time);
            assertEquals(tr2.length,cols);
            assertEquals(tr2[0].length,rows);
            for (int i = 0; i < tr1.length; i++) {
                for (int j = 0; j < tr1[0].length; j++) {
                    assertEquals(trmatrix[i][j],tr1[i][j],0.0000001);
                    assertEquals("I: " + i + " J:" + j, trmatrix[i][j],tr2[i][j],0.0000001);
                }
            }
        }
        System.out.println("    Tr time: " + trtime);
        System.out.println("Par Tr time: " + partrtime);
    }
    */

    /*
    @Test
    public void timeScalarMultNist() {
        MatrixOps mo = new MatrixOps();
        double [][] matrix1 = MatrixUtils.simpleRead2DMatrix(new File("src/test/resources/datasets/mnist2500_X.txt"), "   ");
        double [][] matrix2 = MatrixUtils.simpleRead2DMatrix(new File("src/test/resources/datasets/mnist2500_X.txt"), "   ");
        int rows = matrix1.length;
        int cols = matrix1[0].length;
        int noLaps = 300;
        long trtime = 0;
        long partrtime = 0;
        long time = 0;
        System.out.println("Size is " + rows + " x " + cols + "...");
        for (int laps = 0; laps < noLaps; laps++) {
            if((laps%100)==0) System.out.println("Iter " + laps + "...");
            time = System.currentTimeMillis();
            double [][] tr1 = mo.scalarMultiply(matrix1, matrix2);
            trtime += (System.currentTimeMillis()-time);
            time = System.currentTimeMillis();
            double [][] tr2 = mo.parScalarMultiply(matrix1, matrix2);
            partrtime += (System.currentTimeMillis()-time);
            for (int i = 0; i < tr1.length; i++) {
                for (int j = 0; j < tr1[0].length; j++) {
                    assertEquals("I: " + i + " J:" + j, tr1[i][j],tr2[i][j],0.0000001);
                }
            }
        }
        System.out.println("    Tr time: " + trtime);
        System.out.println("Par Tr time: " + partrtime);
    }
    */

    @Test
    public void testExtractRowFromFlatFirst() {
        double [] flatMatrix = {1,2,3,4,5,6,7,8,9,0};
        int dimension = 2;
        int rowIdx = 0;
        double [] row = MatrixOps.extractRowFromFlatMatrix(flatMatrix, rowIdx, dimension);
        double [] expected = {1, 2};
        assertArrayEquals(expected, row, 0.000000001);
    }

    @Test
    public void testExtractRowFromFlatMidRange() {
        double [] flatMatrix = {1,2,3,4,5,6,7,8,9,0};
        int dimension = 2;
        int rowIdx = 2;
        double [] row = MatrixOps.extractRowFromFlatMatrix(flatMatrix, rowIdx, dimension);
        double [] expected = {5, 6};
        assertArrayEquals(expected, row, 0.000000001);
    }

    @Test
    public void testExtractRowFromFlatLast() {
        double [] flatMatrix = {1,2,3,4,5,6,7,8,9,0};
        int dimension = 2;
        int rowIdx = 4;
        double [] row = MatrixOps.extractRowFromFlatMatrix(flatMatrix, rowIdx, dimension);
        double [] expected = {9, 0};
        assertArrayEquals(expected, row, 0.000000001);
    }
}

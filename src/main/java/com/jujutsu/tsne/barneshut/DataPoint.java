package com.jujutsu.tsne.barneshut;

import static java.lang.Math.min;
import static java.lang.Math.sqrt;

/**
 * One point of a data set, as the vantage point tree sees it.
 * <p>
 * A point holds its own {@code D} coordinates and reads them from index 0. It never writes to them.
 * <p>
 * It was a <em>view</em> into the flat {@code N x D} input matrix for a while, which saved copying
 * every row - the original copied each one twice, 63 MB of transient allocation for 5000 points of 784
 * dimensions. That turned out to cost more than it saved: the kNN search reads these coordinates
 * billions of times, and reading them out of one shared array through an offset made the search about
 * 1.25x slower at 20 000 points, half of it from the shared array and a third from the offset in the
 * index. The row is copied once now, so the allocation is still half of what the original did and the
 * search reads a short array from index 0.
 */
public class DataPoint {

    int _ind;
    /** this point's own coordinates, {@code _D} of them, starting at index 0 */
    double [] _x;
    int _D;

    public DataPoint() {
        _D = 1;
        _ind = -1;
    }

    /**
     * A point over a standalone row.
     *
     * @param D the dimensionality
     * @param ind index of the point in its data set
     * @param x the coordinates, of which the first {@code D} are copied
     */
    public DataPoint(int D, int ind, double [] x) {
        this(x, 0, D, ind);
    }

    /**
     * A point over one row of a flat {@code N x D} matrix. The row is copied, so the matrix may be
     * changed or discarded afterwards without affecting the point.
     *
     * @param x the flat matrix, at least {@code offset + D} elements
     * @param offset index of this point's first coordinate in {@code x}
     * @param D the dimensionality
     * @param ind index of the point in its data set
     */
    public DataPoint(double [] x, int offset, int D, int ind) {
        _x = new double[D];
        System.arraycopy(x, offset, _x, 0, D);
        _D = D;
        _ind = ind;
    }

    @Override
    public String toString() {
        String xStr = "";
        for (int i = 0; i < min(20,_D); i++) {
            xStr += _x[i] + ", ";
        }
        return "DataPoint (index=" + _ind+ ", Dim=" + _D + ", point=" + xStr + ")";
    }

    public int index() { return _ind; }
    int dimensionality() { return _D; }
    double x(int d) { return _x[d]; }

    public double euclidean_distance( DataPoint t1 ) {
        return sqrt(EuclideanDistance.squaredDistance(t1._x, _x, t1._D));
    }

    public static double euclidean_distance( DataPoint t1, DataPoint t2 ) {
        return sqrt(EuclideanDistance.squaredDistance(t1._x, t2._x, t1._D));
    }
}

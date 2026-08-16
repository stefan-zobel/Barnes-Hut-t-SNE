package com.jujutsu.tsne.barneshut;

import static java.lang.Math.min;
import static java.lang.Math.sqrt;

/**
 * One point of a data set, as the vantage point tree sees it.
 * <p>
 * A point is a <em>view</em>: it holds the array its coordinates live in together with the index at
 * which they start, and reads them from there. It neither copies them nor ever writes to them. The
 * ball tree is built over the flat {@code N x D} input matrix, so all {@code N} points share that one
 * array and not a single row is copied. Constructing them used to copy every row twice, once out of
 * the flat matrix and once again in the constructor here, which came to 63 MB of transient allocation
 * for 5000 points of 784 dimensions.
 * <p>
 * The flip side of sharing is that one surviving point keeps the whole matrix reachable, where it
 * used to keep only its own row alive. That costs nothing here - the tree holds all {@code N} points
 * anyway - but a caller that keeps a single point out of a large data set should know it.
 */
public class DataPoint {

    int _ind;
    /** the array the coordinates live in, shared with the other points of the same data set */
    double [] _x;
    /** index of this point's first coordinate in {@link #_x} */
    int _offset;
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
     * @param x the coordinates. They are <em>not</em> copied: the point reads them from this array for
     *            as long as it lives, so writing to it afterwards changes the point.
     */
    public DataPoint(int D, int ind, double [] x) {
        this(x, 0, D, ind);
    }

    /**
     * A point over one row of a flat {@code N x D} matrix.
     *
     * @param x the flat matrix, not copied
     * @param offset index of this point's first coordinate in {@code x}
     * @param D the dimensionality
     * @param ind index of the point in its data set
     */
    public DataPoint(double [] x, int offset, int D, int ind) {
        _x = x;
        _offset = offset;
        _D = D;
        _ind = ind;
    }

    @Override
    public String toString() {
        String xStr = "";
        for (int i = 0; i < min(20,_D); i++) {
            xStr += _x[_offset + i] + ", ";
        }
        return "DataPoint (index=" + _ind+ ", Dim=" + _D + ", point=" + xStr + ")";
    }

    public int index() { return _ind; }
    int dimensionality() { return _D; }
    double x(int d) { return _x[_offset + d]; }

    public double euclidean_distance( DataPoint t1 ) {
        return sqrt(EuclideanDistance.squaredDistance(t1._x, t1._offset, _x, _offset, t1._D));
    }

    public static double euclidean_distance( DataPoint t1, DataPoint t2 ) {
        return sqrt(EuclideanDistance.squaredDistance(t1._x, t1._offset, t2._x, t2._offset, t1._D));
    }
}

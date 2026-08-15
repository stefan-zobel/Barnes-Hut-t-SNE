package com.jujutsu.tsne.barneshut;

import static java.lang.Math.min;
import static java.lang.Math.sqrt;

public class DataPoint {
	
	int _ind;
	double [] _x;
	int _D;
	
	public DataPoint() {
        _D = 1;
        _ind = -1;
    }

	public DataPoint(int D, int ind, double [] x) {
		_D = D;
		_ind = ind;
		_x = x.clone();
	}
	
	@Override
	public String toString() {
		String xStr = "";
		for (int i = 0; i < min(20,_x.length); i++) {
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

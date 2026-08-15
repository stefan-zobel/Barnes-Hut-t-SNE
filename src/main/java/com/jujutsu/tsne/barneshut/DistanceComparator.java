package com.jujutsu.tsne.barneshut;

import java.util.Comparator;

public class DistanceComparator implements Comparator<DataPoint> {
	DataPoint refItem; 
	Distance dist;
	
	DistanceComparator(DataPoint refItem) {
		this.refItem = refItem;
		this.dist = new EuclideanDistance();
	}
	
	DistanceComparator(DataPoint refItem, Distance dist) {
		this.refItem = refItem;
		this.dist = dist;
	}

	@Override
	public int compare(DataPoint o1, DataPoint o2) {
		// evaluate each distance exactly once, the ternary chain used to compute up to four of them
		double d1 = dist.distance(o1, refItem);
		double d2 = dist.distance(o2, refItem);
		return d1 < d2 ? -1 : (d1 > d2 ? 1 : 0);
	}
}
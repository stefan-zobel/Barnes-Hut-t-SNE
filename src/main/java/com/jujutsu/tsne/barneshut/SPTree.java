package com.jujutsu.tsne.barneshut;

import static java.lang.Math.max;

import com.jujutsu.utils.MatrixOps;

public class SPTree {

	  // Fixed constants
    final static int QT_NODE_CAPACITY = 1;
        
	protected SPTree parent;
	protected int dimension;
	protected boolean is_leaf;
	protected int size;
	protected int cum_size;
	
	 // Axis-aligned bounding box stored as a center with half-dimensions to represent the boundaries of this quad tree
    Cell boundary;
    
    // Indices in this space-partitioning tree node, corresponding center-of-mass, and list of all children
    double[] data;
    double[] center_of_mass;
    int [] index = new int[QT_NODE_CAPACITY];
    
    // Children, only allocated once this node has been subdivided
    SPTree [] children;
    int no_children;
    double max_width_sq;

	public SPTree(int D, double[] inp_data, int N) {
		// Compute mean, width, and height of current map (boundaries of SPTree)
		int nD = 0;
		double [] mean_Y = new double [D];
		double []  min_Y = new double [D]; 
		double []  max_Y = new double [D]; 
		for(int d = 0; d < D; d++)  {
			min_Y[d] = Double.POSITIVE_INFINITY;
			max_Y[d] = Double.NEGATIVE_INFINITY;
		}
		for( int n = 0; n < N; n++) {
			for( int d = 0; d < D; d++) {
				int idx = n * D + d;
				mean_Y[d] += inp_data[idx];
				if(inp_data[nD + d] < min_Y[d]) min_Y[d] = inp_data[nD + d];
				if(inp_data[nD + d] > max_Y[d]) max_Y[d] = inp_data[nD + d];
			}
			nD += D;
		}
		for(int d = 0; d < D; d++) mean_Y[d] /= (double) N;

		// Construct SPTree
		double [] width = new double [D];
		for(int d = 0; d < D; d++) width[d] = max(max_Y[d] - mean_Y[d], mean_Y[d] - min_Y[d]) + 1e-5;
		init(null, D, inp_data, mean_Y, width);
		fill(N);
	}

	// Main initialization function
	void init(SPTree inp_parent, int D, double [] inp_data, double [] inp_corner, double [] inp_width)
	{
		parent = inp_parent;
		dimension = D;
		no_children = 2;
		for(int d = 1; d < D; d++) no_children *= 2;
		data = inp_data;
		is_leaf = true;
		size = 0;
		cum_size = 0;

		center_of_mass = new double[D];
		boundary = new Cell(dimension);
		double max_width = 0.0;
		for(int d = 0; d < D; d++) {
			boundary.setCorner(d, inp_corner[d]);
			boundary.setWidth( d, inp_width[d]);
			center_of_mass[d] = .0;
			max_width = (max_width > inp_width[d]) ? max_width : inp_width[d];
		}
		max_width_sq = max_width * max_width;

		// the children are only allocated when this node is subdivided, leaf nodes do not need them
	}
	
	// Constructor for SPTree with particular size and parent -- build the tree, too!
	SPTree(int D, double [] inp_data, int N, double [] inp_corner, double [] inp_width)
	{
		init(null, D, inp_data, inp_corner, inp_width);
		fill(N);
	}


	// Constructor for SPTree with particular size (do not fill the tree)
	SPTree(int D, double [] inp_data, double [] inp_corner, double [] inp_width)
	{
		init(null, D, inp_data, inp_corner, inp_width);
	}


	// Constructor for SPTree with particular size and parent (do not fill tree)
	SPTree(SPTree inp_parent, int D, double [] inp_data, double [] inp_corner, double [] inp_width) {
		init(inp_parent, D, inp_data, inp_corner, inp_width);
	}


	// Constructor for SPTree with particular size and parent -- build the tree, too!
	SPTree(SPTree inp_parent, int D, double [] inp_data, int N, double [] inp_corner, double [] inp_width)
	{
		init(inp_parent, D, inp_data, inp_corner, inp_width);
		fill(N);
	}

	// Update the data underlying this tree
	void setData(double [] inp_data)
	{
		data = inp_data;
	}


	// Get the parent of the current tree
	SPTree getParent()
	{
		return parent;
	}

	SPTree[] getTreeArray(int no_children) {
		return new SPTree[no_children];
	}

	// Insert a point into the SPTree
	boolean insert(int new_index)
	{
		// Read the point in place, an offset into the flat data array is all we need
		final int offset = new_index * dimension;

		// Ignore objects which do not belong in this quad tree
		if(!boundary.containsPoint(data, offset))
			return false;

		// Online update of cumulative size and center-of-mass
		cum_size++;
		double mult1 = (double) (cum_size - 1) / (double) cum_size;
		double mult2 = 1.0 / (double) cum_size;
		for(int d = 0; d < dimension; d++) {
			center_of_mass[d] *= mult1;
			center_of_mass[d] += mult2 * data[offset + d];
		}

		// If there is space in this quad tree and it is a leaf, add the object here
		if(is_leaf && size < QT_NODE_CAPACITY) {
			index[size] = new_index;
			size++;
			return true;
		}

		// Don't add duplicates for now (this is not very nice)
		boolean any_duplicate = false;
		for(int n = 0; n < size; n++) {
			boolean duplicate = true;
			int other = index[n] * dimension;
			for(int d = 0; d < dimension; d++) {
				if(data[offset + d] != data[other + d]) { duplicate = false; break; }
			}
			any_duplicate = any_duplicate || duplicate;
		}
		if(any_duplicate) return true;

		// Otherwise, we need to subdivide the current cell
		if(is_leaf) subdivide();

		// The child containing the point follows directly from the position relative to the corner,
		// so there is no need to probe all of the children
		return children[childIndex(offset)].insert(new_index);
	}

	// Index of the child cell that contains the point starting at the given offset. Bit d of the
	// child index is set if the point lies in the lower half of dimension d, which is exactly the
	// layout subdivide() creates.
	int childIndex(int offset) {
		int child = 0;
		for(int d = 0; d < dimension; d++) {
			if(data[offset + d] < boundary.getCorner(d)) child |= (1 << d);
		}
		return child;
	}

	// Create four children which fully divide this cell into four quads of equal area
	void subdivide() {

		// Create new children
		children = getTreeArray(no_children);
		double [] new_corner = new double[dimension];
		double [] new_width  = new double[dimension];
		for(int i = 0; i < no_children; i++) {
			int div = 1;
			for(int d = 0; d < dimension; d++) {
				new_width[d] = .5 * boundary.getWidth(d);
				if((i / div) % 2 == 1) new_corner[d] = boundary.getCorner(d) - .5 * boundary.getWidth(d);
				else                   new_corner[d] = boundary.getCorner(d) + .5 * boundary.getWidth(d);
				div *= 2;
			}
			children[i] = getNewTree(this, new_corner, new_width);
		}

		// Move existing points to correct children
		for(int i = 0; i < size; i++) {
			children[childIndex(index[i] * dimension)].insert(index[i]);
			index[i] = -1;
		}


		// Empty parent node
		size = 0;
		is_leaf = false;
	}

	SPTree getNewTree(SPTree root, double[] new_corner, double[] new_width) {
		return new SPTree(root, dimension, data, new_corner, new_width);
	}

	// Build SPTree on dataset
	void fill(int N)
	{
		for(int i = 0; i < N; i++) { insert(i); }
	}


	// Checks whether the specified tree is correct
	boolean isCorrect()
	{
		for(int n = 0; n < size; n++) {
			double [] point = MatrixOps.extractRowFromFlatMatrix(data, index[n], dimension);
			if(!boundary.containsPoint(point)) return false;
		}
		if(!is_leaf) {
			boolean correct = true;
			for(int i = 0; i < no_children; i++) correct = correct && children[i].isCorrect();
			return correct;
		}
		else return true;
	}



	// Build a list of all indices in SPTree
	void getAllIndices(int [] indices)
	{
		getAllIndices(indices, 0);
	}


	// Build a list of all indices in SPTree
	int getAllIndices(int [] indices, int loc)
	{

		// Gather indices in current quadrant
		for(int i = 0; i < size; i++) indices[loc + i] = index[i];
		loc += size;

		// Gather indices in children
		if(!is_leaf) {
			for(int i = 0; i < no_children; i++) loc = children[i].getAllIndices(indices, loc);
		}
		return loc;
	}


	int getDepth() {
		if(is_leaf) return 1;
		int depth = 0;
		for(int i = 0; i < no_children; i++) depth = max(depth, children[i].getDepth());
		return 1 + depth;
	}
	
	// Compute non-edge forces using Barnes-Hut algorithm
    double computeNonEdgeForces(int point_index, double theta, double[] neg_f, 
        double buff[], double[] sum_Q)
    {
        // Make sure that we spend no time on empty nodes or self-interactions
        if (cum_size == 0 || (is_leaf && size == 1 && index[0] == point_index))
            return 0.0;

        // Compute distance between point and center-of-mass
        double D = .0;
        int ind = point_index * dimension;
        // Check whether we can use this node as a "summary"
        for (int d = 0; d < dimension; d++)
        {
            buff[d] = data[ind + d] - center_of_mass[d];
            D += buff[d] * buff[d];
        }

        // avoid sqrt in this function since it is used ALOT and is more
        // computationally demanding than multiplication
        if (is_leaf || max_width_sq < theta * theta * D)
        {
            // Compute and add t-SNE force between point and current node
            D = 1.0 / (1.0 + D);
            double mult = cum_size * D;
            sum_Q[point_index] += mult;
            mult *= D;
            for (int d = 0; d < dimension; d++)
                neg_f[d] += mult * buff[d];
        }
        else
        {

            // Recursively apply Barnes-Hut to children
            for (int i = 0; i < no_children; i++)
                children[i].computeNonEdgeForces(point_index, theta, neg_f,
                    buff, sum_Q);
        }
        return sum_Q[point_index];
    }



	// Computes edge forces
	void computeEdgeForces(int [] row_P, int [] col_P, double [] val_P, int N, double [] pos_f)
	{
		// Loop over all edges in the graph
		double [] buff = new double[dimension];
		int ind1 = 0;
		int ind2 = 0;
		double D;
		for(int n = 0; n < N; n++) {
			for(int i = row_P[n]; i < row_P[n + 1]; i++) {

				// Compute pairwise distance and Q-value
				D = 1.0;
				ind2 = col_P[i] * dimension;
				for(int d = 0; d < dimension; d++) { 
					buff[d] = data[ind1 + d] - data[ind2 + d];
					D += buff[d] * buff[d];
				} 
				D = val_P[i] / D;

				// Sum positive force
				for(int d = 0; d < dimension; d++) pos_f[ind1 + d] += D * buff[d];
			}
			ind1 += dimension;
		}
	}

	// Print out tree
	void print() 
	{
		print(0);
	}	

	// Print out tree
	void print(int lvl) 
	{
		String prefix = "";
		for(int i = 0; i < lvl; i++) {
			prefix += "    ";
		}

		if(cum_size == 0) {
			System.out.printf(prefix + "Empty node\n");
			return;
		}

		if(is_leaf) {
			System.out.printf(prefix + "Leaf node; data = [");
			for(int i = 0; i < size; i++) {
				double [] point = MatrixOps.extractRowFromFlatMatrix(data, index[i], dimension);
				for(int d = 0; d < dimension; d++) System.out.printf("%f, ", point[d]);
				System.out.printf(" (index = %d)", index[i]);
				if(i < size - 1) System.out.printf("\n");
				else System.out.printf("]\n");
			}        
		}
		else {
			System.out.printf(prefix + "Intersection node with center-of-mass = [");
			for(int d = 0; d < dimension; d++) System.out.printf("%f, ", center_of_mass[d]);
			System.out.print(" +- ");
			for(int d = 0; d < dimension; d++) System.out.printf("%f, ", boundary.width[d]);
			System.out.printf("]; children are:\n");
			for(int i = 0; i < no_children; i++) children[i].print(lvl+1);
		}
	}
	
	class Cell {		
		int dimension;
		double [] corner;
		double [] width;
		    
		// Constructs cell
		Cell(int inp_dimension) {
			dimension = inp_dimension;
			corner = new double[dimension];
			width  = new double[dimension];
		}

		Cell(int inp_dimension, double [] inp_corner, double [] inp_width) {
			dimension = inp_dimension;
			corner = new double[dimension];
			width  = new double[dimension];
			for(int d = 0; d < dimension; d++) setCorner(d, inp_corner[d]);
			for(int d = 0; d < dimension; d++) setWidth( d,  inp_width[d]);
		}

		double getCorner(int d) {
			return corner[d];
		}

		double getWidth(int d) {
			return width[d];
		}

		void setCorner(int d, double val) {
			corner[d] = val;
		}

		void setWidth(int d, double val) {
			width[d] = val;
		}

		// Checks whether a point lies in a cell
		boolean containsPoint(double point[])
		{
			return containsPoint(point, 0);
		}

		// Checks whether the point starting at the given offset of a flat matrix lies in a cell
		boolean containsPoint(double[] flatMatrix, int offset)
		{
			for(int d = 0; d < dimension; d++) {
				double coordinate = flatMatrix[offset + d];
				if(corner[d] - width[d] > coordinate) return false;
				if(corner[d] + width[d] < coordinate) return false;
			}
			return true;
		}
	}
}

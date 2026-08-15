package com.jujutsu.tsne.barneshut;

import java.io.File;

import com.jujutsu.tsne.TSneConfiguration;
import com.jujutsu.utils.MatrixUtils;
import com.jujutsu.utils.TSneUtils;

public class TSneTest {
    public static void main(String[] args) {
        int initial_dims = 55;
        double perplexity = 20.0;
        double[][] X = MatrixUtils.simpleRead2DMatrix(new File("src/test/resources/datasets/mnist2500_X.txt"), "   ");
        System.out.println("Finished reading data, starting t-SNE...");

        BarnesHutTSne tsne;
        boolean parallel = true;
        if (parallel) {
            tsne = new ParallelBHTsne();
        } else {
            tsne = new BHTSne();
        }
        TSneConfiguration config = TSneUtils.buildConfig(X, 2, initial_dims, perplexity, 1000);
        double[][] Y = tsne.tsne(config);

        // Plot Y or save Y to file and plot with some other tool such as for
        // instance R
    }
}

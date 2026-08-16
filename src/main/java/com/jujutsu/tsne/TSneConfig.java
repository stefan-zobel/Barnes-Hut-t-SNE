package com.jujutsu.tsne;

public class TSneConfig implements TSneConfiguration {

    protected double[][] xin;
    protected int outputDims;
    protected int initial_dims;
    protected double perplexity;
    protected int max_iter;
    protected boolean use_pca;
    protected double theta;
    protected boolean silent;
    protected boolean print_error;

    public TSneConfig(double[][] xin, int outputDims, int initial_dims, double perplexity, int max_iter,
            boolean use_pca, double theta, boolean silent, boolean print_error) {
        this.xin = xin;
        this.outputDims = outputDims;
        this.initial_dims = initial_dims;
        this.perplexity = perplexity;
        this.max_iter = max_iter;
        this.use_pca = use_pca;
        this.theta = theta;
        this.silent = silent;
        this.print_error = print_error;
    }

    /**
     * @inheritDoc
     */
    @Override
    public double[][] getXin() {
        return xin;
    }

    /**
     * @inheritDoc
     */
    @Override
    public void setXin(double[][] xin) {
        this.xin = xin;
    }

    /**
     * @inheritDoc
     */
    @Override
    public int getOutputDims() {
        return outputDims;
    }

    /**
     * @inheritDoc
     */
    @Override
    public void setOutputDims(int n) {
        this.outputDims = n;
    }

    /**
     * @inheritDoc
     */
    @Override
    public int getInitialDims() {
        return initial_dims;
    }

    /**
     * @inheritDoc
     */
    @Override
    public void setInitialDims(int initial_dims) {
        this.initial_dims = initial_dims;
    }

    /**
     * @inheritDoc
     */
    @Override
    public double getPerplexity() {
        return perplexity;
    }

    /**
     * @inheritDoc
     */
    @Override
    public void setPerplexity(double perplexity) {
        this.perplexity = perplexity;
    }

    /**
     * @inheritDoc
     */
    @Override
    public int getMaxIter() {
        return max_iter;
    }

    /**
     * @inheritDoc
     */
    @Override
    public void setMaxIter(int max_iter) {
        this.max_iter = max_iter;
    }

    /**
     * @inheritDoc
     */
    @Override
    public boolean usePca() {
        return use_pca;
    }

    /**
     * @inheritDoc
     */
    @Override
    public void setUsePca(boolean use_pca) {
        this.use_pca = use_pca;
    }

    /**
     * @inheritDoc
     */
    @Override
    public double getTheta() {
        return theta;
    }

    /**
     * @inheritDoc
     */
    @Override
    public void setTheta(double theta) {
        this.theta = theta;
    }

    /**
     * @inheritDoc
     */
    @Override
    public boolean silent() {
        return silent;
    }

    /**
     * @inheritDoc
     */
    @Override
    public void setSilent(boolean silent) {
        this.silent = silent;
    }

    /**
     * @inheritDoc
     */
    @Override
    public boolean printError() {
        return print_error;
    }

    /**
     * @inheritDoc
     */
    @Override
    public void setPrintError(boolean print_error) {
        this.print_error = print_error;
    }

    @Override
    public int getXStartDim() {
        return xin[0].length;
    }

    @Override
    public int getNrRows() {
        return xin.length;
    }
}

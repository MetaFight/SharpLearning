﻿using System;

namespace SharpLearning.GradientBoost.GBM
{
    /// <summary>
    /// <summary>
    /// Regression gradient boost learner based on 
    /// http://statweb.stanford.edu/~jhf/ftp/trebst.pdf
    /// A series of regression trees are fitted stage wise on the residuals of the previous stage.
    /// The resulting models are ensembled together using addition. Implementation based on:
    /// http://gradientboostedmodels.googlecode.com/files/report.pdf
    /// </summary>
    /// </summary>
    public class RegressionQuantileLossGradientBoostLearner : GBMGradientBoostRegressorLearner
    {
        /// <summary>
        /// <summary>
        ///  Quantile regression gradient boost learner. 
        ///  A series of regression trees are fitted stage wise on the residuals of the previous stage
        /// </summary>
        /// <param name="iterations">The number of iterations or stages</param>
        /// <param name="learningRate">How much each iteration should contribute with</param>
        /// <param name="maximumTreeDepth">The maximum depth of the tree models</param>
        /// <param name="maximumLeafCount">The maximum leaf count of the tree models</param>
        /// <param name="minimumSplitSize">minimum node split size in the trees 1 is default</param>
        /// <param name="minimumInformationGain">The minimum improvement in information gain before a split is made</param>
        /// <param name="subSampleRatio">ratio of observations sampled at each iteration. Default is 1.0. 
        /// If below 1.0 the algorithm changes to stochastic gradient boosting. 
        /// This reduces variance in the ensemble and can help ounter overfitting</param>
        /// <param name="quantile">The quantile used in quantile regression. 
        /// 0.5 is the median and corresponds to absolute loss or LAD regression</param>
        /// <param name="numberOfThreads">Number of threads to use for paralization</param>
        public RegressionQuantileLossGradientBoostLearner(int iterations, double learningRate, int maximumTreeDepth,
            int minimumSplitSize, double minimumInformationGain, double subSampleRatio, double quantile, int numberOfThreads)
            : base(iterations, learningRate, maximumTreeDepth, minimumSplitSize, minimumInformationGain,
                subSampleRatio, new GBMQuantileLoss(quantile), numberOfThreads)

        {
        }

        /// <summary>
        /// <summary>
        ///  Quantile regression gradient boost learner.
        ///  A series of regression trees are fitted stage wise on the residuals of the previous stage
        /// </summary>
        /// <param name="iterations">The number of iterations or stages</param>
        /// <param name="learningRate">How much each iteration should contribute with</param>
        /// <param name="maximumTreeDepth">The maximum depth of the tree models</param>
        /// <param name="maximumLeafCount">The maximum leaf count of the tree models</param>
        /// <param name="minimumSplitSize">minimum node split size in the trees 1 is default</param>
        /// <param name="minimumInformationGain">The minimum improvement in information gain before a split is made</param>
        /// <param name="subSampleRatio">ratio of observations sampled at each iteration. Default is 1.0. 
        /// If below 1.0 the algorithm changes to stochastic gradient boosting. 
        /// This reduces variance in the ensemble and can help ounter overfitting</param>
        /// <param name="quantile">The quantile used in quantile regression. 
        /// 0.5 is the median and corresponds to absolute loss or LAD regression</param>
        public RegressionQuantileLossGradientBoostLearner(int iterations = 100, double learningRate = 0.1, int maximumTreeDepth = 3,
            int minimumSplitSize = 1, double minimumInformationGain = 0.000001, double subSampleRatio = 1.0, double quantile = 0.9)
            : base(iterations, learningRate, maximumTreeDepth, minimumSplitSize, minimumInformationGain,
                subSampleRatio, new GBMQuantileLoss(quantile), Environment.ProcessorCount)
        {
        }
    }
}

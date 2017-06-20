package com.github.okwrtdsh.dl4k_examples.mnist

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.LoggerFactory

object MLP {
    private val log = LoggerFactory.getLogger(MLP::class.java)

    @Throws(Exception::class)
    @JvmStatic
    fun main(args : Array<String>) {
        val numRows = 28
        val numColumns = 28
        val outputNum = 10
        val hiddenNum = 100
        val batchSize = 128
        val rngSeed = 123 // random number seed for reproducibility
        val numEpochs = 15

        // Get the DataSetIterators
        val mnistTrain = MnistDataSetIterator(batchSize, true, rngSeed)
        val mnistTest = MnistDataSetIterator(batchSize, false, rngSeed)

        // Build model
        log.info("Build model....")
        val model = MultiLayerNetwork(
                NeuralNetConfiguration.Builder().apply {
                    seed = rngSeed.toLong()
                    numIterations = numEpochs
                    updater = Updater.ADAM
                    regularization(true)
                    l2 = 1e-4
                }.list()/* Convert ListBuilder */.apply {
                    // Add DenseLayer
                    layer(0, DenseLayer.Builder()
                            .nIn(numRows * numColumns)
                            .nOut(hiddenNum)
                            .activation(Activation.RELU)
                            .weightInit(WeightInit.XAVIER)
                            .build())
                    // Add OutputLayer
                    layer(1, OutputLayer.Builder(LossFunction.COSINE_PROXIMITY)
                            .nIn(hiddenNum)
                            .nOut(outputNum)
                            .activation(Activation.SOFTMAX)
                            .weightInit(WeightInit.XAVIER)
                            .build())

                    isPretrain = false
                    isBackprop = true

                }.build() // Build ListBuilder
        ).apply {
            init()
            // Print the score with every 1 epoch
            setListeners(ScoreIterationListener((mnistTrain.numExamples() / batchSize).toInt()))
        }

        // Train model
        log.info("Train model....")
        model.fit(mnistTrain)

        // Evaluate model
        log.info("Evaluate model....")
        val eval = Evaluation(outputNum) // Create an evaluation object with 10 possible classes
        mnistTest.asSequence().map {
            // Check the prediction against the true class
            eval.eval(it.labels, model.output(it.featureMatrix)/* get the networks prediction */)
        }.toList()
        log.info(eval.stats())
        log.info("****************Example finished********************")
    }
}
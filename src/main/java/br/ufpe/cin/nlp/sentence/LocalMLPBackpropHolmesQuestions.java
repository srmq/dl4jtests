package br.ufpe.cin.nlp.sentence;

import java.io.File;
import java.util.Arrays;
import java.util.Collections;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LocalMLPBackpropHolmesQuestions {
	
	private static Logger log = LoggerFactory.getLogger(LocalMLPBackpropHolmesQuestions.class);

	public static void main(String[] args) {
        final int numInputs = 200;
        int outputNum = 5;
        int numSamples = 2040;
        int batchSize = 1040;//1040;
        int iterations = 40; //10000;
        long seed = 123;
        int listenerFreq = 100;

        log.info("Load data....");
        DataSetIterator iter = new HolmesDatasetIterator(batchSize, numSamples, new File("/home/srmq/git/holmes-question-producer/Holmes-NumericQuestions-Training.txt"));

        log.info("Build model....");
        
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)

                .learningRate(1e-3)
                .l1(0.3).regularization(true).l2(1e-3)
                .constrainGradientToUnitNorm(true)
                .list(3)
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numInputs)
                        .activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(numInputs).nOut(300)
                        .activation("tanh")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunction.MCXENT)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                		.nIn(300).nOut(outputNum).build())
                .backprop(true).pretrain(false)
                .build(); 

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));

        log.info("Train model....");
        while(iter.hasNext()) {
            DataSet holmes = iter.next();
            holmes.normalizeZeroMeanZeroUnitVariance();
            model.fit(holmes);
        }
        iter.reset();

        log.info("Evaluate weights....");
        for(org.deeplearning4j.nn.api.Layer layer : model.getLayers()) {
            INDArray w = layer.getParam(DefaultParamInitializer.WEIGHT_KEY);
            log.info("Weights: " + w);
        }


        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum);
        DataSetIterator iterTest = new HolmesDatasetIterator(numSamples, numSamples , new File("/home/srmq/git/holmes-question-producer/Holmes-NumericQuestions-Training.txt"));
        DataSet test = iterTest.next();
        test.normalizeZeroMeanZeroUnitVariance();
        INDArray output = model.output(test.getFeatureMatrix());
        eval.eval(test.getLabels(), output);
        log.info(eval.stats());
        log.info("****************Example finished********************");
        

	}

}

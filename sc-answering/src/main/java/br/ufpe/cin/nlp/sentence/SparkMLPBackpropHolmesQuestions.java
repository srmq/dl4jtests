package br.ufpe.cin.nlp.sentence;

import java.io.IOException;
import java.io.OutputStreamWriter;
import java.net.URI;

import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.feature.StandardScaler;
import org.apache.spark.mllib.feature.StandardScalerModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import scala.Tuple2;

public class SparkMLPBackpropHolmesQuestions {
	private static Logger log = LoggerFactory.getLogger(SparkMLPBackpropHolmesQuestions.class);
	
	
	public static void main(String[] args) throws IOException {
		// 1st parameter should be directory where to put model files
		// 2nd parameter should be the training data url
		if (args.length != 2) {
			throw new IllegalArgumentException("Should pass the directory where to save model information as first parameter and training dataset URL as second parameter");
		}
		final String outputDirURL = args[0]; //ex.: "file:///home/srmq/git/holmes-question-producer";
		final String trainingDatasetURL = args[1]; //ex.: "file:///home/srmq/git/holmes-question-producer/Holmes-NumericQuestions-Training.txt";
        final long seed = 1;
        final int iterations = 10000;
        final int numInputs = 200;
        final int outputNum = 5;
        final int batchSize = 10000;
        
		final JavaSparkContext sc = new JavaSparkContext(new SparkConf().setAppName("holmes"));
		JavaRDD<String> linesRDD = sc.textFile(trainingDatasetURL);
		JavaRDD<LabeledPoint> pointsRDD = linesRDD.map(s -> Util.questionProducerLine2LabeledPoint(s));
        StandardScaler scaler = new StandardScaler(true,true);
        

        final StandardScalerModel scalarModel = scaler.fit(pointsRDD.map(p -> p.features()).rdd());
        JavaRDD<LabeledPoint> pointsRDDNormalized = pointsRDD.map(p -> {return new LabeledPoint(p.label(), scalarModel.transform(p.features()));});
        JavaRDD<LabeledPoint>[] trainTestSplit = pointsRDDNormalized.randomSplit(new double[]{90, 10});
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
        
        //train the network
        SparkDl4jMultiLayer trainLayer = new SparkDl4jMultiLayer(sc.sc(),conf);
        //fit on the training set
        MultiLayerNetwork trainedNetwork = trainLayer.fit(trainTestSplit[0],batchSize);
        final SparkDl4jMultiLayer trainedNetworkWrapper = new SparkDl4jMultiLayer(sc.sc(),trainedNetwork);

        // Compute raw scores on the test set.
        JavaRDD<Tuple2<Object, Object>> predictionAndLabels = trainTestSplit[1].map(
                new Function<LabeledPoint, Tuple2<Object, Object>>() {
                    /**
					 * 
					 */
					private static final long serialVersionUID = 8706762370097553951L;

					public Tuple2<Object, Object> call(LabeledPoint p) {
                        Vector prediction = trainedNetworkWrapper.predict(p.features());
                        double max = 0;
                        double idx = 0;
                        for(int i = 0; i < prediction.size(); i++) {
                            if(prediction.apply(i) > max) {
                                idx = i;
                                max = prediction.apply(i);
                            }
                        }

                        return new Tuple2<Object, Object>(idx, p.label());
                    }
                }
        );

        log.info("Saving model...");

        URI outputURI = URI.create(outputDirURL);
        FileSystem hadoopFS = FileSystem.get(sc.hadoopConfiguration());
        FSDataOutputStream modelFile = hadoopFS.create(new Path(new Path(outputURI), "model.bin"));
        Nd4j.write(trainedNetwork.params(), modelFile);
        modelFile.close();
        

        FSDataOutputStream confFile = hadoopFS.create(new Path(new Path(outputURI), "conf.yaml"));
        OutputStreamWriter outWConf = new OutputStreamWriter(confFile, "UTF-8");
        outWConf.write(trainedNetwork.conf().toYaml());
        outWConf.close();
        confFile.close();

        MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
        double fMeasure = metrics.fMeasure();
        double precision = metrics.precision();
        StringBuffer stb = new StringBuffer();
        stb.append("Precision: ");
        stb.append(precision);
        stb.append(System.lineSeparator());
        stb.append("F-Measure: ");
        stb.append(fMeasure);
        stb.append(System.lineSeparator());
        
        FSDataOutputStream metricsFile = hadoopFS.create(new Path(new Path(outputURI), "metrics.txt"));
        OutputStreamWriter outWMetrics = new OutputStreamWriter(metricsFile, "UTF-8");
        outWMetrics.write(stb.toString());
        outWMetrics.close();
        metricsFile.close();
        sc.stop();
        sc.close();
	}

}

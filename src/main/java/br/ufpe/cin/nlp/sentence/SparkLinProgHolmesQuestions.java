package br.ufpe.cin.nlp.sentence;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutionException;

import org.apache.spark.Partitioner;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaFutureAction;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.nd4j.linalg.api.ndarray.INDArray;

import br.ufpe.cin.nlp.sentence.spark.functions.CombineLinProgCorrections;
import br.ufpe.cin.nlp.sentence.spark.functions.CreateLinProgCorrectionCombiner;
import br.ufpe.cin.nlp.sentence.spark.functions.MergeLinProgQuestionCorrection;
import br.ufpe.cin.nlp.sentence.spark.functions.WordsInQuestions;
import br.ufpe.cin.util.io.JsonSerializer;
import scala.Tuple2;

public class SparkLinProgHolmesQuestions {
/*
	private static class CorrectionsCollector implements Runnable {
		private Map<Integer, Map<Tuple2<String, Integer>, Double>> correctionsByPartitionMap;
		private JavaPairRDD<Integer, Map<Tuple2<String, Integer>, Double>> correctionsByPartition;
		private int partitionNumber;

		public CorrectionsCollector(Map<Integer, Map<Tuple2<String, Integer>, Double>> correctionsByPartitionMap,
				JavaPairRDD<Integer, Map<Tuple2<String, Integer>, Double>> correctionsByPartition, int partitionNumber) {
			this.correctionsByPartitionMap = correctionsByPartitionMap;
			this.correctionsByPartition = correctionsByPartition;
			this.partitionNumber = partitionNumber;
		}
		
		@Override
		public void run() {
			List<Tuple2<Integer, Map<Tuple2<String, Integer>, Double>>> collects[] = this.correctionsByPartition.collectPartitions(new int[]{this.partitionNumber});
			assert(collects.length == 1);
			for (Tuple2<Integer, Map<Tuple2<String, Integer>, Double>> correct : collects[0]) {
				this.correctionsByPartitionMap.put(correct._1(), correct._2());
			}
			this.correctionsByPartition = null;
			this.correctionsByPartitionMap = null;
		}
		
	}
*/	
	public static class WordEmbeddings {
		private InMemoryLookupTable lookupTable;

		public WordEmbeddings(File file) throws FileNotFoundException {
			final Pair<InMemoryLookupTable, VocabCache> vocabPair = WordVectorSerializer.loadTxt(file);
			this.lookupTable = vocabPair.getFirst();
		}

		public Map<String, INDArray> embeddingsForWords(Collection<String> words) {
			Map<String, INDArray> result = new HashMap<String, INDArray>(words.size());
			for (String word : words) {
				result.put(word, this.lookupTable.vector(word));
			}
			return result;
		}
	}

	public static <T> List<T> createList(T elem) {
		ArrayList<T> ret = new ArrayList<T>(1);
		ret.add(elem);
		return ret;
	}

	@SuppressWarnings("unchecked")
	public static void main(String[] args) throws IOException, InterruptedException, ExecutionException {
/*		final JavaSparkContext sc = new JavaSparkContext(
				new SparkConf().setMaster("local").setAppName("holmes-linprog").set("spark.default.parallelism", "4").set("spark.driver.memory", "6g").set("spark.executor.memory", "3g"));*/
		final JavaSparkContext sc = new JavaSparkContext(
				new SparkConf().setAppName("holmes-linprog"));
		
		final String jsonQuestionsURL = "file:///home/srmq/git/holmes-question-producer/trainingQuestions-1m.spark.json";
		URL tfIdfInfoUrl = new URL(args[0]);
		final String embeddingsPath = "/home/srmq/git/sc-answering/WordVec-Holmes-Glove-StopwordsRemoved.txt";
		
		final File outputDir = new File("/home/srmq/git/sc-answering/results");

		final int defaultParallelism = sc.defaultParallelism();

		System.out.println("default parallelism is " + defaultParallelism);
		final String maxCores = sc.getConf().get("spark.cores.max");
		System.out.println("Maximum number of cores is: " + maxCores);
		final Broadcast<Map<String, Tuple2<Integer, Integer>>> bTfIdfInfo;
		double learnRate = 0.1;
		double iterations = 100;

		{
			Map<String, Tuple2<Integer, Integer>> wordFreqInfo = readTFIdfInfo(tfIdfInfoUrl);
			bTfIdfInfo = sc.broadcast(wordFreqInfo);
		}

		// Partitioner partitioner = new HashPartitioner(defaultParallelism);

		// Integer -- o numero da particao
		// String -- O objeto Question em JSON
		// Map<String, INDArray> -- mapa das palavras da questao para embeddings
		JavaPairRDD<Integer, Tuple2<String, Map<String, INDArray>>> partitionJsonQuestionAndEmbeddings = computeQuestionsAndEmbeddingsRDD(
				sc, jsonQuestionsURL, embeddingsPath);
		//partitionJsonQuestionAndEmbeddings.persist(StorageLevel.MEMORY_AND_DISK());
		// agora trabalhar nesse rdd. considerar fazer mais particoes e fazer
		// minibatches com todas as questões ou fazer individualmente
		// vou começar fazendo individualmente

		Map<Tuple2<String, Integer>, Double> unifiedCorrections = new HashMap<Tuple2<String, Integer>, Double>();
		
		@SuppressWarnings("rawtypes")
		JsonSerializer serializer = new JsonSerializer(unifiedCorrections.getClass()); 
		
		int initialIteration = 1;
		if (args.length == 3) { //1st is tfidffile, 2nd is initial corrections, 3rd is initial iteration
			unifiedCorrections = SentenceCompletionAnswering.readCorrections(new String[]{args[1]});
			initialIteration = Integer.parseInt(args[2]);
		}

		for (int i = initialIteration; i <= iterations; i++) { 
			Broadcast<Map<Tuple2<String, Integer>, Double>> bdCastUnifiedCorrections = sc.broadcast(unifiedCorrections);
			JavaPairRDD<Integer, Map<Tuple2<String, Integer>, Double>> correctionsByPartition = partitionJsonQuestionAndEmbeddings
					.combineByKey(new CreateLinProgCorrectionCombiner(bTfIdfInfo, bdCastUnifiedCorrections, learnRate),
							new MergeLinProgQuestionCorrection(bTfIdfInfo, learnRate), new CombineLinProgCorrections());

			cleanOutputFilesIterationI(outputDir, i);
			final Map<Integer, Map<Tuple2<String, Integer>, Double>> correctionsByPartitionMap = new HashMap<Integer, Map<Tuple2<String, Integer>, Double>>(correctionsByPartition.partitions().size()); //substituir por algo paralelo
			final int iter = i;

			//--- INICIO PARALELO FUNCIONA MAS ESCREVE MUITOS ARQUIVOS
			
			System.out.println(">>>>>>>> FINAL RDD to process has " + correctionsByPartition.partitions().size() + " partitions");
			correctionsByPartition.foreachPartition(itCorrection -> {
				final Tuple2<Integer, Map<Tuple2<String, Integer>, Double>> corrPartition = itCorrection.next();
				final File tempOutFile = new File(outputDir, "corrections_Iter" + iter + "_Part" + corrPartition._1() + ".json.gz");
				@SuppressWarnings("rawtypes")
				JsonSerializer serializerCorr = new JsonSerializer(corrPartition._2().getClass());
				serializerCorr.serialize(corrPartition._2(), tempOutFile, true);
			});
			for (int part = 0; part < correctionsByPartition.partitions().size(); part++) {
				final File tempOutFile = new File(outputDir, "corrections_Iter" + iter + "_Part" + part + ".json.gz");
				final Map<Tuple2<String, Integer>, Double> correct = SentenceCompletionAnswering.readCorrections(new String[]{tempOutFile.getAbsolutePath()});
				correctionsByPartitionMap.put(part, correct);
				unifyCorrections(unifiedCorrections, correctionsByPartitionMap);
				correctionsByPartitionMap.clear();
			}
			cleanOutputFilesIterationI(outputDir, i);
			
			//--- FIM PARALELO FUNCIONA MAS ESCREVE MUITOS ARQUIVOS
			
			
			//inicio paralelo nao funciona dessa forma, fazer collect paralelo de cada particao
			//usar ExecutorService como em SentenceCompletionAnswering com collect para cada uma das particoes
			/*
			{
				ExecutorService executor = Executors.newFixedThreadPool(Integer.parseInt(maxCores));
				final int totalPartitions = correctionsByPartition.partitions().size();
				List<Future<?>> futures = new ArrayList<Future<?>>(totalPartitions);
				for (int part = 0; part < totalPartitions; part++) {
					futures.add(executor.submit(new CorrectionsCollector(correctionsByPartitionMap, correctionsByPartition, part)));
				}
				for (Future<?> future : futures) {
					future.get();
				}
				executor.shutdown();
				
			}*/

			
			// vamos ver se funciona agora
			/*
			{
				final JavaFutureAction<List<Tuple2<Integer, Map<Tuple2<String, Integer>, Double>>>> correctionsListActions = correctionsByPartition.collectAsync();
				final List<Tuple2<Integer, Map<Tuple2<String, Integer>, Double>>> correctionsList = correctionsListActions.get();
				for (Tuple2<Integer, Map<Tuple2<String, Integer>, Double>> tuple2 : correctionsList) {
					correctionsByPartitionMap.put(tuple2._1(), tuple2._2());
					unifyCorrections(unifiedCorrections, correctionsByPartitionMap);
				    correctionsByPartitionMap.clear();
				}
			}
			*/
			//funciona mas precisa de muita memoria no driver

			bdCastUnifiedCorrections.destroy();
			serializer.serialize(unifiedCorrections, new File("/home/srmq/git/sc-answering/results/corrections-1mHolmes-iteration- " + i + ".json.gz"), true);
		}

		sc.stop();
	}

	private static void cleanOutputFilesIterationI(File outputDir, int iter) {
		File f = new File(outputDir, "corrections_Iter" + iter + "_Part" + "0.json.gz");
		for (int i = 1; f.exists(); i++) {
			f.delete();
			f = new File(outputDir, "corrections_Iter" + iter + "_Part" + i + ".json.gz");
		}
	}

	private static Map<Tuple2<String, Integer>, Double> unifyCorrections(
			Map<Tuple2<String, Integer>, Double> unifiedCorrections,
			Map<Integer, Map<Tuple2<String, Integer>, Double>> correctionsByPartitionMap) {

		for (Map<Tuple2<String, Integer>, Double> correction : correctionsByPartitionMap.values()) {
			for (Map.Entry<Tuple2<String, Integer>, Double> correctionEntry : correction.entrySet()) {
				final Double currUnifiedEntry = unifiedCorrections.get(correctionEntry.getKey());
				if (currUnifiedEntry == null) {
					unifiedCorrections.put(correctionEntry.getKey(), correctionEntry.getValue());
				} else {
					if (!currUnifiedEntry.equals(correctionEntry)) {
						final double newValue = (currUnifiedEntry.doubleValue() + correctionEntry.getValue()) / 2.0;
						unifiedCorrections.put(correctionEntry.getKey(), newValue);
					}
				}
			}
		}
		return unifiedCorrections;
	}

	private static JavaPairRDD<Integer, Tuple2<String, Map<String, INDArray>>> computeQuestionsAndEmbeddingsRDD(
			final JavaSparkContext sc, final String jsonQuestionsURL, final String embeddingsPath)
					throws FileNotFoundException, InterruptedException, ExecutionException {
		JavaRDD<String> jsonQuestionsRDD = sc.textFile(jsonQuestionsURL, 100);
		
		System.out.println("Initial RDD has " + jsonQuestionsRDD.partitions().size() + " partitions");
		JavaPairRDD<Integer, String> partitionAndJsonQuestions = questionsByPartition(jsonQuestionsRDD);
		// transformar o acima em javapairrdd

		JavaPairRDD<Integer, Set<String>> wordsByPart = wordsByPartition(jsonQuestionsRDD);

		// fazer join do primeiro javapairrdd (partitionAndJsonQuestions) com
		// outro com vetores de embeddings filtrados por partition
		// (partitionEmbeddingMapRDD)
		JavaFutureAction<List<Tuple2<Integer, Set<String>>>> wordsByPartitionFuture = wordsByPart.collectAsync();
		SparkLinProgHolmesQuestions.WordEmbeddings embeds = new SparkLinProgHolmesQuestions.WordEmbeddings(
				new File(embeddingsPath));
		final List<Tuple2<Integer, Set<String>>> wordsByPartitionList = wordsByPartitionFuture.get();
		List<Tuple2<Integer, Map<String, INDArray>>> filteredEmbeddingsObj = filterEmbeddings(wordsByPartitionList,
				embeds);
		JavaPairRDD<Integer, Map<String, INDArray>> partitionEmbeddingMapRDD = sc
				.parallelizePairs(filteredEmbeddingsObj);
		Partitioner keyPartionRDD = new Partitioner() {

			/**
			 * 
			 */
			private static final long serialVersionUID = -7779784669867185708L;

			@Override
			public int getPartition(Object arg0) {
				Integer partition = (Integer) arg0;
				return partition;
			}

			@Override
			public int numPartitions() {
				return wordsByPartitionList.size();
			}

		};
		partitionEmbeddingMapRDD = partitionEmbeddingMapRDD.partitionBy(keyPartionRDD);

		JavaPairRDD<Integer, Tuple2<String, Map<String, INDArray>>> partitionJsonQuestionAndEmbeddings = partitionAndJsonQuestions
				.join(partitionEmbeddingMapRDD);
		return partitionJsonQuestionAndEmbeddings;
	}

	private static List<Tuple2<Integer, Map<String, INDArray>>> filterEmbeddings(
			List<Tuple2<Integer, Set<String>>> wordsByPartitionList,
			SparkLinProgHolmesQuestions.WordEmbeddings embeds) {

		final List<Tuple2<Integer, Map<String, INDArray>>> result = new ArrayList<Tuple2<Integer, Map<String, INDArray>>>(
				wordsByPartitionList.size());
		for (Tuple2<Integer, Set<String>> wordByPart : wordsByPartitionList) {
			Tuple2<Integer, Map<String, INDArray>> embedsForThisPartition = new Tuple2<Integer, Map<String, INDArray>>(
					wordByPart._1(), embeds.embeddingsForWords(wordByPart._2()));
			result.add(embedsForThisPartition);
		}
		return result;
	}

	private static JavaPairRDD<Integer, String> questionsByPartition(JavaRDD<String> jsonQuestionsRDD) {
		JavaRDD<Tuple2<Integer, String>> partJsonQuestion = jsonQuestionsRDD
				.mapPartitionsWithIndex((index, iterLines) -> {
					return new Iterator<Tuple2<Integer, String>>() {
						@Override
						public boolean hasNext() {
							return iterLines.hasNext();
						}

						@Override
						public Tuple2<Integer, String> next() {
							return new Tuple2<Integer, String>(index, iterLines.next());
						}
					};
				} , true);
		System.out.println(">>>>>>>>>> AFTER mapPartitionsWithIndex has " + partJsonQuestion.partitions().size() + " partitions");
		JavaPairRDD<Integer, String> partitionAndJsonQuestions = partJsonQuestion
				.mapToPair(tuple -> new Tuple2<Integer, String>(tuple._1(), tuple._2()));
		System.out.println(">>>>>>>>>> AFTER JavaPairRDD has " + partitionAndJsonQuestions.partitions().size() + " partitions");
		return partitionAndJsonQuestions;
	}

	private static JavaPairRDD<Integer, Set<String>> wordsByPartition(JavaRDD<String> jsonQuestionsRDD) {
		JavaRDD<Tuple2<Integer, Set<String>>> partitionWords = jsonQuestionsRDD
				.mapPartitionsWithIndex(new WordsInQuestions(), true);
		JavaPairRDD<Integer, Set<String>> partitionAndWords = partitionWords
				.mapToPair(tuple -> new Tuple2<Integer, Set<String>>(tuple._1(), tuple._2()));
		return partitionAndWords;
	}

	private static Map<String, Tuple2<Integer, Integer>> readTFIdfInfo(URL tfIdfInfoUrl) throws IOException {
		final JsonSerializer<TfIdfInfo> tfIdfSerializer = new JsonSerializer<TfIdfInfo>(TfIdfInfo.class);
		TfIdfInfo tfIdfInfo = tfIdfSerializer.deserialize(tfIdfInfoUrl.openStream(), true);
		Map<String, Tuple2<Integer, Integer>> tfIdfMap = new HashMap<String, Tuple2<Integer, Integer>>(
				tfIdfInfo.getWordFrequencies().size());
		for (String word : tfIdfInfo.getWordFrequencies().keySet()) {
			Integer wordFreq = tfIdfInfo.wordFrequency(word);
			Integer docFreq = tfIdfInfo.docAppearedIn(word);
			tfIdfMap.put(word, new Tuple2<Integer, Integer>(wordFreq, docFreq));
		}
		return tfIdfMap;
	}

}

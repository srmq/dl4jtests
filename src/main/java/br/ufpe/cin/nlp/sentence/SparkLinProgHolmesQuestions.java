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
import org.apache.spark.sql.SQLContext;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.nd4j.linalg.api.ndarray.INDArray;

import br.ufpe.cin.nlp.sentence.spark.functions.WordsInQuestions;
import br.ufpe.cin.util.io.JsonSerializer;
import scala.Tuple2;



public class SparkLinProgHolmesQuestions {
	public static class WordEmbeddings {
		private InMemoryLookupTable lookupTable;
		private VocabCache vocabCache;

		public WordEmbeddings(File file) throws FileNotFoundException {
			final Pair<InMemoryLookupTable, VocabCache> vocabPair = WordVectorSerializer.loadTxt(file);
			this.lookupTable = vocabPair.getFirst();
			this.vocabCache = vocabPair.getSecond();
			
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
	
	public static void main(String[] args) throws IOException, InterruptedException, ExecutionException {
		final JavaSparkContext sc = new JavaSparkContext(new SparkConf().setMaster("local").setAppName("holmes-linprog").set("spark.default.parallelism", "4"));
		final String jsonQuestionsURL = "file:///home/srmq/git/holmes-question-producer/trainingQuestions.spark.json.gz";
		URL tfIdfInfoUrl = new URL(args[0]);
		final String embeddingsPath = "/home/srmq/git/sc-answering/WordVec-Holmes-Glove-StopwordsRemoved.txt";
		
		final int defaultParallelism = sc.defaultParallelism();
		System.out.println("default parallelism is " + defaultParallelism);
		Broadcast<Map<String, Tuple2<Integer, Integer>>> bTfIdfInfo;

		
		{
			Map<String, Tuple2<Integer, Integer>> wordFreqInfo = readTFIdfInfo(tfIdfInfoUrl);
			bTfIdfInfo = sc.broadcast(wordFreqInfo);
		}
		
		//Partitioner partitioner = new HashPartitioner(defaultParallelism);

		//ler com json
		//zip with index http://stackoverflow.com/questions/26828815/how-to-get-element-by-index-in-spark-rdd-java
		//pegar palavras em particoes
		// filtrar map e repassar para elas transformarem em vetores
		SQLContext sqlContext = new SQLContext(sc);
		JavaRDD<String> jsonQuestionsRDD = sc.textFile(jsonQuestionsURL, 100);
		
		JavaPairRDD<Integer, String> partitionAndJsonQuestions = questionsByPartition(jsonQuestionsRDD);
		//transformar o acima em javapairrdd
		
		JavaPairRDD<Integer, Set<String>> wordsByPart = wordsByPartition(jsonQuestionsRDD);

		//fazer join do primeiro javapairrdd (partitionAndJsonQuestions) com outro com vetores de embeddings  filtrados por partition (partitionEmbeddingMapRDD)
		JavaFutureAction<List<Tuple2<Integer, Set<String>>>> wordsByPartitionFuture = wordsByPart.collectAsync();
		SparkLinProgHolmesQuestions.WordEmbeddings embeds = new SparkLinProgHolmesQuestions.WordEmbeddings(new File(embeddingsPath)); 
		final List<Tuple2<Integer, Set<String>>> wordsByPartitionList = wordsByPartitionFuture.get();
		List<Tuple2<Integer, Map<String, INDArray>>> filteredEmbeddingsObj = filterEmbeddings(wordsByPartitionList, embeds);
		JavaPairRDD<Integer, Map<String, INDArray>> partitionEmbeddingMapRDD = sc.parallelizePairs(filteredEmbeddingsObj);
		Partitioner keyPartionRDD = new Partitioner() {

			/**
			 * 
			 */
			private static final long serialVersionUID = -7779784669867185708L;

			@SuppressWarnings("unchecked")
			@Override
			public int getPartition(Object arg0) {
				Tuple2<Integer, Map<String, INDArray>> element = (Tuple2<Integer, Map<String, INDArray>>) arg0;
				return element._1();
			}

			@Override
			public int numPartitions() {
				return wordsByPartitionList.size();
			}
			
		};
		partitionEmbeddingMapRDD = partitionEmbeddingMapRDD.partitionBy(keyPartionRDD);
		
		 JavaPairRDD<Integer, Tuple2<String, Map<String, INDArray>>> partitionJsonQuestionAndEmbeddings = partitionAndJsonQuestions.join(partitionEmbeddingMapRDD);
		 //agora trabalhar nesse array. considerar fazer mais particoes e fazer minibatches com todas as questões ou fazer individualmente
		
	 	sc.stop();
	}

	private static List<Tuple2<Integer, Map<String, INDArray>>> filterEmbeddings(
			List<Tuple2<Integer, Set<String>>> wordsByPartitionList, SparkLinProgHolmesQuestions.WordEmbeddings embeds) {
		final List<Tuple2<Integer, Map<String, INDArray>>> result = new ArrayList<Tuple2<Integer, Map<String, INDArray>>>(wordsByPartitionList.size());
		for (Tuple2<Integer, Set<String>> wordByPart : wordsByPartitionList) {
			Tuple2<Integer, Map<String, INDArray>> embedsForThisPartition = new Tuple2<Integer, Map<String, INDArray>>(wordByPart._1(), embeds.embeddingsForWords(wordByPart._2()));
			result.add(embedsForThisPartition);
		}
		return result;
	}

	private static JavaPairRDD<Integer, String> questionsByPartition(JavaRDD<String> jsonQuestionsRDD) {
		JavaRDD<Tuple2<Integer, String>> partJsonQuestion = jsonQuestionsRDD.mapPartitionsWithIndex((index, iterLines) -> 
			{ 
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
			}, true);
		JavaPairRDD<Integer, String> partitionAndJsonQuestions = partJsonQuestion.mapToPair(tuple -> new Tuple2<Integer, String>(tuple._1(), tuple._2()));
		return partitionAndJsonQuestions;
	}

	private static JavaPairRDD<Integer, Set<String>> wordsByPartition(JavaRDD<String> jsonQuestionsRDD) {
		JavaRDD<Tuple2<Integer, Set<String>>> partitionWords = jsonQuestionsRDD.mapPartitionsWithIndex(new WordsInQuestions(), true);
		JavaPairRDD<Integer, Set<String>> partitionAndWords = partitionWords.mapToPair(tuple -> new Tuple2<Integer, Set<String>>(tuple._1(), tuple._2())) ;
		return partitionAndWords;
	}

	private static Map<String, Tuple2<Integer, Integer>> readTFIdfInfo(URL tfIdfInfoUrl) throws IOException {
		final JsonSerializer<TfIdfInfo> tfIdfSerializer = new JsonSerializer<TfIdfInfo>(TfIdfInfo.class);
		TfIdfInfo tfIdfInfo = tfIdfSerializer.deserialize(tfIdfInfoUrl.openStream(), true);
		Map<String, Tuple2<Integer, Integer>> tfIdfMap = new HashMap<String, Tuple2<Integer, Integer>>(tfIdfInfo.getWordFrequencies().size());
		for (String word : tfIdfInfo.getWordFrequencies().keySet()) {
			Integer wordFreq = tfIdfInfo.wordFrequency(word);
			Integer docFreq = tfIdfInfo.docAppearedIn(word);
			tfIdfMap.put(word, new Tuple2<Integer, Integer>(wordFreq, docFreq));
		}
		return tfIdfMap;
	}
	
}

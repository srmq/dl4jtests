package br.ufpe.cin.nlp.sentence;

import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import br.ufpe.cin.util.io.OptionalSerializable;

import org.apache.spark.HashPartitioner;
import org.apache.spark.Partitioner;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;

import br.ufpe.cin.util.io.JsonSerializer;
import scala.Tuple2;



public class SparkLinProgHolmesQuestions {
	
	public static <T> List<T> createList(T elem) {
		ArrayList<T> ret = new ArrayList<T>(1);
		ret.add(elem);
		return ret;
	}
	
	public static void main(String[] args) throws IOException {
		final JavaSparkContext sc = new JavaSparkContext(new SparkConf().setMaster("local").setAppName("holmes-linprog").set("spark.default.parallelism", "4"));

		URL tfIdfInfoUrl = new URL(args[0]);
		final int defaultParallelism = sc.defaultParallelism();
		System.out.println("default parallelism is " + defaultParallelism);
		Partitioner partitioner = new HashPartitioner(defaultParallelism);

		JavaRDD<HashMap<String, Tuple2<Integer, Integer>>> distributedHash = tfIdfInfoRDD(sc, tfIdfInfoUrl,
				partitioner);
		distributedHash.cache();
	 	final String desiredWord = "woman";
	 	Tuple2<Integer, Integer> freqs = retrieveWordFreqDocFreqFor(partitioner, distributedHash,
				desiredWord);
	 	System.out.println("wordFreq is " + freqs._1 + " and docFreq is " + freqs._2);
	 	sc.stop();
	}

	private static Tuple2<Integer, Integer> retrieveWordFreqDocFreqFor(Partitioner partitioner,
			JavaRDD<HashMap<String, Tuple2<Integer, Integer>>> distributedHash, final String word) {
		final int desiredIndex = partitioner.getPartition(word);
	 	System.out.println("Desired partition is " + desiredIndex);


	 	List<OptionalSerializable<Tuple2<Integer, Integer>>> stats = distributedHash.mapPartitionsWithIndex((index, iterator) -> 
	 			{ 
	 				if (index == desiredIndex){
	 					System.out.println("index and desiredIndex are : " + index);
	 					final HashMap<String, Tuple2<Integer, Integer>> hashMap = iterator.next();
	 					assert hashMap.containsKey(word);
	 					return createList(OptionalSerializable.of(hashMap.get(word))).iterator(); 
	 				}
	 				else {
	 					final OptionalSerializable<Tuple2<Integer, Integer>> op = OptionalSerializable.empty();
	 					return createList(op).iterator();
	 				}
	 			}, true).collect();
	 	
	 	OptionalSerializable<Tuple2<Integer, Integer>> freqs = stats.get(desiredIndex);
	 	assert freqs.isPresent();
		return freqs.get();
	}

	private static JavaRDD<HashMap<String, Tuple2<Integer, Integer>>> tfIdfInfoRDD(final JavaSparkContext sc,
			URL tfIdfInfoUrl, Partitioner partitioner) throws IOException {
		final JsonSerializer<TfIdfInfo> tfIdfSerializer = new JsonSerializer<TfIdfInfo>(TfIdfInfo.class);
		TfIdfInfo tfIdfInfo = tfIdfSerializer.deserialize(tfIdfInfoUrl.openStream(), true);
		List<Tuple2<String, Tuple2<Integer, Integer>>> tfList = new ArrayList<Tuple2<String, Tuple2<Integer, Integer>>>(tfIdfInfo.getWordFrequencies().keySet().size());
		for (String word : tfIdfInfo.getWordFrequencies().keySet()) {
			final int wordFreq = tfIdfInfo.wordFrequency(word);
			final int docFreq = tfIdfInfo.docAppearedIn(word);
			final Tuple2<String, Tuple2<Integer, Integer>> tuple = new Tuple2<String, Tuple2<Integer, Integer>>(word, new Tuple2<Integer, Integer>(wordFreq, docFreq));
			tfList.add(tuple);
		}
		
	 	JavaPairRDD<String, Tuple2<Integer, Integer>> pairRDD = sc.parallelizePairs(tfList);
	 	pairRDD = pairRDD.partitionBy(partitioner);

	 	
	 	@SuppressWarnings("serial")
		FlatMapFunction<Iterator<Tuple2<String, Tuple2<Integer, Integer>>>
                        , HashMap<String, Tuple2<Integer, Integer>>> func = new FlatMapFunction<Iterator<Tuple2<String, Tuple2<Integer, Integer>>>
                        , HashMap<String, Tuple2<Integer, Integer>>>() {
	 		public Iterable<HashMap<String, Tuple2<Integer, Integer>>> call(Iterator<Tuple2<String,Tuple2<Integer,Integer>>> seq) {
	 			HashMap<String, Tuple2<Integer, Integer>> ret = new HashMap<String, Tuple2<Integer, Integer>>();

	 			while(seq.hasNext()) {
	 				final Tuple2<String,Tuple2<Integer,Integer>> tuple = seq.next();
	 				ret.put(tuple._1, tuple._2);
	 			}
	 			List<HashMap<String, Tuple2<Integer, Integer>>> retList = new ArrayList<HashMap<String, Tuple2<Integer, Integer>>>(1);
	 			retList.add(ret);
	 			
	 			return retList ; 
	 		}
	 		
	 	};
	 	JavaRDD<HashMap<String, Tuple2<Integer, Integer>>> distributedHash = pairRDD.mapPartitions(func, true);
		return distributedHash;
	}
	
}

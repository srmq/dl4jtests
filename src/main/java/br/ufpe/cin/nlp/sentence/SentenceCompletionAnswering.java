package br.ufpe.cin.nlp.sentence;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.zip.GZIPInputStream;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.distances.CosineSimilarity;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.util.concurrent.AtomicDoubleArray;

import br.ufpe.cin.nlp.sentence.base.SentenceCompletionQuestions;
import br.ufpe.cin.nlp.sentence.base.SentenceCompletionQuestions.Question;
import br.ufpe.cin.util.io.JsonSerializer;
import scala.Int;
import scala.Tuple2;

public class SentenceCompletionAnswering {

	private static class IterationProcessor implements Runnable {
		// iteration is one-based
		private int iteration;
		private SentenceCompletionAnswering scAns;
		private AtomicIntegerArray correctQuestions;
		private AtomicDoubleArray percentCorret;

		public IterationProcessor(int iteration, SentenceCompletionAnswering scAns, AtomicIntegerArray correctQuestions,
				AtomicDoubleArray percentCorret) {
			this.iteration = iteration;
			this.scAns = scAns;
			this.correctQuestions = correctQuestions;
			this.percentCorret = percentCorret;
		}

		public void processIteration() throws FileNotFoundException, IOException {
			final String correctionsFile = "/home/srmq/git/sc-answering/results/corrections-1mHolmes-iteration- "
					+ iteration + ".json.gz";
			Map<Tuple2<String, Integer>, Double> correctionsMap = readCorrections(new String[] { correctionsFile });

			 //final Iterator<Question> listQuestions = questions().iterator();
			final Iterator<Question> listQuestions = trainingQuestions();
			int correct = 0;

			{
				int i = 0;
				for (; listQuestions.hasNext(); i++) {
					Question q = listQuestions.next();
					INDArray[] distances = scAns.computeDistancesForQuestion(q, correctionsMap);
					for (int k = 0; k < distances.length; k++) {
						distances[k] = Transforms.unitVec(distances[k]);
					}
					// scAns.computeAllDistancesForQuestion(q);
					int bestIndex = (distances.length == 1) ? NDMathUtils.indexMin(distances[0])
							: scAns.minMaxRegretIndex(distances);
					log.debug("QUESTION " + (i + 1));
					log.debug("Tokens before: " + q.getTokensBefore().toString());
					log.debug("Tokens after: " + q.getTokensAfter().toString());
					log.debug("Options: " + q.getOptions().toString());
					log.debug(
							"Answer: " + (q.getCorrectIndex() + 1) + q.getCorrectLetter() + ") " + q.getCorrectWord());
					log.debug("Predicted: " + (bestIndex + 1) + (char) ('a' + bestIndex) + ") "
							+ q.getOptions().get(bestIndex));
					if (bestIndex == q.getCorrectIndex())
						correct++;
				}
				log.debug("");
				correctQuestions.set(iteration - 1, correct);
				percentCorret.set(iteration - 1, (100.0 * correct) / i);
				System.out.println(correct + " answers out of " + i + " (" + percentCorret.get(iteration - 1) + "%)");
				System.out.println("Corrections were used " + scAns._correctionsUsedTimes.get() + " times");

			}

		}

		@Override
		public void run() {
			try {
				processIteration();
			} catch (FileNotFoundException e) {
				e.printStackTrace();
				throw new IllegalStateException(e);
			} catch (IOException e) {
				e.printStackTrace();
				throw new IllegalStateException(e);
			}
		}

	}

	private static final Logger log = LoggerFactory.getLogger(SentenceCompletionAnswering.class);

	private List<EmbeddingView> embeddingViews;

	private ThreadLocal<Integer> _correctionsUsedTimes;

	public enum DistanceType {
		IDF_DECWEIGHT_EUCLIDIAN, IDF_DECWEIGHT_COSINE, DECWEIGHT_EUCLIDIAN, DECWEIGHT_COSINE, IDF_EUCLIDIAN, IDF_COSINE, EUCLIDIAN, COSINE
	}

	public SentenceCompletionAnswering(String[] embeddingFileNames, String[] tfIdfFileNames) throws IOException {
		assert (embeddingFileNames != null && tfIdfFileNames != null);
		assert (embeddingFileNames.length > 0 && tfIdfFileNames.length > 0);
		assert (embeddingFileNames.length == tfIdfFileNames.length);
		this.embeddingViews = new ArrayList<EmbeddingView>(embeddingFileNames.length);
		for (int i = 0; i < embeddingFileNames.length; i++) {
			this.embeddingViews.add(new EmbeddingView(new File(embeddingFileNames[i]), new File(tfIdfFileNames[i])));
		}
		this._correctionsUsedTimes = new ThreadLocal<Integer>() {
			@Override
			protected Integer initialValue() {
				return 0;
			}
		};
	}

	private static JsonSerializer<TfIdfInfo> serializer = new JsonSerializer<TfIdfInfo>(TfIdfInfo.class);

	private class EmbeddingView {
		private InMemoryLookupTable lookupTable;
		private VocabCache vocabCache;
		private TfIdfInfo tfIdfInfo;

		private EmbeddingView(File embeddingFile, File tfIdfFile) throws IOException {
			final Pair<InMemoryLookupTable, VocabCache> vocabPair = WordVectorSerializer.loadTxt(embeddingFile);
			this.lookupTable = vocabPair.getFirst();
			this.vocabCache = vocabPair.getSecond();
			this.tfIdfInfo = serializer.deserialize(tfIdfFile);

		}

		public INDArray computeDistancesForQuestion(Question q) {
			return this.computeDistancesForQuestion(q, DistanceType.IDF_DECWEIGHT_EUCLIDIAN, null);
		}

		public INDArray computeDistancesForQuestion(Question q, DistanceType distType,
				Map<Tuple2<String, Integer>, Double> correctedWeights) {
			INDArray distVector = Nd4j.create(
					new int[] { q.getOptions().size(), q.getTokensBefore().size() + q.getTokensAfter().size() });
			INDArray weightVector = Nd4j
					.create(new int[] { q.getTokensBefore().size() + q.getTokensAfter().size(), 1 });
			switch (distType) {
			case IDF_DECWEIGHT_COSINE:
			case IDF_DECWEIGHT_EUCLIDIAN:
			case DECWEIGHT_EUCLIDIAN:
			case DECWEIGHT_COSINE:
				decreasingWeightsVector(q, weightVector);
				break;
			default:
				weightVector.assign(1.0);
				break;
			}

			for (int option = 0; option < q.getOptions().size(); option++) {
				String opWord = q.getOptions().get(option);
				if (opWord.equals(Word2Vec.UNK)) {
					log.warn("Option word \"" + opWord + "\" is UNKNOWN");
					continue;
				}
				INDArray opVec = this.lookupTable.vector(opWord);
				if (opVec == null) {
					log.warn("Option word \"" + opWord + "\" is not in vocab");
					continue;
				}
				for (int n = 0; n < q.getTokensBefore().size(); n++) {
					String word = q.getTokensBefore().get(n);
					// log.info("Word: " + word + " has word frequency " +
					// vectorizer.getCache().wordFrequency(word) + " and
					// appeared in
					// " + vectorizer.getCache().docAppearedIn(word) + "
					// documents");
					final int dist = -(q.getTokensBefore().size() - n);
					if (this.vocabCache.containsWord(word)) {
						Double corrWeight = null;
						if (correctedWeights != null) {
							corrWeight = correctedWeights.get(new Tuple2<String, Integer>(word.toLowerCase(), dist));
							if (corrWeight != null)
								_correctionsUsedTimes.set(_correctionsUsedTimes.get() + 1);
						}
						// adjusting weights using IDF, if needed, and computing
						// distance
						final double distValue;

						INDArray wordVec = this.lookupTable.vector(word);

						switch (distType) {
						case IDF_DECWEIGHT_COSINE:
						case IDF_COSINE:
							if (corrWeight == null) {
								idfAdjustWeight(weightVector, n, word);
							} else {
								weightVector.putScalar(n, corrWeight);
							}
							distValue = -Nd4j.getExecutioner().execAndReturn(new CosineSimilarity(opVec, wordVec))
									.getFinalResult().doubleValue();
							break;
						case IDF_DECWEIGHT_EUCLIDIAN:
						case IDF_EUCLIDIAN:
							if (corrWeight == null) {
								idfAdjustWeight(weightVector, n, word);
							} else {
								weightVector.putScalar(n, corrWeight);
							}
							distValue = opVec.distance2(wordVec);
							break;
						case DECWEIGHT_EUCLIDIAN:
						case EUCLIDIAN:
							distValue = opVec.distance2(wordVec);
							if (corrWeight != null) {
								weightVector.putScalar(n, corrWeight);
							}
							break;
						case DECWEIGHT_COSINE:
						case COSINE:
							distValue = -Nd4j.getExecutioner().execAndReturn(new CosineSimilarity(opVec, wordVec))
									.getFinalResult().doubleValue();
							if (corrWeight != null) {
								weightVector.putScalar(n, corrWeight);
							}
							break;
						default:
							distValue = Int.MinValue(); // ERROR!
							throw new IllegalStateException("distValue was not set!");
						}

						distVector.slice(option).putScalar(n, distValue);

					} else {
						weightVector.putScalar(n, 0.0);
					}
				}
				for (int n = 0; n < q.getTokensAfter().size(); n++) {
					final int pos = q.getTokensBefore().size() + n;
					String word = q.getTokensAfter().get(n);
					final int dist = n + 1;
					if (this.vocabCache.containsWord(word)) {
						Double corrWeight = null;
						if (correctedWeights != null) {
							corrWeight = correctedWeights.get(new Tuple2<String, Integer>(word.toLowerCase(), dist));
							if (corrWeight != null)
								_correctionsUsedTimes.set(_correctionsUsedTimes.get() + 1);
						}
						final double distValue;

						INDArray wordVec = this.lookupTable.vector(word);

						// adjusting weights using IDF, if needed
						switch (distType) {
						case IDF_DECWEIGHT_COSINE:
						case IDF_COSINE:
							if (corrWeight == null) {
								idfAdjustWeight(weightVector, pos, word);
							} else {
								weightVector.putScalar(pos, corrWeight);
							}
							distValue = -Nd4j.getExecutioner().execAndReturn(new CosineSimilarity(opVec, wordVec))
									.getFinalResult().doubleValue();
							break;
						case IDF_DECWEIGHT_EUCLIDIAN:
						case IDF_EUCLIDIAN:
							if (corrWeight == null) {
								idfAdjustWeight(weightVector, pos, word);
							} else {
								weightVector.putScalar(pos, corrWeight);
							}
							distValue = opVec.distance2(wordVec);
							break;
						case DECWEIGHT_EUCLIDIAN:
						case EUCLIDIAN:
							distValue = opVec.distance2(wordVec);
							if (corrWeight != null) {
								weightVector.putScalar(pos, corrWeight);
							}
							break;
						case DECWEIGHT_COSINE:
						case COSINE:
							distValue = -Nd4j.getExecutioner().execAndReturn(new CosineSimilarity(opVec, wordVec))
									.getFinalResult().doubleValue();
							if (corrWeight != null) {
								weightVector.putScalar(pos, corrWeight);
							}
							break;
						default:
							distValue = Int.MinValue(); // ERROR!
							throw new IllegalStateException("distValue was not set!");
						}
						distVector.slice(option).putScalar(pos, distValue);
					} else {
						weightVector.putScalar(pos, 0.0);
					}

				}
			} // finished for all options

			INDArray scaledWeights = Transforms.unitVec(weightVector);
			INDArray distances = distVector.mmul(scaledWeights);
			return distances;
		}

		private void idfAdjustWeight(INDArray weightVector, int pos, String word) {
			weightVector.putScalar(pos, weightVector.getFloat(pos) / Math.log10(1 + tfIdfInfo.docAppearedIn(word)));
		}

		private void decreasingWeightsVector(Question q, INDArray weightVector) {
			for (int d = 1; d <= q.getTokensBefore().size(); d++) {
				final int pos = q.getTokensBefore().size() - d;
				assert pos >= 0 && pos < weightVector.length();
				weightVector.putScalar(pos, 1.0 / d);
			}
			for (int d = 1; d <= q.getTokensAfter().size(); d++) {
				final int pos = q.getTokensBefore().size() + d - 1;
				assert pos >= 0 && pos < weightVector.length();
				weightVector.putScalar(pos, 1.0 / d);
			}
		}
	}

	public INDArray[] computeDistancesForQuestion(Question q, Map<Tuple2<String, Integer>, Double> correctedWeights) {
		return this.computeDistancesForQuestion(q, DistanceType.IDF_DECWEIGHT_EUCLIDIAN, correctedWeights);
	}

	public INDArray[] computeDistancesForQuestion(Question q, DistanceType distType,
			Map<Tuple2<String, Integer>, Double> correctedWeights) {
		INDArray[] ret = new INDArray[this.embeddingViews.size()];
		for (int i = 0; i < this.embeddingViews.size(); i++) {
			ret[i] = this.embeddingViews.get(i).computeDistancesForQuestion(q, distType, correctedWeights);
		}
		return ret;
	}

	public Map<DistanceType, INDArray[]> computeAllDistancesForQuestion(Question q) {
		final Map<DistanceType, INDArray[]> ret = new HashMap<DistanceType, INDArray[]>(DistanceType.values().length);
		for (DistanceType dist : DistanceType.values()) {
			INDArray[] result = this.computeDistancesForQuestion(q, dist, null);
			ret.put(dist, result);
		}
		return ret;
	}

	public static void main(String[] args) throws Exception {
		/*
		 * SentenceCompletionAnswering scAns = new SentenceCompletionAnswering(
		 * new String[] { "WordVec-Holmes-MikolovRNN1600-StopwordsRemoved.txt",
		 * "WordVec-Holmes-HuangOriginalVectors-StopwordsRemoved.txt",
		 * "WordVec-Holmes-GoogleNews-StopwordsPresent.txt",
		 * "WordVec-Holmes-Glove-StopwordsRemoved.txt",
		 * "WordVec-Holmes-SennaOriginalVectors-StopwordsPresent.txt" }, new
		 * String[] {
		 * "TfIdfInfo-Holmes-MikolovRNN1600-StopwordsRemoved.json.gz",
		 * "TfIdfInfo-Holmes-HuangOriginalVectors-StopwordsRemoved.json.gz",
		 * "TfIdfInfo-Holmes-GoogleNews-StopwordsPresent.json.gz",
		 * "TfIdfInfo-Holmes-Glove-StopwordsRemoved.json.gz" ,
		 * "TfIdfInfo-Holmes-SennaOriginalVectors-StopwordsPresent.json.gz" });
		 */
		SentenceCompletionAnswering scAns = new SentenceCompletionAnswering(
				new String[] { "WordVec-Holmes-Glove-StopwordsRemoved.txt" },
				new String[] { "TfIdfInfo-Holmes-Glove-StopwordsRemoved.json.gz" });
		final int iterations = 30;
		AtomicIntegerArray correctQuestionsArr = new AtomicIntegerArray(iterations);
		AtomicDoubleArray percentCorrectArr = new AtomicDoubleArray(iterations);

		ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());


		List<Future<?>> futures = new ArrayList<Future<?>>(iterations);
		for (int j = 21; j <= iterations; j++) {
			futures.add(executor.submit(new IterationProcessor(j, scAns, correctQuestionsArr, percentCorrectArr)));
		}
		for (Future<?> future : futures) {
			future.get();
		}
		executor.shutdown();
		System.out.println("Iter #Correct %Correct");
		for (int i = 1; i <= iterations; i++) {
			System.out.println(i + " " + correctQuestionsArr.get(i - 1) + " " + percentCorrectArr.get(i - 1));
		}

	}

	@SuppressWarnings({ "unchecked", "rawtypes" })
	public static Map<Tuple2<String, Integer>, Double> readCorrections(String[] args) throws IOException {
		File correctionsFile;
		Map jsonMap = null;
		Map<Tuple2<String, Integer>, Double> correctionsMap = null;
		if (args.length == 1) { // argument should be corrections file: .json.gz
								// of Map<Tuple2<String, Integer>, Double>
			correctionsFile = new File(args[0]);
			assert (correctionsFile.isFile() && correctionsFile.canRead());
		} else
			correctionsFile = null;
		if (correctionsFile != null) {
			JsonSerializer jsonSerializer = new JsonSerializer(Map.class);
			jsonMap = (Map) jsonSerializer.deserialize(correctionsFile);
			correctionsMap = new HashMap<Tuple2<String, Integer>, Double>(jsonMap.size());
			for (Object jsonEntry : jsonMap.entrySet()) {
				Map.Entry<Object, Double> mapEntry = (Map.Entry<Object, Double>) jsonEntry;
				String stringKey = mapEntry.getKey().toString();
				correctionsMap.put(new Tuple2<String, Integer>(
						stringKey.substring(stringKey.indexOf('(') + 1, stringKey.indexOf(',')),
						Integer.parseInt(
								stringKey.substring(stringKey.lastIndexOf(',') + 1, stringKey.lastIndexOf(')')))),
						mapEntry.getValue());
			}
		}

		return correctionsMap;
	}

	private int minMaxRegretIndex(INDArray[] distances) {
		double minMaxRegret = WeightedRegretMinimizer.BIG_CONSTANT;
		int index = -1;
		for (int i = 0; i < distances.length; i++) {
			double[] regretsI = new double[distances[i].size(0)];
			for (int j = 0; j < regretsI.length; j++) {
				regretsI[j] = distances[i].getDouble(j);
			}
			double thisRegret = WeightedRegretMinimizer.minWeightedRegretFor(regretsI, null);
			if (thisRegret < minMaxRegret) {
				minMaxRegret = thisRegret;
				index = i;
			}
		}
		return index;
	}

	private static List<Question> questions() {
		final SentenceCompletionQuestions questions = new SentenceCompletionQuestions();
		final List<Question> listQuestions = questions.getQuestions();
		return listQuestions;
	}

	@SuppressWarnings("resource")
	private static Iterator<Question> trainingQuestions() throws FileNotFoundException, IOException {
		final String trainingQuestions = "/home/srmq/git/holmes-question-producer/trainingQuestions-1m.spark.json.gz";
		final BufferedReader bufR = new BufferedReader(
				new InputStreamReader(new GZIPInputStream(new FileInputStream(trainingQuestions))));
		Iterator<Question> questionIterator = new Iterator<Question>() {
			private BufferedReader file = bufR;
			private String nextLine = file.readLine();
			private JsonSerializer<Question> serializer = new JsonSerializer<Question>(Question.class);

			@Override
			public boolean hasNext() {
				return nextLine != null && nextLine.trim().length() > 0;
			}

			@Override
			public Question next() {
				Question q;
				try {
					q = serializer.deserialize(nextLine);
					nextLine = bufR.readLine();
				} catch (IOException e) {
					throw new IllegalStateException(e);
				}
				return q;
			}

			@Override
			protected void finalize() throws Throwable {
				super.finalize();
				file.close();
			}
		};
		return questionIterator;
	}
}

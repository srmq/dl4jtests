package br.ufpe.cin.nlp.sentence;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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

import br.ufpe.cin.nlp.sentence.base.SentenceCompletionQuestions;
import br.ufpe.cin.nlp.sentence.base.SentenceCompletionQuestions.Question;
import scala.Int;

public class SentenceCompletionAnswering {
	private static final Logger log = LoggerFactory.getLogger(SentenceCompletionAnswering.class);

	private List<EmbeddingView> embeddingViews;

	public enum DistanceType {
		IDF_DECWEIGHT_EUCLIDIAN, IDF_DECWEIGHT_COSINE, DECWEIGHT_EUCLIDIAN, DECWEIGHT_COSINE, IDF_EUCLIDIAN, IDF_COSINE, EUCLIDIAN, COSINE
	}

	public SentenceCompletionAnswering(String[] embeddingFileNames, String[] tfIdfFileNames) throws IOException {
		assert (embeddingFileNames.length == tfIdfFileNames.length);
		this.embeddingViews = new ArrayList<EmbeddingView>(embeddingFileNames.length);
		for (int i = 0; i < embeddingFileNames.length; i++) {
			this.embeddingViews.add(new EmbeddingView(new File(embeddingFileNames[i]), new File(tfIdfFileNames[i])));
		}
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
			return this.computeDistancesForQuestion(q, DistanceType.IDF_DECWEIGHT_EUCLIDIAN);
		}

		public INDArray computeDistancesForQuestion(Question q, DistanceType distType) {
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
					if (this.vocabCache.containsWord(word)) {
						INDArray wordVec = this.lookupTable.vector(word);
						
						//adjusting weights using IDF, if needed, and computing distance
						final double distValue;
						switch (distType) {
						case IDF_DECWEIGHT_COSINE:
						case IDF_COSINE:
							idfAdjustWeight(weightVector, n, word);
							distValue = -Nd4j.getExecutioner().execAndReturn(new CosineSimilarity(opVec,wordVec)).getFinalResult().doubleValue();
							break;
						case IDF_DECWEIGHT_EUCLIDIAN:
						case IDF_EUCLIDIAN:
							idfAdjustWeight(weightVector, n, word);
							distValue = opVec.distance2(wordVec);
							break;
						case DECWEIGHT_EUCLIDIAN:
						case EUCLIDIAN:
							distValue = opVec.distance2(wordVec);
							break;
						case DECWEIGHT_COSINE:
						case COSINE:
							distValue = -Nd4j.getExecutioner().execAndReturn(new CosineSimilarity(opVec,wordVec)).getFinalResult().doubleValue();
							break;
						default:
							distValue = Int.MinValue(); //ERROR!
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
					if (this.vocabCache.containsWord(word)) {
						INDArray wordVec = this.lookupTable.vector(word);
						final double distValue;
						
						//adjusting weights using IDF, if needed
						switch (distType) {
						case IDF_DECWEIGHT_COSINE:
						case IDF_COSINE:
							idfAdjustWeight(weightVector, pos, word);
							distValue = -Nd4j.getExecutioner().execAndReturn(new CosineSimilarity(opVec,wordVec)).getFinalResult().doubleValue();
							break;
						case IDF_DECWEIGHT_EUCLIDIAN:
						case IDF_EUCLIDIAN:
							idfAdjustWeight(weightVector, pos, word);
							distValue = opVec.distance2(wordVec);
							break;
						case DECWEIGHT_EUCLIDIAN:
						case EUCLIDIAN:
							distValue = opVec.distance2(wordVec);
							break;
						case DECWEIGHT_COSINE:
						case COSINE:
							distValue = -Nd4j.getExecutioner().execAndReturn(new CosineSimilarity(opVec,wordVec)).getFinalResult().doubleValue();
							break;
						default:
							distValue = Int.MinValue(); //ERROR!
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
	
	public INDArray[] computeDistancesForQuestion(Question q) {
		return this.computeDistancesForQuestion(q, DistanceType.IDF_DECWEIGHT_EUCLIDIAN);
	}

	public INDArray[] computeDistancesForQuestion(Question q, DistanceType distType) {
		INDArray[] ret = new INDArray[this.embeddingViews.size()];
		for (int i = 0; i < this.embeddingViews.size(); i++) {
			ret[i] = this.embeddingViews.get(i).computeDistancesForQuestion(q, distType);
		}
		return ret;
	}
	
	public Map<DistanceType, INDArray[]> computeAllDistancesForQuestion(Question q) {
		final Map<DistanceType, INDArray[]> ret = new HashMap<DistanceType, INDArray[]>(DistanceType.values().length);
		for (DistanceType dist : DistanceType.values()) {
			INDArray[] result = this.computeDistancesForQuestion(q, dist);
			ret.put(dist, result);
		}
		return ret;
	}

	public static void main(String[] args) throws Exception {
		SentenceCompletionAnswering scAns = new SentenceCompletionAnswering(
				new String[] {
						"WordVec-Holmes-MikolovRNN1600-StopwordsRemoved.txt",
						"WordVec-Holmes-HuangOriginalVectors-StopwordsRemoved.txt",
						"WordVec-Holmes-GoogleNews-StopwordsPresent.txt",
						"WordVec-Holmes-Glove-StopwordsRemoved.txt",
						"WordVec-Holmes-SennaOriginalVectors-StopwordsPresent.txt" },
				new String[] {
						"TfIdfInfo-Holmes-MikolovRNN1600-StopwordsRemoved.json.gz",
						"TfIdfInfo-Holmes-HuangOriginalVectors-StopwordsRemoved.json.gz",
						"TfIdfInfo-Holmes-GoogleNews-StopwordsPresent.json.gz",
						"TfIdfInfo-Holmes-Glove-StopwordsRemoved.json.gz" ,
						"TfIdfInfo-Holmes-SennaOriginalVectors-StopwordsPresent.json.gz" });
		final List<Question> listQuestions = questions();
		int correct = 0;

		for (int i = 0; i < listQuestions.size(); i++) {
			Question q = listQuestions.get(i);
			INDArray[] distances = scAns.computeDistancesForQuestion(q);
			for (int j = 0; j < distances.length; j++) {
				distances[j] = Transforms.unitVec(distances[j]);
			}
			scAns.computeAllDistancesForQuestion(q);
			int bestIndex = (distances.length == 1) ? NDMathUtils.indexMin(distances[0])
					: scAns.minMaxRegretIndex(distances);
			System.out.println("QUESTION " + (i + 1));
			System.out.println("Tokens before: " + q.getTokensBefore().toString());
			System.out.println("Tokens after: " + q.getTokensAfter().toString());
			System.out.println("Options: " + q.getOptions().toString());
			System.out
					.println("Answer: " + (q.getCorrectIndex() + 1) + q.getCorrectLetter() + ") " + q.getCorrectWord());
			System.out.println(
					"Predicted: " + (bestIndex + 1) + (char) ('a' + bestIndex) + ") " + q.getOptions().get(bestIndex));
			if (bestIndex == q.getCorrectIndex())
				correct++;
		}
		System.out.println("");
		System.out.println(correct + " answers out of " + listQuestions.size() + " ("
				+ (100.0f * correct) / listQuestions.size() + "%)");
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
}

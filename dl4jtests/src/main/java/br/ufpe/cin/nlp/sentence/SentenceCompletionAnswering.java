package br.ufpe.cin.nlp.sentence;

import java.io.File;
import java.util.List;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import br.ufpe.cin.nlp.sentence.SentenceCompletionQuestions.Question;

public class SentenceCompletionAnswering {
	private static final Logger log = LoggerFactory.getLogger(SentenceCompletionAnswering.class);	
	
	public static void main(String[] args) throws Exception {
		final File embeddingsFile = new File("/home/srmq/git/dl4jtests/dl4jtests/WordVec-Holmes-Glove-StopwordsPresent.txt");
		final File tfIdfFile = new File("TfIdfInfo-Holmes-Glove-StopwordsPresent.json.gz");
		final Pair<InMemoryLookupTable, VocabCache> vocabPair = WordVectorSerializer.loadTxt(embeddingsFile);

		log.info("Initial vocab has " + vocabPair.getSecond().numWords() + " words");

		final List<Question> listQuestions = questions();
		int correct = 0;
		
		JsonSerializer<TfIdfInfo> serializer = new JsonSerializer<TfIdfInfo>(TfIdfInfo.class);
		TfIdfInfo tfIdfInfo = serializer.deserialize(tfIdfFile);
		log.info("After vectorizer vocab has " + vocabPair.getSecond().numWords() + " words");
		
		for (int i = 0; i < listQuestions.size(); i++) {
			Question q = listQuestions.get(i);
			INDArray distances = computeDistancesForQuestion(vocabPair, tfIdfInfo, q);
			
			int minIndex = NDMathUtils.indexMin(distances);
			System.out.println("QUESTION " + (i+1));
			System.out.println("Tokens before: " + q.getTokensBefore().toString());
			System.out.println("Tokens after: " + q.getTokensAfter().toString());			
			System.out.println("Options: " + q.getOptions().toString());
			System.out.println("Answer: " + (q.getCorrectIndex() + 1) + q.getCorrectLetter() + ") " + q.getCorrectWord());
			System.out.println("Predicted: " + (minIndex+1) +  (char)('a'+minIndex) + ") " + q.getOptions().get(minIndex));
			if (minIndex == q.getCorrectIndex()) correct++;
		}
		System.out.println("");
		System.out.println(correct + " answers out of " + listQuestions.size() + " (" + (100.0f * correct)/listQuestions.size() + "%)");
	}
	
	private static INDArray computeDistancesForQuestion(Pair<InMemoryLookupTable, VocabCache> vocabPair, 
			TfIdfInfo tfIdfInfo, Question q)  {

		INDArray distVector = Nd4j
				.create(new int[] { q.getOptions().size(), q.getTokensBefore().size() + q.getTokensAfter().size() });
		INDArray weightVector = Nd4j.create(new int[] { q.getTokensBefore().size() + q.getTokensAfter().size(), 1 });
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
		for (int option = 0; option < q.getOptions().size(); option++) {
			String opWord = q.getOptions().get(option);
			if (opWord.equals(Word2Vec.UNK)) {
				log.warn("Option word \"" + opWord + "\" is UNKNOWN");
				continue;
			}
			INDArray opVec = vocabPair.getFirst().vector(opWord);
			if (opVec == null) {
				log.warn("Option word \"" + opWord + "\" is not in vocab");
				continue;
			}
			for (int n = 0; n < q.getTokensBefore().size(); n++) {
				String word = q.getTokensBefore().get(n);
				// log.info("Word: " + word + " has word frequency " +
				// vectorizer.getCache().wordFrequency(word) + " and appeared in
				// " + vectorizer.getCache().docAppearedIn(word) + "
				// documents");
				if (vocabPair.getSecond().containsWord(word)) {
					INDArray wordVec = vocabPair.getFirst().vector(word);
					distVector.slice(option).putScalar(n, opVec.distance2(wordVec));
					weightVector.putScalar(n, weightVector.getFloat(n) / Math.log10(1 + tfIdfInfo.docAppearedIn(word)));
				} else {
					weightVector.putScalar(n, 0.0);
				}
			}
			for (int n = 0; n < q.getTokensAfter().size(); n++) {
				final int pos = q.getTokensBefore().size() + n;
				String word = q.getTokensAfter().get(n);
				if (vocabPair.getSecond().containsWord(word)) {
					INDArray wordVec = vocabPair.getFirst().vector(word);
					distVector.slice(option).putScalar(pos, opVec.distance2(wordVec));
					weightVector.putScalar(pos,
							weightVector.getFloat(pos) / Math.log10(1 + tfIdfInfo.docAppearedIn(word)));
				} else {
					weightVector.putScalar(pos, 0.0);
				}

			}
		} // finished for all options

		INDArray scaledWeights = Transforms.unitVec(weightVector);
		INDArray distances = distVector.mmul(scaledWeights);
		return distances;
	}

	private static List<Question> questions() {
		final SentenceCompletionQuestions questions = new SentenceCompletionQuestions();		
		final List<Question> listQuestions = questions.getQuestions();
		return listQuestions;
	}
}

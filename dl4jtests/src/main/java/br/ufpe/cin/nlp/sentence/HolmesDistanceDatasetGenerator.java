package br.ufpe.cin.nlp.sentence;

import java.io.IOException;
import java.util.Iterator;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;

import br.ufpe.cin.nlp.sentence.SentenceCompletionAnswering.DistanceType;
import br.ufpe.cin.nlp.sentence.SentenceCompletionQuestions.Question;

public class HolmesDistanceDatasetGenerator {

	public static void main(String[] args) throws IOException {
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
		final SentenceCompletionQuestions questions = new SentenceCompletionQuestions();
		int qNum = 1;
		for (Iterator<Question> it = questions.getQuestions().iterator(); it.hasNext();) {
			Question q = it.next();
			System.out.print(qNum);
			Map<DistanceType, INDArray[]> dTypeToDistances = scAns.computeAllDistancesForQuestion(q);
			for (INDArray[] distArray : dTypeToDistances.values()) {
				for (int i = 0; i < distArray.length; i++) {
					for (int j = 0; j < distArray[i].length(); j++) {
						System.out.print(" " + distArray[i].getDouble(j));
					}
				}
			}
			System.out.println(" " + q.getCorrectIndex());
			qNum++;
		}

	}

}

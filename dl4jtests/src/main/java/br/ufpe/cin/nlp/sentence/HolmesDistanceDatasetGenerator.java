package br.ufpe.cin.nlp.sentence;

import java.io.IOException;
import java.io.PrintStream;
import java.util.Iterator;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;

import br.ufpe.cin.nlp.sentence.SentenceCompletionAnswering.DistanceType;
import br.ufpe.cin.nlp.sentence.base.SentenceCompletionQuestions;
import br.ufpe.cin.nlp.sentence.base.SentenceCompletionQuestions.Question;

public class HolmesDistanceDatasetGenerator {
	
	private Iterator<Question> questionIt;
	private SentenceCompletionAnswering scAns;
	
	public HolmesDistanceDatasetGenerator(SentenceCompletionAnswering scAns, Iterator<Question> questionIt) throws IOException {
		assert(questionIt != null);
		this.questionIt = questionIt;
		this.scAns = scAns;
	}
	
	public void printDistances(PrintStream stream) {
		int qNum = 1;
		while(questionIt.hasNext()) {
			Question q = questionIt.next();
			stream.print(qNum);
			Map<DistanceType, INDArray[]> dTypeToDistances = scAns.computeAllDistancesForQuestion(q);
			for (INDArray[] distArray : dTypeToDistances.values()) {
				for (int i = 0; i < distArray.length; i++) {
					for (int j = 0; j < distArray[i].length(); j++) {
						stream.print(" " + distArray[i].getDouble(j));
					}
				}
			}
			stream.println(" " + q.getCorrectIndex());
			qNum++;
		}
	}

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
		Iterator<Question> it = questions.getQuestions().iterator();
		HolmesDistanceDatasetGenerator generator = new HolmesDistanceDatasetGenerator(scAns, it);
		generator.printDistances(System.out);
	}

}

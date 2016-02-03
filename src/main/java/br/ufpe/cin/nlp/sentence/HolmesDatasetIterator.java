package br.ufpe.cin.nlp.sentence;

import java.io.File;

import org.nd4j.linalg.dataset.api.iterator.BaseDatasetIterator;

public class HolmesDatasetIterator extends BaseDatasetIterator {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1148858231466986809L;
	
	// (int batch, int numExamples, DataSetFetcher fetcher)
	public HolmesDatasetIterator(int batch, int numExamples) {
		super(batch, numExamples, new HolmesQuestionDataFetcher(numExamples));
	}
	
	public HolmesDatasetIterator(int batch, int numExamples, File inputFile) {
		super(batch, numExamples, new HolmesQuestionDataFetcher(numExamples, inputFile));
	}
	

}

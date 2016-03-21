package br.ufpe.cin.nlp.sentence;

import java.io.IOException;

public interface WordWeightMemory {

	/**
	 * Weight <code>word</code> that is at <code>distance</code> of the target word 
	 * @param word
	 * @param distance positive if <code>word</code> is on the right of the target word, negative otherwise.
	 * @return the weight (no normalization is applied).
	 */
	public double weightFor(String word, int distance) throws UnknownWordException, IOException;
	
	public void newWeightFor(String word, int distance, double value) throws IOException; 

}

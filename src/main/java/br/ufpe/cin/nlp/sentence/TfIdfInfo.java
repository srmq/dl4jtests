package br.ufpe.cin.nlp.sentence;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;

public class TfIdfInfo implements Serializable{
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -2101597441752357093L;
	
	private Map<String, Double> wordFrequencies;
	private Map<String, Double> docFrequencies;
	private int numDocs;
	
	@SuppressWarnings("unused")
	private TfIdfInfo() { };
	
	public TfIdfInfo(InMemoryLookupCache vocabCache) {
		wordFrequencies = createMap(vocabCache.getWordFrequencies());
		docFrequencies = createMap(vocabCache.docFrequencies);
		numDocs = vocabCache.totalNumberOfDocs();
	}

	private Map<String, Double> createMap(Counter<String> fromFrequencies) {
		Set<Entry<String, Double>> entries = fromFrequencies.entrySet();
		Map<String, Double> ret = new HashMap<String, Double>(entries.size());
		for (Entry<String, Double> entry : entries) {
			ret.put(entry.getKey(), entry.getValue());
		}
		return ret;
	}
	
	public int totalNumberDocs() {
		return this.numDocs;
	}
	
	public int docAppearedIn(String word) {
		return intFrequency(word, docFrequencies);
	}
	
	public int wordFrequency(String word) {
		return intFrequency(word, wordFrequencies);
	}
	
	private int intFrequency(String word, Map<String, Double> map) {
		final Double res;
		synchronized(map) {
			res = map.get(word);
		}
		final int ret = (res != null) ? res.intValue() : 0;
		return ret;
	}

	public Map<String, Double> getWordFrequencies() {
		return wordFrequencies;
	}

	public Map<String, Double> getDocFrequencies() {
		return docFrequencies;
	}
	
}

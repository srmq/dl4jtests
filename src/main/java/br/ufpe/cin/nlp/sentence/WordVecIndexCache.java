package br.ufpe.cin.nlp.sentence;

import java.util.Map;
import java.util.WeakHashMap;

public class WordVecIndexCache implements VectorVocab {
	private WordVecLuceneIndex wrappedIndex;
	private Map<String, float[]> cache;
	private boolean flushed;
	
	public WordVecIndexCache(WordVecLuceneIndex index) {
		this.wrappedIndex = index;
		allocateCache();
	}

	private void allocateCache() {
		this.cache = new WeakHashMap<String, float[]>(50000);
		flushed = false;
	}
	
	public synchronized float[] embeddingFor(String word) {
		if (flushed) allocateCache();
		float[] ret = cache.get(word);
		if (ret == null) {
			ret = wrappedIndex.embeddingFor(word);
			if (ret != null) {
				cache.put(word, ret);
			}
		}
		
		return ret;
	}
	
	public synchronized boolean contains(String word) {
		if (flushed) allocateCache();
		boolean ret = false;
		if (cache.containsKey(word)) {
			ret = true;
		} else {
			final float[] embed = wrappedIndex.embeddingFor(word);
			if (embed != null) {
				cache.put(word, embed);
				ret = true;
			}
		}
		return ret;
	}
	
	public synchronized void flush() {
		flushed = true;
		this.cache = null;
	}

	@Override
	public int numWords() {
		return wrappedIndex.numWords();
	}

	@Override
	public int embedSize() {
		return wrappedIndex.embedSize();
	}
}

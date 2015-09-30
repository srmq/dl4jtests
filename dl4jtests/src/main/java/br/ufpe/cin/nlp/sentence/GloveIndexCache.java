package br.ufpe.cin.nlp.sentence;

import java.util.Map;
import java.util.WeakHashMap;

public class GloveIndexCache implements GloveVocab {
	private GloveLuceneIndex wrappedIndex;
	private Map<String, float[]> cache;
	
	public GloveIndexCache(GloveLuceneIndex index) {
		this.wrappedIndex = index;
		this.cache = new WeakHashMap<String, float[]>();
	}
	
	public float[] embeddingFor(String word) {
		float[] ret = cache.get(word);
		if (ret == null) {
			ret = wrappedIndex.embeddingFor(word);
			if (ret != null) {
				cache.put(word, ret);
			}
		}
		
		return ret;
	}
	
	public boolean contains(String word) {
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

	@Override
	public int numWords() {
		return wrappedIndex.numWords();
	}

	@Override
	public int embedSize() {
		return wrappedIndex.embedSize();
	}
}

package br.ufpe.cin.nlp.sentence;

public interface GloveVocab {
	public float[] embeddingFor(String word);
	public boolean contains(String word);
	public int numWords();
	public int embedSize();
}

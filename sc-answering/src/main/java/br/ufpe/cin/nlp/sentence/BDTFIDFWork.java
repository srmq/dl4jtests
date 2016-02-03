package br.ufpe.cin.nlp.sentence;

import org.deeplearning4j.models.word2vec.VocabWord;

public class BDTFIDFWork {
	private VocabWord vocab;
	
	public BDTFIDFWork(VocabWord vocab) {
		this.vocab = vocab;
	}

	public VocabWord getVocab() {
		return vocab;
	}

	public void setVocab(VocabWord vocab) {
		this.vocab = vocab;
	}

}

package br.ufpe.cin.nlp.sentence;

import java.util.Locale;

import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;

public class VectorVocabTokenPreprocessor implements TokenPreProcess {
	
	private VectorVocab vocab;
	
	public VectorVocabTokenPreprocessor(VectorVocab vocab) {
		this.vocab = vocab;
	}

	@Override
	public String preProcess(String token) {
		String ret = token;
		if (!vocab.contains(token.toLowerCase(Locale.ENGLISH))) {
			ret = Word2Vec.UNK;
		}
		return ret;
	}

}

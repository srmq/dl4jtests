package br.ufpe.cin.nlp.sentence;

import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.NGramTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

public class NGramTokenizerTest {

	public static void main(String[] args) {
        String toTokenize = "Mary had a little lamb.";
        TokenizerFactory factory = new NGramTokenizerFactory(new DefaultTokenizerFactory(), 3, 3);
        Tokenizer tokenizer = factory.create(toTokenize);
        while (tokenizer.hasMoreTokens()) {
            System.out.println(tokenizer.nextToken());
        }
		

	}

}

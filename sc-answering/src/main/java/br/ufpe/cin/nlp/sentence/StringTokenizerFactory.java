package br.ufpe.cin.nlp.sentence;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

public class StringTokenizerFactory implements TokenizerFactory {
	
	private String delimiters;
	
	private TokenPreProcess preProcessor;
	
	public StringTokenizerFactory(String delimiters) {
		this.delimiters = delimiters;
	}
	
	public StringTokenizerFactory() {
		this.delimiters = " \t\n\r\f,.:;?![]'\"(){}";
	}
	
	private class CustomStringTokenizer implements Tokenizer {
		private StringTokenizer tokenizer;
		
		private TokenPreProcess myProcessor;
		
		public CustomStringTokenizer(String stringToTokenize, TokenPreProcess processor) {
			this.tokenizer = new StringTokenizer(stringToTokenize, delimiters);
			this.myProcessor = processor;
		}
		
		@Override
		public boolean hasMoreTokens() {
			return this.tokenizer.hasMoreTokens();
		}

		@Override
		public int countTokens() {
			return this.tokenizer.countTokens();
		}

		@Override
		public String nextToken() {
			String ret = this.tokenizer.nextToken();
			if (myProcessor != null) {
				ret = myProcessor.preProcess(ret);
			}
			return ret;
		}

		@Override
		public List<String> getTokens() {
			List<String> ret = new ArrayList<String>(countTokens());
			while(hasMoreTokens()) {
				ret.add(nextToken());
			}
			return ret;
		}

		@Override
		public void setTokenPreProcessor(TokenPreProcess tokenPreProcessor) {
			this.myProcessor = tokenPreProcessor;
		}
		
	}

	@Override
	public Tokenizer create(String toTokenize) {
		return new CustomStringTokenizer(toTokenize, this.preProcessor);
	}

	@Override
	public Tokenizer create(InputStream toTokenize) {
		throw new UnsupportedOperationException("Tokenization of streams is not supported by this factory");
	}

	@Override
	public void setTokenPreProcessor(TokenPreProcess preProcessor) {
		this.preProcessor = preProcessor;
	}

	public String getDelimiters() {
		return delimiters;
	}

	public void setDelimiters(String delimiters) {
		this.delimiters = delimiters;
	}

}

package br.ufpe.cin.nlp.sentence;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

import org.deeplearning4j.bagofwords.vectorizer.TextVectorizer;
import org.deeplearning4j.bagofwords.vectorizer.TfidfVectorizer;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.invertedindex.InvertedIndex;
import org.deeplearning4j.text.invertedindex.LuceneInvertedIndex;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.sentenceiterator.UimaSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SentenceCompletionGoogleNewsWriter {
	
	private static final Logger log = LoggerFactory.getLogger(SentenceCompletionGoogleNewsWriter.class);	

	public static void main(String[] args) throws Exception {
		SentenceIterator iterator = UimaSentenceIterator
				.createWithPath("/home/srmq/devel/Holmes_Sem_Cabecalho_e_Rodape");
		iterator.setPreProcessor(new SentencePreProcessor() {
			/**
			 * 
			 */
			private static final long serialVersionUID = -7192858487052715236L;

			@Override
			public String preProcess(String sentence) {
				return sentence.toLowerCase(Locale.ENGLISH);
			}
		});

		InvertedIndex index = new LuceneInvertedIndex(null, false, "vector-index-googleNews");
		TokenizerFactory tokenFactory = new UimaTokenizerFactory();
		VectorVocab vectorVocab;
		{
			WordVecLuceneIndex wordVecIndex = new WordVecLuceneIndex(new
					File("/home/srmq/devel/word2vec/trunk/GoogleNews-vectors-negative300.txt.gz"), "googleNews-300d", true);
			vectorVocab = new WordVecIndexCache(wordVecIndex);
		}
		{ // in order to change words not in vector file to UNK
			VectorVocabTokenPreprocessor tokenPre = new VectorVocabTokenPreprocessor(vectorVocab);
			tokenFactory.setTokenPreProcessor(tokenPre);
		}
		
		final VocabCache cache = new InMemoryLookupCache(true);
		
		//List<String> stopWrds = StopWords.getStopWords();
		List<String> stopWrds = Arrays.asList(new String[]{""});
		
		TextVectorizer textVectorizer = new TfidfVectorizer.Builder().index(index).iterate(iterator).minWords(1)
				.stopWords(stopWrds).cache(cache).tokenize(tokenFactory).build();
		textVectorizer.fit();

		InMemoryLookupTable table =  SentenceCompletionWithWordVec.readLookupTableForCorpus(vectorVocab, textVectorizer);
        WordVectorSerializer.writeWordVectors(table, (InMemoryLookupCache) cache, "GoogleNews-Holmes-NOSTOPW-notraining.txt");		
		
		
	}

}

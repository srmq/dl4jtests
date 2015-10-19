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
import org.deeplearning4j.text.stopwords.StopWords;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;

public class HolmesVectorizer {
	
	public void vectorize(File embeddingsFile, String indexVectorizerPath, String indexEmbeddingsPath, boolean removeStopWords, File toFileTfIdfInfo, File toFileEmbeddings, boolean gzipCompression) throws Exception {
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

		InvertedIndex index = new LuceneInvertedIndex(null, false, indexVectorizerPath);
		TokenizerFactory tokenFactory = new UimaTokenizerFactory();
		VectorVocab vectorVocab;
		{
			WordVecLuceneIndex wordVecIndex = new WordVecLuceneIndex(embeddingsFile, indexEmbeddingsPath, true);
			vectorVocab = new WordVecIndexCache(wordVecIndex);
		}
		{ // in order to change words not in vector file to UNK
			VectorVocabTokenPreprocessor tokenPre = new VectorVocabTokenPreprocessor(vectorVocab);
			tokenFactory.setTokenPreProcessor(tokenPre);
		}
		
		final VocabCache cache = new InMemoryLookupCache(true);
		
		List<String> stopWrds;
		if (removeStopWords) {
			stopWrds = StopWords.getStopWords();
		} else {
			stopWrds = Arrays.asList(new String[]{""});
		}
		
		TextVectorizer textVectorizer = new TfidfVectorizer.Builder().index(index).iterate(iterator).minWords(1)
				.stopWords(stopWrds).cache(cache).tokenize(tokenFactory).build();
		textVectorizer.fit();

		InMemoryLookupTable table =  SentenceCompletionWithWordVec.readLookupTableForCorpus(vectorVocab, textVectorizer);
        WordVectorSerializer.writeWordVectors(table, (InMemoryLookupCache) cache, toFileEmbeddings.getName());		
		
		
		JsonSerializer<TfIdfInfo> serializer = new JsonSerializer<TfIdfInfo>(TfIdfInfo.class);
		TfIdfInfo tfIdfInfo = new TfIdfInfo((InMemoryLookupCache)textVectorizer.vocab());		
		serializer.serialize(tfIdfInfo, toFileTfIdfInfo, gzipCompression);
		
	}
	
	public static void main(String[] args) throws Exception {
		HolmesVectorizer vec = new HolmesVectorizer();
		
		vec.vectorize(new File("/home/srmq/devel/glove/glove.6B.300d-WITHHEADERLINE.txt.gz"), "glove-index-StopwordsRemoved", "glove-6B-300d-StopwordsRemoved", true, new File("TfIdfInfo-Holmes-Glove-StopwordsRemoved.json.gz"), new File("WordVec-Holmes-Glove-StopwordsRemoved.txt"), true);
		vec.vectorize(new File("/home/srmq/devel/glove/glove.6B.300d-WITHHEADERLINE.txt.gz"), "glove-index-StopwordsPresent", "glove-6B-300d-StopwordsPresent", false, new File("TfIdfInfo-Holmes-Glove-StopwordsPresent.json.gz"), new File("WordVec-Holmes-Glove-StopwordsPresent.txt"), true);
		
		
		vec.vectorize(new
				File("/home/srmq/devel/word2vec/trunk/GoogleNews-vectors-negative300.txt.gz"), "vector-index-googleNews-StopwordsRemoved", "googleNews-300d-StopwordsRemoved", true, new File("TfIdfInfo-Holmes-GoogleNews-StopwordsRemoved.json.gz"), new File("WordVec-Holmes-GoogleNews-StopwordsRemoved.txt"), true);
		vec.vectorize(new
				File("/home/srmq/devel/word2vec/trunk/GoogleNews-vectors-negative300.txt.gz"), "vector-index-googleNews-StopwordsPresent", "googleNews-300d-StopwordsPresent", false, new File("TfIdfInfo-Holmes-GoogleNews-StopwordsPresent.json.gz"), new File("WordVec-Holmes-GoogleNews-StopwordsPresent.txt"), true);
	}

}

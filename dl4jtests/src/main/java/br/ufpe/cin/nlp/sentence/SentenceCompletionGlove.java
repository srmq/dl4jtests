package br.ufpe.cin.nlp.sentence;

import java.io.File;
import java.io.IOException;
import java.util.Locale;

import org.deeplearning4j.bagofwords.vectorizer.TextVectorizer;
import org.deeplearning4j.bagofwords.vectorizer.TfidfVectorizer;
import org.deeplearning4j.models.glove.Glove;
import org.deeplearning4j.models.glove.GloveWeightLookupTable;
import org.deeplearning4j.models.word2vec.Word2Vec;
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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SentenceCompletionGlove {

	private static final Logger log = LoggerFactory.getLogger(SentenceCompletionGlove.class);

	public static Glove readGloveTextModelForCorpus(GloveVocab vocabGlove, SentenceIterator iter, TextVectorizer vectorizer,
			boolean symmetric, boolean shuffle, int minWordFrequency, int iterations, double learningRate) throws IOException, NumberFormatException {

		final int layerSize = vocabGlove.embedSize();

		

		final VocabCache corpusVocab = vectorizer.vocab();

		INDArray syn0 = Nd4j.create(corpusVocab.numWords(), layerSize);
		Random rng = Nd4j.getRandom();
		{ // init embeddings for UNKNOWN word to random values
			INDArray randUnk = Nd4j.rand(1, layerSize, rng).subi(0.5).divi(layerSize);
			final int idx = corpusVocab.indexOf(Word2Vec.UNK);
			syn0.slice(idx).assign(randUnk);
		}
		
		for (String word : corpusVocab.words()) {
			final int index = corpusVocab.indexOf(word);
			final float[] embed = vocabGlove.embeddingFor(word);
			if (embed != null) {
				syn0.slice(index).assign(Transforms.unitVec(Nd4j.create(embed)));
			} else {
				log.warn("Embedding for word \"{}\" was not found in GloveIndex", word);
			}
		}

		log.info("Total words in vocabulary: {}", corpusVocab.numWords());

		GloveWeightLookupTable gloveTable = (new GloveWeightLookupTable.Builder().gen(rng).cache(corpusVocab)
				.vectorLength(layerSize).lr(learningRate)).build();
		gloveTable.setSyn0(syn0);
		gloveTable.resetWeights(false);
		gloveTable.setSyn1(Nd4j.create(syn0.shape()));

		Glove ret = (new Glove.Builder().cache(corpusVocab).iterations(iterations).iterate(iter).layerSize(layerSize)
				.learningRate(learningRate).minWordFrequency(minWordFrequency).shuffle(shuffle).symmetric(symmetric)
				.weights(gloveTable).vectorizer(vectorizer)).build();
		return ret;
	}

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

		InvertedIndex index = new LuceneInvertedIndex(null, false, "glove-index");
		TokenizerFactory tokenFactory = new UimaTokenizerFactory();
		GloveVocab gloveVocab;
		{
			GloveLuceneIndex gloveIndex = new GloveLuceneIndex(new
					File("/home/srmq/devel/glove/glove.6B.300d-WITHHEADERLINE.txt.gz"), "glove-6B-300d", true);
			gloveVocab = new GloveIndexCache(gloveIndex);
		}
		{ // in order to change words not in glove file to UNK
			GloveVocabTokenPreprocessor tokenPre = new GloveVocabTokenPreprocessor(gloveVocab);
			tokenFactory.setTokenPreProcessor(tokenPre);
		}
		
		final VocabCache cache = new InMemoryLookupCache(true);
		
		TextVectorizer textVectorizer = new TfidfVectorizer.Builder().index(index).iterate(iterator).minWords(1)
				.stopWords(StopWords.getStopWords()).cache(cache).tokenize(tokenFactory).build();
		textVectorizer.fit();
		
		Glove gloveModel = SentenceCompletionGlove.readGloveTextModelForCorpus(gloveVocab, iterator, textVectorizer, true, true, 1, 100, 0.05);
		
		((GloveIndexCache)gloveVocab).flush();
		
		gloveModel.fit();
		

	}

}

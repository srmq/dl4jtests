package br.ufpe.cin.nlp.sentence;

import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicInteger;

import org.deeplearning4j.bagofwords.vectorizer.TextVectorizer;
import org.deeplearning4j.bagofwords.vectorizer.TfidfVectorizer;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.glove.CoOccurrences;
import org.deeplearning4j.models.glove.Glove;
import org.deeplearning4j.models.glove.GloveWeightLookupTable;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.invertedindex.InvertedIndex;
import org.deeplearning4j.text.invertedindex.LuceneInvertedIndex;
import org.deeplearning4j.text.movingwindow.Util;
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

public class SentenceCompletionWithWordVec {

	private static final Logger log = LoggerFactory.getLogger(SentenceCompletionWithWordVec.class);

	public static Glove readGloveTextModelForCorpus(VectorVocab vectorVocab, SentenceIterator iter, TextVectorizer vectorizer,
			boolean symmetric, boolean shuffle, int minWordFrequency, int iterations, double learningRate) throws IOException, NumberFormatException {

		final int layerSize = vectorVocab.embedSize();

		

		final VocabCache corpusVocab = vectorizer.vocab();

		Random rng = Nd4j.getRandom();
		INDArray syn0 = loadEmbeddings(vectorVocab, layerSize, corpusVocab, rng);

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
	
	public static InMemoryLookupTable readLookupTableForCorpus(VectorVocab vectorVocab, TextVectorizer vectorizer) throws IOException, NumberFormatException {
		final int layerSize = vectorVocab.embedSize();

		

		final VocabCache corpusVocab = vectorizer.vocab();

		Random rng = Nd4j.getRandom();
		INDArray syn0 = loadEmbeddings(vectorVocab, layerSize, corpusVocab, rng);

		log.info("Total words in vocabulary: {}", corpusVocab.numWords());
		InMemoryLookupTable lookupTable = (InMemoryLookupTable)(new InMemoryLookupTable.Builder()).cache(corpusVocab).vectorLength(layerSize).gen(rng).build();
		lookupTable.setSyn0(syn0);
		lookupTable.resetWeights(false);
		
		return lookupTable;
	}

	private static INDArray loadEmbeddings(VectorVocab vectorVocab, final int layerSize, final VocabCache corpusVocab,
			Random rng) {
		INDArray syn0 = Nd4j.create(corpusVocab.numWords(), layerSize);
		{ // init embeddings for UNKNOWN word to random values
			INDArray randUnk = Nd4j.rand(1, layerSize, rng).subi(0.5).divi(layerSize);
			final int idx = corpusVocab.indexOf(Word2Vec.UNK);
			syn0.slice(idx).assign(randUnk);
		}
		
		for (String word : corpusVocab.words()) {
			final int index = corpusVocab.indexOf(word);
			final float[] embed = vectorVocab.embeddingFor(word);
			if (embed != null) {
				syn0.slice(index).assign(Transforms.unitVec(Nd4j.create(embed)));
			} else {
				log.warn("Embedding for word \"{}\" was not found in VectorIndex", word);
			}
		}
		return syn0;
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

		InvertedIndex index = new LuceneInvertedIndex(null, false, "vector-index");
		TokenizerFactory tokenFactory = new UimaTokenizerFactory();
		VectorVocab vectorVocab;
		{
			WordVecLuceneIndex wordVecIndex = new WordVecLuceneIndex(new
					File("/home/srmq/devel/glove/glove.6B.300d-WITHHEADERLINE.txt.gz"), "glove-6B-300d", true);
			vectorVocab = new WordVecIndexCache(wordVecIndex);
		}
		{ // in order to change words not in vector file to UNK
			VectorVocabTokenPreprocessor tokenPre = new VectorVocabTokenPreprocessor(vectorVocab);
			tokenFactory.setTokenPreProcessor(tokenPre);
		}
		
		final VocabCache cache = new InMemoryLookupCache(true);
		
		List<String> stopWrds = StopWords.getStopWords();
		//List<String> stopWrds = Arrays.asList(new String[]{""});
		
		TextVectorizer textVectorizer = new TfidfVectorizer.Builder().index(index).iterate(iterator).minWords(1)
				.stopWords(stopWrds).cache(cache).tokenize(tokenFactory).build();
		textVectorizer.fit();
		
		final boolean symmetric = true;
		final int windowSize = 5;
		final int iterations = 20;
		
		Glove gloveModel = SentenceCompletionWithWordVec.readGloveTextModelForCorpus(vectorVocab, iterator, textVectorizer, symmetric, true, 1, iterations, 0.05);
		log.info("Finished loading glove vectors");

		
		iterator.reset();
        CoOccurrences coOccurrences = new CoOccurrences.Builder()
                .cache(gloveModel.vocab()).iterate(iterator).symmetric(symmetric)
                .tokenizer(tokenFactory).windowSize(windowSize)
                .build();

        coOccurrences.fit();
        log.info("Finished coOccurrences fit()");
		((WordVecIndexCache)vectorVocab).flush();
		
		gloveModel.setCoOccurrences(coOccurrences);
		final List<Pair<String,String>> pairList = coOccurrences.coOccurrenceList();
		Collections.shuffle(pairList,new java.util.Random());
		
        final AtomicInteger countUp = new AtomicInteger(0);
        final Counter<Integer> errorPerIteration = Util.parallelCounter();
        log.info("Processing # of co occurrences " + coOccurrences.numCoOccurrences());
        WordVectorSerializer.writeWordVectors(gloveModel.lookupTable(), (InMemoryLookupCache) cache, "Glove-Holmes-NOSTOPW-notraining.txt");        
        for(int i = 0; i < iterations; i++) {
            final AtomicInteger processed = new AtomicInteger(coOccurrences.numCoOccurrences());
            gloveModel.doIteration(i, pairList, errorPerIteration, processed, countUp);
            log.info("Processed " + countUp.doubleValue() + " out of " + (pairList.size() * iterations) + " error was " + errorPerIteration.getCount(i));
            WordVectorSerializer.writeWordVectors(gloveModel.lookupTable(), (InMemoryLookupCache) cache, "Glove-Holmes-NOSTOPW-iteration-" + i + ".txt");            
        }

	}

}

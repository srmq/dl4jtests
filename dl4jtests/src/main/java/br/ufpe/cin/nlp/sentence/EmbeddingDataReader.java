package br.ufpe.cin.nlp.sentence;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public abstract class EmbeddingDataReader {
	protected File vocabFile;
	protected File embeddingFile;
	
	public EmbeddingDataReader(File vocabFile, File embeddingFile) {
		this.vocabFile = vocabFile;
		this.embeddingFile = embeddingFile;
	}

	public Pair<InMemoryLookupTable, VocabCache> loadOriginalEmbeddings() throws IOException {
	    VocabCache cache = new InMemoryLookupCache();
	    
	    List<INDArray> arrays = populateCacheAndLoadEmbeddings(cache);
	    
	    INDArray syn = Nd4j.create(new int[] { arrays.size(), arrays.get(0).columns() });
	    for (int i = 0; i < syn.rows(); i++) {
	        syn.putRow(i, arrays.get(i));
	    }

	    InMemoryLookupTable lookupTable = (InMemoryLookupTable) new InMemoryLookupTable.Builder()
	            .vectorLength(arrays.get(0).columns())
	            .useAdaGrad(false).cache(cache)
	            .build();
	    Nd4j.clearNans(syn);
	    lookupTable.setSyn0(syn);
	    
	    return new Pair<>(lookupTable, cache);
	}
	
	private List<INDArray> populateCacheAndLoadEmbeddings(VocabCache cache) throws IOException {
		List<INDArray> arrays;
		final List<String> vocabWords = loadVocab();
		final Pair<double[][], Boolean> embedPair = loadEmbeddings(vocabWords.size());
		final double[][] embeddings = embedPair.getFirst();
		final boolean transposed = embedPair.getSecond();
		arrays = new ArrayList<>(vocabWords.size());
		assert(vocabWords.size() == (transposed ? embeddings[0].length : embeddings.length));
        final int embedSize = transposed ? embeddings.length : embeddings[0].length;
	    for (int n = 0; n < vocabWords.size(); n++) {
	    	final String word = vocabWords.get(n);
	        final VocabWord word1 = new VocabWord(1.0, word);
	        cache.addToken(word1);
	        cache.addWordToIndex(cache.numWords(), word);
	        word1.setIndex(cache.numWords());
	        cache.putVocabWord(word);
	        INDArray row = Nd4j.create(Nd4j.createBuffer(embedSize));
	        for (int i = 0; i < embedSize; i++) {
	            row.putScalar(i, transposed ? embeddings[i][n] : embeddings[n][i]);
	        }
	        arrays.add(row);
		}
		return arrays;
	}
	
	protected abstract List<String> loadVocab() throws IOException;
	
	protected abstract Pair<double[][], Boolean> loadEmbeddings(int numWords) throws IOException;
	
}

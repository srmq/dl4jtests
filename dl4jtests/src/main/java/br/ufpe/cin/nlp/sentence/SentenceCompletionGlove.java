package br.ufpe.cin.nlp.sentence;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.zip.GZIPInputStream;

import org.apache.commons.compress.compressors.gzip.GzipUtils;
import org.deeplearning4j.models.glove.Glove;
import org.deeplearning4j.models.glove.GloveWeightLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SentenceCompletionGlove {
	
    private static final Logger log = LoggerFactory.getLogger(SentenceCompletionGlove.class);
    
    private SentenceIterator iter;
    
    public SentenceCompletionGlove(SentenceIterator sentenceIterator) {
    	this.iter = sentenceIterator;
    }
    
	public static Glove readFullGloveTextModel(File modelFile, SentenceIterator iter,
			boolean symmetric, boolean shuffle, int minWordFrequency, int iterations, double learningRate,
			int layerSize) throws IOException, NumberFormatException {
    	
		final BufferedReader reader = getTextFileModelReader(modelFile);
		
        String line = reader.readLine();
        String[] initial = line.split(" ");
        final int words = Integer.parseInt(initial[0]);
        final int fileLayerSize = Integer.parseInt(initial[1]);
        
        assert (layerSize == fileLayerSize);
        
        
    	VocabCache cache = new InMemoryLookupCache(true);
        INDArray syn0 = Nd4j.create(words + 1, layerSize);
        Random rng = Nd4j.getRandom();
        { // init embeddings for UNKNOWN word to random values
	        INDArray randUnk = Nd4j.rand(1, layerSize, rng).subi(0.5).divi(layerSize);
	        final int idx = cache.indexOf(Word2Vec.UNK);
	        syn0.slice(idx).assign(randUnk);        
        }
    	
        int index = cache.numWords();
        while ((line = reader.readLine()) != null) {
            String[] split = line.split(" ");
            assert split.length == layerSize + 1;
            String word = split[0];
            float[] vector = new float[split.length - 1];
            for (int i = 1; i < split.length; i++) {
                vector[i - 1] = Float.parseFloat(split[i]);
            }
			syn0.putRow(index, Transforms.unitVec(Nd4j.create(vector)));
            cache.addWordToIndex(index, word);
            cache.addToken(new VocabWord(1, word));
            cache.putVocabWord(word);
            index++;
        }
		reader.close();

		log.info("Total words in vocabulary: {}", cache.numWords());
		
		GloveWeightLookupTable gloveTable = (new GloveWeightLookupTable.Builder().gen(rng).cache(cache)
				.vectorLength(layerSize).lr(learningRate)).build();
		gloveTable.setSyn0(syn0);
		gloveTable.resetWeights(false);
		gloveTable.setSyn1(Nd4j.create(syn0.shape()));
		
		Glove ret = (new Glove.Builder().cache(cache).iterations(iterations).iterate(iter).layerSize(layerSize)
				.learningRate(learningRate).minWordFrequency(minWordFrequency).shuffle(shuffle).symmetric(symmetric))
						.build();
    	return ret; 
    }    
	

	private static BufferedReader getTextFileModelReader(File modelFile) throws IOException, FileNotFoundException {
		final BufferedReader reader = new BufferedReader(new InputStreamReader(
                GzipUtils.isCompressedFilename(modelFile.getName())
                        ? new GZIPInputStream(new FileInputStream(modelFile))
                        : new FileInputStream(modelFile), "UTF-8"));
		return reader;
	}

	public static void main(String[] args) throws IOException {
		//FIXME todo
		//SentenceCompletionWords words = new SentenceCompletionWords(); 
		//Glove gloveModel = SentenceCompletionGlove.readFullGloveTextModel(new File("/home/srmq/devel/glove/glove.6B.300d-WITHHEADERLINE.txt.gz"), );
		//log.info("Vector model loaded with vocabulary size: {}", vecModel.vocab().numWords());
	}

}

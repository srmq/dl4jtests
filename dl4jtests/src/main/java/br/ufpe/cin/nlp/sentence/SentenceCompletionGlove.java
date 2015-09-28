package br.ufpe.cin.nlp.sentence;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.zip.GZIPInputStream;

import org.apache.commons.compress.compressors.gzip.GzipUtils;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SentenceCompletionGlove {
	
    private static final Logger log = LoggerFactory.getLogger(SentenceCompletionGlove.class);
	
	private static Word2Vec readTextModel(File modelFile, Set<String> sentenceWords)
            throws IOException, NumberFormatException {

		final BufferedReader reader = getTextFileModelReader(modelFile);
		
        String line = reader.readLine();
        String[] initial = line.split(" ");
        final int words = Integer.parseInt(initial[0]);
        final int layerSize = Integer.parseInt(initial[1]);
        
        List<String> wordsWithVectors = new ArrayList<String>(sentenceWords.size());
        List<float[]> embeddings = new ArrayList<float[]>(sentenceWords.size());
        while ((line = reader.readLine()) != null) {
            String[] split = line.split(" ");
            assert split.length == layerSize + 1;
            String word = split[0];
            if (sentenceWords.contains(word)) {
            	wordsWithVectors.add(word);
                float[] vector = new float[split.length - 1];
                for (int i = 1; i < split.length; i++) {
                    vector[i - 1] = Float.parseFloat(split[i]);
                }
                embeddings.add(vector);
            }
        }
		reader.close();
		final int newWords = wordsWithVectors.size();
		log.info("Total words in vocabulary: {}. Will use {}.", words, newWords);
		
		INDArray syn0 = Nd4j.create(newWords, layerSize);
		VocabCache cache = new InMemoryLookupCache(false);
		for (int i = 0; i < newWords; i++) {
			final String word = wordsWithVectors.get(i);
			final float[] embed = embeddings.get(i);
			syn0.putRow(i, Transforms.unitVec(Nd4j.create(embed)));
            cache.addWordToIndex(cache.numWords(), word);
            cache.addToken(new VocabWord(1, word));
            cache.putVocabWord(word);
		}
		InMemoryLookupTable lookupTable = (InMemoryLookupTable) new InMemoryLookupTable.Builder().
                cache(cache).vectorLength(layerSize).build();
		lookupTable.setSyn0(syn0);
		
		Word2Vec ret = new Word2Vec();
        ret.setVocab(cache);
        ret.setLookupTable(lookupTable);

		
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
		SentenceCompletionWords words = new SentenceCompletionWords(); 
		Word2Vec vecModel = SentenceCompletionGlove.readTextModel(new File("/home/srmq/devel/glove/glove.6B.300d-WITHHEADERLINE.txt.gz"), words.words());
		log.info("Vector model loaded with vocabulary size: {}", vecModel.vocab().numWords());
	}

}

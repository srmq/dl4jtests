package br.ufpe.cin.nlp.sentence;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLChar;
import com.jmatio.types.MLDouble;

public class HuangMatDataReader extends EmbeddingDataReader {

	public HuangMatDataReader(File vocabFile, File embeddingFile) {
		super(vocabFile, embeddingFile);
	}

	public static void main(String[] args) throws IOException {

		HuangMatDataReader dataReader = new HuangMatDataReader(
				new File("/home/srmq/Dropbox/CIn/research/textmining/Huang/wordrep/vocab.mat"),
				new File("/home/srmq/Dropbox/CIn/research/textmining/Huang/wordrep/wordreps_orig.mat"));
		Pair<InMemoryLookupTable, VocabCache> embeds = dataReader.loadOriginalEmbeddings();
		WordVectorSerializer.writeWordVectors(embeds.getFirst(), (InMemoryLookupCache) embeds.getSecond(),
				"Huang-OriginalVectors.txt");

		System.out.println("all done");

	}

	@Override
	protected List<String> loadVocab() throws IOException {
		MatFileReader matReader = new MatFileReader(vocabFile);
		Map<String, MLArray> data = matReader.getContent();
		MLArray vocab = data.get("vocab");
		assert (vocab.isCell());
		List<MLArray> wordCells = ((MLCell) vocab).cells();
		List<String> vocabWords = new ArrayList<String>(wordCells.size());
		for (MLArray mlArray : wordCells) {
			assert (mlArray.isChar());
			String word = ((MLChar) mlArray).getString(0);
			vocabWords.add(word);
		}
		return vocabWords;
	}

	@Override
	protected Pair<double[][], Boolean> loadEmbeddings(int numWords) throws IOException {
		MatFileReader matReader = new MatFileReader(embeddingFile);
		Map<String, MLArray> data = matReader.getContent();
		MLArray originalEmbeddings = data.get("oWe");
		assert (originalEmbeddings.isDouble());
		double[][] embeddings = ((MLDouble) originalEmbeddings).getArray();
		//true because the embeddings array is transposed
		return new Pair<double[][], Boolean>(embeddings, true);
	}

}

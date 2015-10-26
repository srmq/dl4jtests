package br.ufpe.cin.nlp.sentence;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;

public class SennaDataReader extends EmbeddingDataReader {
	
	public SennaDataReader(File vocabFile, File embeddingFile) {
		super(vocabFile, embeddingFile);
	}

	@Override
	protected List<String> loadVocab() throws IOException {
		BufferedReader buf = new BufferedReader(new InputStreamReader(new FileInputStream(this.vocabFile), "UTF-8"));
		LinkedList<String> vocabWords = new LinkedList<String>();
		String line;
		while ((line = buf.readLine()) != null) {
			line = line.trim();
			if (line.length() == 0) continue;
			vocabWords.add(line);
		}
		buf.close();
		return new ArrayList<String>(vocabWords);
	}

	@Override
	protected Pair<double[][], Boolean> loadEmbeddings(int numWords) throws IOException {
		double ret[][] = new double[numWords][];
		int ind = 0;
		BufferedReader buf = new BufferedReader(new InputStreamReader(new FileInputStream(this.embeddingFile), "UTF-8"));
		String line;
		while ((line = buf.readLine()) != null) {
			line = line.trim();
			if (line.length() == 0) continue;
			String values[] = line.split("\\s+");
			ret[ind] = new double[values.length];
			for (int i = 0; i < values.length; i++) {
				ret[ind][i] = Double.parseDouble(values[i]);
			}
			ind++;
		}
		buf.close();
		//false because the embeddings array is not transposed
		return new Pair<double[][], Boolean>(ret, false);
	}

	public static void main(String[] args)  throws IOException {
		SennaDataReader sdr = new SennaDataReader(new File("/home/srmq/devel/senna/hash/words.lst"), new File("/home/srmq/devel/senna/embeddings/embeddings.txt"));
		Pair<InMemoryLookupTable, VocabCache> embeds = sdr.loadOriginalEmbeddings();
		WordVectorSerializer.writeWordVectors(embeds.getFirst(), (InMemoryLookupCache) embeds.getSecond(),
				"Senna-OriginalVectors.txt");

		System.out.println("all done");
		
	}
}

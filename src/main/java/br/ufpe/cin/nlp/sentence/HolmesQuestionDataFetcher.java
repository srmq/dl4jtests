package br.ufpe.cin.nlp.sentence;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.dataset.api.iterator.fetcher.BaseDataFetcher;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.core.io.ClassPathResource;

public class HolmesQuestionDataFetcher extends BaseDataFetcher {
	private int fileCursor;
	private BufferedReader reader;
	
	private File inputFile;
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -3233812699590480647L;
	public HolmesQuestionDataFetcher(int numExamples) {
		this(numExamples, null);
	}
	public HolmesQuestionDataFetcher(int numExamples, File inputFile) {
		numOutcomes = 5;
		inputColumns = 200;
		totalExamples = numExamples;
		this.inputFile = inputFile;
	}

	@Override
	public void fetch(int numExamples) {
		int from = cursor;
		int to = cursor + numExamples;
		if(to > totalExamples)
			to = totalExamples;
		
		try {
			initializeCurrFromList(loadHolmes(from, to, this.inputFile));
			cursor += numExamples;
		} catch (IOException e) {
			throw new IllegalStateException("Unable to load textfile with dataset");
		}
		
		
	}
	
	private void reopenFile(InputStream fileStream) throws IOException {
		this.close();

		this.reader = new BufferedReader(new InputStreamReader(fileStream));
	}
	
	public List<DataSet> loadHolmes(int from, int to) throws IOException {
		return loadHolmes(from, to, null);
	}
	
	public List<DataSet> loadHolmes(int from, int to, File inputFile) throws IOException {
		if (this.reader == null || from < this.fileCursor){
			InputStream iStream;
			if (inputFile == null) {
		        ClassPathResource resource = new ClassPathResource("/HolmesDistanceDataset.txt");
		        iStream = resource.getInputStream();
			} else {
				iStream = new FileInputStream(inputFile);
			}
			this.reopenFile(iStream);
		}
        return getDataset(from, to);
	}
	

	private List<DataSet> getDataset(int from, int to) throws IOException {
		while (this.fileCursor < from) {
        	nextLine();
        }
        assert this.fileCursor == from;
        
        INDArray ret = Nd4j.ones(Math.abs(to - from), inputColumns);
        double[][] outcomes = new double[to - from][numOutcomes];
        List<DataSet> list = new ArrayList<DataSet>(to - from);
        
        int putCount = 0;
        for(int i = from; i < to; i++) {
            String line = nextLine();
            String[] split = line.split(" ");

            addRow(ret,putCount++,split);
        	
            String outcome = split[split.length - 1];
            double[] rowOutcome = new double[numOutcomes];
            rowOutcome[Integer.parseInt(outcome)] = 1;
            outcomes[i - from] = rowOutcome;            
        }
        for(int i = 0; i < ret.rows(); i++) {
            DataSet add = new DataSet(ret.getRow(i), Nd4j.create(outcomes[i]));
            list.add(add);
        }
        return list;
	}
	
	private void addRow(INDArray ret,int row,String[] line) {
		double[] inputs = new double[inputColumns];
		for (int i = 1; i <= inputColumns; i++) {
			inputs[i-1] = Double.parseDouble(line[i]);
		}
		ret.putRow(row,Nd4j.create(inputs));
	}
	
	private String nextLine() throws IOException {
    	final String ret = this.reader.readLine();
    	this.fileCursor++;
    	return ret;
	}
	
	public void close() throws IOException {
		if (this.reader != null) this.reader.close();
		this.fileCursor = 0;
	}
	
	@Override
	protected void finalize() throws Throwable {
		super.finalize();
		this.close();
	}
	
	public static void main(String[] args) {
		HolmesQuestionDataFetcher hqdf = new HolmesQuestionDataFetcher(1040);
		hqdf.fetch(5);
		hqdf.fetch(1);
	}

}

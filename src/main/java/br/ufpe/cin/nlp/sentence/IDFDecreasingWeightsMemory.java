package br.ufpe.cin.nlp.sentence;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import lombok.AccessLevel;
import lombok.AllArgsConstructor;
import lombok.NonNull;


@AllArgsConstructor(access = AccessLevel.PUBLIC)
public class IDFDecreasingWeightsMemory implements WordWeightMemory {
	
	@NonNull
	private TfIdfInfo tfIdfInfo;
	
	private Map<String, Map<Integer, Double>> overrideWeights;
	
	public IDFDecreasingWeightsMemory(TfIdfInfo tfIdfInfo) {
		this(tfIdfInfo, null); 
	}
	

	@Override
	public double weightFor(String word, int distance) throws UnknownWordException, IOException {
		double ret;
		if (overrideWeights != null && overrideWeights.containsKey(word) && overrideWeights.get(word).containsKey(distance)) {
			ret = overrideWeights.get(word).get(distance);
		} else {
			ret = 1.0/Math.abs(distance);
			ret /= Math.log10(1 + tfIdfInfo.docAppearedIn(word));
		}
		return ret;
	}

	@Override
	public void newWeightFor(String word, int distance, double value) throws IOException {
		if (overrideWeights == null) {
			overrideWeights = new HashMap<String, Map<Integer, Double>>();
			insertWordMap(word, distance, value);
		} else if (!overrideWeights.containsKey(word)) {
			insertWordMap(word, distance, value);
		} else {
			final Map<Integer, Double> thisWordMap = overrideWeights.get(word);
			thisWordMap.put(distance, value);
		}

	}

	private void insertWordMap(String word, int distance, double value) {
		final Map<Integer, Double> thisWordMap = new HashMap<Integer, Double>();
		thisWordMap.put(distance, value);
		overrideWeights.put(word, thisWordMap);
		

//		JavaConversions.asScalaBuffer(Arrays.asList(this.overrideWeights.entrySet().toArray())); //FIXME

		
	}
	

}

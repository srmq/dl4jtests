package br.ufpe.cin.nlp.sentence;

import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;

public class Util {
	
	public static LabeledPoint questionProducerLine2LabeledPoint (String line) {
		final String[] split = line.split(" ");
		final int classIndex = Integer.parseInt(split[split.length - 1]);
		final double values[] = new double[split.length - 2];
		for (int i = 1; i < split.length - 1; i++) {
			values[i-1] = Double.parseDouble(split[i]);
		}
		final LabeledPoint ret = new LabeledPoint(classIndex, Vectors.dense(values));
		return ret;
	}

}

package br.ufpe.cin.nlp.sentence;

import org.nd4j.linalg.api.ndarray.INDArray;

public class NDMathUtils {
	public static final int indexMin(INDArray arr) {
		if (arr == null || arr.length() == 0) return -1;
		int ret = 0;
		float minVal = arr.getFloat(0);
		for (int i = 1; i < arr.length(); i++) {
			final float currVal = arr.getFloat(i);
			if (currVal < minVal) {
				minVal = currVal;
				ret = i;
			}
		}
		return ret;
	}
}

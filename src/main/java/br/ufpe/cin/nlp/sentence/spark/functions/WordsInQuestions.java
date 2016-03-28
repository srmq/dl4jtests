package br.ufpe.cin.nlp.sentence.spark.functions;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Set;

import org.apache.spark.api.java.function.Function2;

import br.ufpe.cin.nlp.sentence.base.SentenceCompletionQuestions.Question;
import br.ufpe.cin.util.io.JsonSerializer;
import scala.Tuple2;

public class WordsInQuestions implements Function2<Integer, Iterator<String>, Iterator<Tuple2<Integer, Set<String>>>> {
	

	@Override
	public Iterator<Tuple2<Integer, Set<String>>> call(Integer partIndex, Iterator<String> jsonLine) throws Exception {
		final List<Tuple2<Integer, Set<String>>> result = new ArrayList<Tuple2<Integer, Set<String>>>(1);
		
		final JsonSerializer<Question> jsonSerializer = new JsonSerializer<Question>(Question.class);
		final Set<String> words = new HashSet<String>();
		
		while(jsonLine.hasNext()) {
			final String qStr = jsonLine.next();
			final Question q = jsonSerializer.deserialize(qStr);
			for (String word : q.getTokensBefore()) {
				words.add(word.toLowerCase(Locale.ENGLISH));
			}
			for (String word : q.getTokensAfter()) {
				words.add(word.toLowerCase(Locale.ENGLISH));
			}
			for (String word : q.getOptions()) {
				words.add(word.toLowerCase(Locale.ENGLISH));
			}
		}
		
		result.add(new Tuple2<Integer, Set<String>>(partIndex, words));
		
		return result.iterator();
	}


}

package br.ufpe.cin.nlp.sentence;

import static org.junit.Assert.*;

import java.io.File;
import java.io.IOException;
import java.util.Set;

import org.junit.Assert;
import org.junit.Test;
import org.springframework.core.io.ClassPathResource;



public class SentenceCompletionWordsTest {

	@Test
	public void testSentenceCompletionWordsFileConstructor() {
		ClassPathResource resource = new ClassPathResource("/questions.txt");

		try {
			File f = resource.getFile();
			SentenceCompletionWords scw = new SentenceCompletionWords(f);
			scw.hashCode();
		} catch (IOException e) {
			e.printStackTrace();
			fail("Could not access questions.txt to build SentenceCompletionWords");
		}
	}
	
	@Test
	public void testSentenceCompletionWordsDefaultConstructor() {
		try {
			SentenceCompletionWords scw = new SentenceCompletionWords();
			scw.hashCode();
		} catch (Exception ex) {
			ex.printStackTrace();
			fail("Could not build SentenceCompletionWords using default constructor");
		}
	}

	@Test
	public void testWords() {
		SentenceCompletionWords scw = new SentenceCompletionWords();
		Set<String> words = scw.words();
		Assert.assertTrue(words != null && !words.isEmpty());
	}

}

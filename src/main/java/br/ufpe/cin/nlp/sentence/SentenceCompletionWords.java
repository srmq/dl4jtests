package br.ufpe.cin.nlp.sentence;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Collections;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;

import org.springframework.core.io.ClassPathResource;



public class SentenceCompletionWords {
	
	private File f;
	private Set<String> wordSet;
	
	public SentenceCompletionWords(File f) throws IOException {
        if (!f.exists() || !f.isFile())
            throw new IllegalArgumentException("Please specify an existing file");
        this.f = f;	
	}
	
	public SentenceCompletionWords() {
		ClassPathResource resource = new ClassPathResource("/questions.txt");
		try {
			this.f = resource.getFile();
		} catch (IOException e) {
			throw new IllegalStateException("Could access questions.txt file in classpath", e);		
		}
	}
	
	public Set<String> words() {
		if (wordSet == null) {
			try {
				loadWords();
			} catch(IOException ex) {
				throw new IllegalStateException("Could not load words from file", ex);
			}
		}
		return this.wordSet;
	}
	
	private void loadWords() throws IOException {
		final BufferedReader bufFile = new BufferedReader(new InputStreamReader(new FileInputStream(f), "UTF-8"));
		String line;
		final Set<String> words = new HashSet<String>();
		while((line = bufFile.readLine()) != null) {
			final String[] tokens = line.split(" ");
			for (int i = 1; i < tokens.length; i++) {
				if (tokens[i].charAt(0) != '[') {
					words.add(tokens[i].toLowerCase(Locale.US));
				} else {
					words.add(tokens[i].substring(1, tokens[i].length()-1).toLowerCase(Locale.US));
				}
			}
		}
		this.wordSet = Collections.unmodifiableSet(words);
		bufFile.close();
	}
	
}

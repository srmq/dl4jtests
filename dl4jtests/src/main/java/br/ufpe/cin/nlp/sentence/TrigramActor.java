package br.ufpe.cin.nlp.sentence;

import java.sql.SQLException;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.UimaSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.NGramTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import akka.actor.ActorRef;
import akka.actor.ActorSystem;
import akka.actor.Props;
import akka.actor.UntypedActor;
import akka.routing.RoundRobinPool;
import br.ufpe.cin.nlp.sentence.base.SentencesLuceneIndex;

public class TrigramActor extends UntypedActor {

	private AtomicLong numberDone;
	
	private AtomicLong uniqueTrigrams;

	private SentencesLuceneIndex index;

	private TokenizerFactory tokenFactory;
	
	private NGramTokenizerFactory trigramFactory;
	
	private VocabCache cache;
	
	private static final AtomicInteger wordIndexIdentity = new AtomicInteger(0);

	public static final int MAXTOKENSIZE = 80;
	
	public static final double INFREQUENT_WORD_THRESHOLD = 1e-4;

	private static final Logger log = LoggerFactory.getLogger(TrigramActor.class);
	private TrigramDBManager dbManager;
	
	public static enum WorkType {
		VOCABWORK, TRIGRAMWORK, BDTFIDF
	}

	public TrigramActor(AtomicLong numberDone, TrigramDBManager dbManager,
			SentencesLuceneIndex index, TokenizerFactory tokenFactory, NGramTokenizerFactory trigramFactory, VocabCache cache, AtomicLong uniqueTrigrams) throws SQLException {
		this.dbManager = dbManager;
		this.numberDone = numberDone;
		this.index = index;
		this.tokenFactory = tokenFactory;
		this.trigramFactory = trigramFactory;
		this.cache = cache;
		this.uniqueTrigrams = uniqueTrigrams;
	}

	@Override
	public void onReceive(Object objMsg) throws Exception {
		if (objMsg instanceof String) {
			final String sentence = (String) objMsg;
			this.index.addSentence(sentence);
			final Tokenizer tok = this.tokenFactory.create(sentence);
			this.cache.incrementTotalDocCount(); //total document count: a document is a sentence.
			final Set<String> tokens = new HashSet<String>();
			while (tok.hasMoreTokens()) {
				final String token = tok.nextToken();
				this.cache.incrementWordCount(token); //number of times this word occurred in the corpus
				if (!tokens.contains(token)) {
					this.cache.incrementDocCount(token, 1); //number of documents in the corpus that contains the word
					tokens.add(token);
				}
				VocabWord word = this.cache.tokenFor(token);
				synchronized(wordIndexIdentity) {
					if (this.cache.indexOf(token) == -1) {
						final int id = wordIndexIdentity.getAndIncrement();
				        cache.addWordToIndex(id, token);
				        word.setIndex(id);
				        this.cache.putVocabWord(token);
					}
				}
				
				//dbManager.updateWordOcurr(token);
			}
			this.numberDone.incrementAndGet();
		} else if (objMsg instanceof TrigramWork) {
			String sentence = ((TrigramWork)objMsg).getSentence();
			Tokenizer tokenizer = this.trigramFactory.create(sentence);
	        while (tokenizer.hasMoreTokens()) {
	        	final String nextToken = tokenizer.nextToken();
	        	String[] grams = nextToken.substring(nextToken.indexOf('[') + 1, nextToken.lastIndexOf(']')).split("\\s*,\\s*");
	        	if (grams.length != 3) {
	        		log.warn("Found tri-gram with number of tokens separated by comma different from 3. Trigram was: " + nextToken);
	        	} else {
	        		//adicionar apenas se a ultima palavra do gram é rara.
	        		if (cache.wordFrequency(grams[2]) < (cache.totalWordOccurrences() * INFREQUENT_WORD_THRESHOLD)) {
	        			final int[] idGrams = new int[3];
	        			for (int i = 0; i < idGrams.length; i++) {
							idGrams[i] = cache.indexOf(grams[i]);
							assert (idGrams[i] >= 0);
						}
	        			final int trigramCount = dbManager.addTrigram(idGrams);
	        			if (trigramCount == 1) this.uniqueTrigrams.incrementAndGet();
	        		}
	        	}
	            //System.out.println(nextToken);
	        }
        	this.numberDone.incrementAndGet();
		} else if (objMsg instanceof BDTFIDFWork) {
			VocabWord token = ((BDTFIDFWork)objMsg).getVocab();
			int index = token.getIndex();
			double frequency = token.getWordFrequency();
			int docCount = cache.docAppearedIn(token.getWord());
			dbManager.insertWordFrequency(index, token.getWord(), frequency, docCount);
			this.numberDone.incrementAndGet();
		} else {
			throw new IllegalStateException("Unknown message received");
		}
	}

	public static void main(String[] args) throws Exception {
		final SentenceIterator iteratorSentence = UimaSentenceIterator
				.createWithPath("/home/srmq/devel/Holmes_Sem_Cabecalho_e_Rodape");

		final SentencesLuceneIndex index = new SentencesLuceneIndex("Holmes-sentence-index", false);

		TokenizerFactory tokenFactory = new NGramTokenizerFactory(new StringTokenizerFactory(), 1, 1);
		tokenFactory.setTokenPreProcessor(new TokenPreProcess() {
			@Override
			public String preProcess(String token) {
				String ret = token.toLowerCase(Locale.ENGLISH);
				if (ret.length() > MAXTOKENSIZE)
					ret = ret.substring(0, MAXTOKENSIZE);
				return ret;
			}
		});

		// conn.setTransactionIsolation(Connection.TRANSACTION_SERIALIZABLE);

		ActorSystem trainingSystem = ActorSystem.create();
		AtomicLong done = new AtomicLong(0l);
		TrigramDBManager dbManager = new TrigramDBManager();
		NGramTokenizerFactory trigramFactory = new NGramTokenizerFactory(new StringTokenizerFactory(), 3, 3);
		VocabCache cache = new InMemoryLookupCache();
		AtomicLong uniqueTrigrams = new AtomicLong(0l);
		trigramFactory.setTokenPreProcessor(new TokenPreProcess() {
			@Override
			public String preProcess(String token) {
				String ret = token.toLowerCase(Locale.ENGLISH);
				if (ret.length() > MAXTOKENSIZE)
					ret = ret.substring(0, MAXTOKENSIZE);
				return ret;
			}
		});

		final ActorRef vocabActor = trainingSystem.actorOf(
				new RoundRobinPool(Runtime.getRuntime().availableProcessors()).props(Props.create(TrigramActor.class,
						done, dbManager, index, tokenFactory, trigramFactory, cache, uniqueTrigrams)));

		AtomicLong sent = new AtomicLong(0l);

		
		log.info("Starting TF IDF parsing");
		final Iterator<String> it = new Iterator<String>() {

			@Override
			public boolean hasNext() {
				return iteratorSentence.hasNext();
			}

			@Override
			public String next() {
				return iteratorSentence.nextSentence();
			}
			
		};
		doWork(it, done, vocabActor, sent, WorkType.VOCABWORK);

		
		log.info("Starting BD TF IDF Update");
		sent.set(0);
		done.set(0);
		dbManager.insertGeneralInfo(cache.totalNumberOfDocs(), cache.totalWordOccurrences());
		doWork(cache.tokens().iterator(), done, vocabActor, sent, WorkType.BDTFIDF);

		
		
		log.info("STARTING TrigramWork");
		sent.set(0);
		done.set(0);
		iteratorSentence.reset();
		doWork(it, done, vocabActor, sent, WorkType.TRIGRAMWORK);
		System.out.println("Total number of unique trigrams: " + uniqueTrigrams.get());
		
		trainingSystem.shutdown();
		dbManager.close();
	}

	private static <T> void doWork(Iterator<T> iterator, AtomicLong done, final ActorRef vocabActor, AtomicLong sent, WorkType work) {
		while (iterator.hasNext()) {
			Object message;
			if (work == WorkType.VOCABWORK) {
				message = iterator.next();
			} else if (work == WorkType.TRIGRAMWORK) {
				message = new TrigramWork((String)iterator.next());
			} else if (work == WorkType.BDTFIDF) {
				message = new BDTFIDFWork((VocabWord)iterator.next());
			} else {
				throw new UnsupportedOperationException("Unsupported work type");
			}
			vocabActor.tell(message, vocabActor);
			sent.incrementAndGet();

			if (sent.get() % 10000 == 0) {
				log.info("Sent " + sent);
				try {
					Thread.sleep(1);
				} catch (InterruptedException e) {
					Thread.currentThread().interrupt();
				}
			}

		}

		long lastDone = 0;
		while (done.get() < sent.get()) {
			try {
				if (done.get() - lastDone > 10000) {
					lastDone = done.get();
					log.info("Done " + done + " out of " + sent + " (" + done.get()/((double)sent.get())*100.0 + "%)");
				}
				Thread.sleep(1000);
			} catch (InterruptedException e) {
				Thread.currentThread().interrupt();
			}
		}
	}

}

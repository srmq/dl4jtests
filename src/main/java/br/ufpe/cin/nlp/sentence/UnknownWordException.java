package br.ufpe.cin.nlp.sentence;

public class UnknownWordException extends Exception {
	/**
	 * 
	 */
	private static final long serialVersionUID = -2881298550816291129L;

	public UnknownWordException() {
		super();
	}
	
	public UnknownWordException(String message) {
        super(message);
    }
	
	public UnknownWordException(String message, Throwable cause) {
        super(message, cause);
    }	
	
	public UnknownWordException(Throwable cause) {
		super(cause);
	}
}

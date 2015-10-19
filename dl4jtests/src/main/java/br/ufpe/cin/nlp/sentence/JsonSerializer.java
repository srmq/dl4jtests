package br.ufpe.cin.nlp.sentence;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import org.apache.commons.compress.compressors.gzip.GzipUtils;

import com.google.gson.Gson;

public class JsonSerializer<T> {
	
	private Gson gson;
	private Class<T> theClass;

	
	public JsonSerializer(Class<T> theClass) {
		this.gson = new Gson();
		this.theClass = theClass;
	}

	public T deserialize(File file) throws IOException {
		BufferedReader buf = new BufferedReader(new InputStreamReader(GzipUtils.isCompressedFilename(file.getName())
				? new GZIPInputStream(new FileInputStream(file)) : new FileInputStream(file), "UTF-8"));
		T ret = gson.fromJson(buf, theClass);
		buf.close();
		return ret;
	}
	
	public void serialize(T obj, File toFile, boolean gzipCompression) throws IOException {
		BufferedWriter buf = new BufferedWriter(new OutputStreamWriter(gzipCompression ? new GZIPOutputStream(new FileOutputStream(toFile)) : new FileOutputStream(toFile), "UTF-8"));
		gson.toJson(obj, theClass, buf);
		buf.close();
	}
}

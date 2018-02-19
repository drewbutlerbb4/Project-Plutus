package yahoo.scraper;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

/**	
 * Scrapes the historical data of a number of instruments pulled from a 
 * 	file specified at instantiation
 * 
 * @author Andrew Butler
 */
public class MarketScraper {

	private String fileName;
	private YahooScraper yahooScraper;
	
	/**	
	 * Sets the path of the file to be read from and the delimiter used in the CSV file
	 * 
	 * @param file			Path to the CSV file that contains the instruments to download
	 * @param csvDelimiter	The string that the CSV file uses to divide parameters
	 */
	public MarketScraper(String file, String csvDelimiter){
		fileName = file;
		yahooScraper = new YahooScraper();
	}
	
	/**
	 * Runs through the file at position 'fileName' 
	 * If toFile then read from file all the instruments and use YahooScraper to download
	 * the information about those instruments. Return a string of instruments that were 
	 * not available from Yahoo
	 * Else return a string of instruments that are contained in the file
	 *  
	 * 
	 * @param toFile	Specifies whether the file should be downloaded or 
	 * 					sent back as a string
	 * 
	 * @return 			If toFile then returns a string of instruments that failed to download
	 * 					Else return the string of instruments contained in the file
	 */
	private String interpretFile(boolean toFile){
		
		BufferedReader br = null;
		String line = "";
		String delimiter = ",";
		String toReturn = "";
		
		try{
			
			br = new BufferedReader(new FileReader(fileName));
			line = br.readLine();
			
			while(line!=null){
				
				String[] instrument = line.split(delimiter);
				if(toFile)
					// If the download fails, then add the instrument to the list
					if(yahooScraper.read(instrument[0]))
						toReturn = toReturn + instrument[0] + "\n";
				else
					toReturn = toReturn + instrument[0] + "\n";
				line = br.readLine();
			}
			
		}catch(FileNotFoundException e){
			System.out.println("Hard fail on file not found");
		}catch(IOException e){
			System.out.println("Hard fail on IO");
		}
		
		return toReturn;
	}
	
	/**	
	 * Returns a string listing all the instruments to be downloaded
	 * 
	 * @return	A string of all the instruments to be downloaded
	 */
	public String previewDownload(){
		return interpretFile(false);
	}
	
	/**
	 * Downloads the files that are listed in the CSV file and returns a string of 
	 * instruments that could not be downloaded
	 * 
	 * @return		Returns a string of instruments that couldn't be downloaded
	 */
	public String downloadFiles(){
		return interpretFile(true);
	}
}

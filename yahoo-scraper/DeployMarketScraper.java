package yahoo.scraper;

/**
 * Deployer of Market Scraper internally
 * 
 * @author Andrew Butler
 *
 */
public class DeployMarketScraper {
	public static void main(String args[]){
		
		String file = "D:\\MarketEval\\InstrumentValueByDay\\InstrumentsLSEStripped - Sheet1.csv";
		MarketScraper ms = new MarketScraper(file,",");
		System.out.println(ms.downloadFiles());
	}
}

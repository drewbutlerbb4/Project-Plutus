package yahoo.scraper;

import java.util.concurrent.TimeUnit;

import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;

/**Scrapes historical data of instruments from Yahoo's financial repository
 * This web scraper uses Selenium as for web crawling
 * 
 * @author Andrew Butler
 *
 */
public class YahooScraper {

	
	private WebDriver driver;
	
	/**
	 * Initializes YahooScraper
	 */
	public YahooScraper(){
		System.setProperty("webdriver.chrome.driver", "C:\\Users\\Andrew Butler\\Documents\\MarketEval\\Libraries\\chromedriver_win32\\chromedriver.exe");
		driver = new ChromeDriver();
		driver.manage().timeouts().implicitlyWait(10, TimeUnit.SECONDS);
		driver.manage().window().maximize();
	}
	
	
	// Current best max value is from period1 value = 0
	// However, this value has not been confirmed as best representation
	//
	// 1. Yahoo! Finance uses crumbs in their url, which forces us to find our bread crumb 
	// in previous pages in order to advance to the download (More on this in the project documentation)
	/**
	 * Downloads a file of historical data to the downloads folder from a given instrument's Yahoo page
	 * Attempts to download the max amount of historical data
	 * 
	 * 
	 * 
	 * @param abbr The abbreviation of the instrument
	 * 
	 * @return	   True if the download went through, false otherwise
	 */
	public boolean read(String abbr){
		
		try{
			
			String fullAbbr = abbr+".L";

			driver.get("https://finance.yahoo.com/quote/"+fullAbbr+"/history?p="+fullAbbr);
			String pageSource = driver.getPageSource();
			
			// See 1. above method header for explanation of crumbHelper
			String crumbHelper = pageSource.substring(pageSource.indexOf("Fl(end) Mt(3px)"));
			crumbHelper = crumbHelper.substring(crumbHelper.indexOf("crumb"));

			String crumb = crumbHelper.substring(crumbHelper.indexOf("=")+1, crumbHelper.indexOf("\""));
			driver.get("https://query1.finance.yahoo.com/v7/finance/download/"
					+fullAbbr+"?period1=0&period2=1503115200&interval=1d&events=history&crumb="+crumb);
		
		}catch(StringIndexOutOfBoundsException e){
			return false;
		}
		return true;
	}
}

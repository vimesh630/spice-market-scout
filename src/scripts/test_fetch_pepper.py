
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collectors.exagri_scraper import get_pepper_scraper

def test_fetch_recent_pepper():
    scraper = get_pepper_scraper()
    
    # Test for a few months after the dataset ends
    months_to_test = [
        (2024, 6),
        (2024, 9),
        (2024, 12),
        (2025, 1)
    ]
    
    print("Testing Pepper Scraper for missing months...")
    
    found_data = False
    for year, month in months_to_test:
        print(f"\nScanning {year}-{month:02d}...")
        try:
            # First check if links exist
            links = scraper.get_weekly_links_for_month(year, month)
            print(f"  Found {len(links)} weekly reports.")
            
            if links:
                # Try to fetch prices
                prices = scraper.get_monthly_average_prices(year, month)
                if prices:
                    print(f"  Successfully fetched prices for {len(prices)} regions.")
                    # Show sample
                    sample_region = list(prices.keys())[0]
                    print(f"  Sample ({sample_region}): {prices[sample_region]}")
                    found_data = True
                else:
                    print("  No prices extracted from reports (tables might have changed format).")
            else:
                print("  No reports found.")
                
        except Exception as e:
            print(f"  Error: {e}")
            
    return found_data

if __name__ == "__main__":
    test_fetch_recent_pepper()

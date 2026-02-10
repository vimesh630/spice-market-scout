"""Check detailed pepper table on exagri.info - print all matches"""
import sys
sys.path.insert(0, 'src')

import requests
from bs4 import BeautifulSoup

url = "https://exagri.info/mkt/2025/28.10.2025.html"
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
})

resp = session.get(url, timeout=30)
resp.raise_for_status()
soup = BeautifulSoup(resp.content, 'html.parser')

tables = soup.find_all('table')
print(f"Total tables found: {len(tables)}")

for i, table in enumerate(tables):
    header_row = table.find('tr')
    if not header_row:
        continue
    headers = [th.get_text(strip=True).lower() for th in header_row.find_all(['th', 'td'])]
    # Check for pepper keywords
    if any('pepper' in h or 'gr1' in h or 'white' in h for h in headers):
        print(f"\n=== Table {i} (possible pepper) ===")
        print(f"Headers: {headers}")
        for j, row in enumerate(table.find_all('tr')[1:8]):
            cells = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
            print(f"  Row {j}: {cells}")
        print(f"  Total rows: {len(table.find_all('tr'))}")

# Also check all table headers for commodity identification
print("\n=== All table headers ===")
for i, table in enumerate(tables):
    header_row = table.find('tr')
    if header_row:
        headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
        if len(headers) > 2:
            print(f"Table {i}: {headers[:6]}{'...' if len(headers) > 6 else ''}")

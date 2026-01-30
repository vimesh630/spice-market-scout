import requests
import json
import sys

BASE_URL = "http://127.0.0.1:8000"

def test_metadata():
    try:
        print(f"Testing GET {BASE_URL}/metadata...")
        response = requests.get(f"{BASE_URL}/metadata")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Error testing metadata: {e}")

def test_predict(region, grade):
    try:
        print(f"\nTesting POST {BASE_URL}/predict for {region}, {grade}...")
        payload = {
            "region": region,
            "grade": grade,
            "months": 6
        }
        headers = {'Content-Type': 'application/json'}
        response = requests.post(f"{BASE_URL}/predict", json=payload, headers=headers)
        print(f"Status Code: {response.status_code}")
        try:
             print(f"Response: {response.json()}")
        except:
             print(f"Response (Text): {response.text}")
             
    except Exception as e:
        print(f"Error testing predict: {e}")

def test_news():
    try:
        print(f"\nTesting GET {BASE_URL}/news...")
        response = requests.get(f"{BASE_URL}/news")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error testing news: {e}")

if __name__ == "__main__":
    meta = test_metadata()
    if meta:
        regions = meta.get('regions', [])
        grades = meta.get('grades', [])
        
        if regions and grades:
            test_predict(regions[0], grades[0])
        else:
            print("No regions or grades found to test predict.")
            
    test_news()

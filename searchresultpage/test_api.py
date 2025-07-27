import requests
import time

# The URL where your API is running
API_URL = "http://127.0.0.1:8000/predict/"

queries = [
    "oneplus mobile",
    "i want a gaming laptop with 16gb ram",
    "addidas shoes"
]

for query in queries:
    start_time = time.time()
    
    response = requests.post(API_URL, json={"query": query})
    
    end_time = time.time()
    
    if response.status_code == 200:
        print(f"Query: '{query}'")
        print(f"Response: {response.json()}")
        print(f"--- Prediction took: {end_time - start_time:.4f} seconds ---\n")
    else:
        print(f"Error: {response.text}")
        
import spacy
import time

# ✅ Load your trained spaCy model
nlp = spacy.load("R:\\sem VII\\Flipkart Grid 7.0\\flipkart-grid\\searchresultpage\\spacy_ner_model")

def extract_entities(text):
    start = time.time()
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        entities.setdefault(ent.label_, []).append(ent.text)
    end = time.time()
    print(f"Query: '{text}'")
    print(f"Response: {{'query': '{text}', 'entities': {entities}}}")
    print(f"--- Prediction took: {end - start:.4f} seconds ---\n")

# ✅ Sample usage with multiple test queries
sample_queries = [
    "oneplus mobile",
    "i want a gaming laptop with 16gb ram",
    "addidas shoes",
    "red color shoes under 3000",
    "iphone 13 with best camera and storage",
    "can i get nike running shoes in blue"
]

for q in sample_queries:
    extract_entities(q)
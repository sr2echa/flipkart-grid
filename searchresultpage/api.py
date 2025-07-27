# api.py
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import json
from typing import List, Dict

# This is the NERInference class from your predict_ner.py script
class NERInference:
    def __init__(self, model_path: str):
        print(f"Loading model from {model_path}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)
            with open(f"{model_path}/label_mappings.json", 'r') as f:
                mappings = json.load(f)
                self.id2label = {int(k): v for k, v in mappings['id2label'].items()}
            self.model.eval()
            print("✅ Model loaded successfully and is ready for inference.")
        except OSError:
            print(f"❌ Error: No model found at '{model_path}'.")
            exit()

    def predict(self, query: str) -> Dict[str, List[str]]:
        tokens = self.tokenizer.tokenize(query)
        inputs = self.tokenizer.encode(query, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(inputs).logits
        predictions = torch.argmax(outputs, dim=2)
        entities = {}
        current_entity = None
        for token, pred_id in zip(tokens, predictions[0].numpy()[1:-1]):
            label = self.id2label[pred_id]
            if label.startswith("B-"):
                entity_type = label[2:]
                if entity_type not in entities:
                    entities[entity_type] = []
                entities[entity_type].append(token)
                current_entity = (entity_type, len(entities[entity_type]) - 1)
            elif label.startswith("I-") and current_entity:
                entity_type, idx = current_entity
                if label[2:] == entity_type:
                    entities[entity_type][idx] += token # <--- CORRECT: Just append the token
                else:
                    current_entity = None
            else:
                current_entity = None
        for entity_type, values in entities.items():
            entities[entity_type] = [v.replace('##', '') for v in values]
        return entities

# --- API Setup ---

# Define the path to your trained model
MODEL_PATH = "R:\\sem VII\\Flipkart Grid 7.0\\ecommerce-ner-model"

# Create the FastAPI app
app = FastAPI(title="E-commerce NER API")

# Load the model ONCE at startup
ner_model = NERInference(MODEL_PATH)

# Define the request body for the API endpoint
class QueryRequest(BaseModel):
    query: str

# Define the prediction endpoint
@app.post("/predict/", summary="Extract entities from a query")
def predict_entities(request: QueryRequest) -> Dict:
    """
    Takes a user query and returns the detected e-commerce entities.
    This prediction is very fast as the model is already in memory.
    """
    entities = ner_model.predict(request.query)
    return {"query": request.query, "entities": entities}
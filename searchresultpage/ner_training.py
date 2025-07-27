# utils/convert_to_spacy_format.py

import pandas as pd
import spacy
from spacy.training import Example

def convert_to_spacy_format(csv_path):
    df = pd.read_csv(csv_path)
    data = []

    for _, row in df.iterrows():
        text = row['query']
        entities = eval(row['entities'])  # Format: [(start, end, "LABEL")]
        data.append((text, {"entities": entities}))

    return data

import pandas as pd
import spacy
from spacy.training import Example

def convert_bio_to_spacy_format(csv_path):
    df = pd.read_csv(csv_path)

    data = []
    grouped = df.groupby("query_id")

    for query_id, group in grouped:
        tokens = group["token"].tolist()
        tags = group["tag"].tolist()

        text = " ".join(tokens)
        entities = []

        current_pos = 0
        for token, tag in zip(tokens, tags):
            start = text.find(token, current_pos)
            end = start + len(token)

            if tag.startswith("B-"):
                entity_label = tag[2:]
                entities.append((start, end, entity_label))

            current_pos = end

        data.append((text, {"entities": entities}))

    return data

def train_spacy_ner(data_path, output_dir="spacy_ner_model"):
    data = convert_bio_to_spacy_format(data_path)
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")

    for _, annotations in data:
        for ent in annotations["entities"]:
            ner.add_label(ent[2])

    optimizer = nlp.begin_training()

    for int in range(20):
        for text, annotations in data:
            example = Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example], drop=0.2, sgd=optimizer)

    nlp.to_disk(output_dir)
    print(f"✅ spaCy NER model saved to: {output_dir}")
    
  
train_spacy_ner("R:\\sem VII\\Flipkart Grid 7.0\\flipkart-grid\\dataset\\ner_dataset.csv")
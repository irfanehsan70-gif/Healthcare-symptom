from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import uvicorn

app = FastAPI(title="Apriori Symptom Analysis")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = {'rules': [], 'frequent_itemsets': None, 'symptom_set': set()}

class PredictRequest(BaseModel):
    symptoms: list
def load_dataset(path="dataset.csv"):
    df = pd.read_csv(path, keep_default_na=False)  # handles empty Symptom3
    transactions = []
    for _, row in df.iterrows():
        # Combine the symptom columns into one list
        trans = []
        for col in ['Symptom1', 'Symptom2', 'Symptom3']:
            val = row.get(col)
            if val and str(val).strip():
                trans.append(str(val).strip().lower())
        # append disease as last element
        if 'Disease' in df.columns and row['Disease']:
            trans.append(str(row['Disease']).strip().lower())
        transactions.append(trans)
    return transactions
def train_apriori(transactions, min_support=0.05, min_confidence=0.4):
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    MODEL['symptom_set'] = set(df.columns)
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    saved_rules = []
    for _, r in rules.iterrows():
        saved_rules.append({
            'antecedent': sorted(list(r['antecedents'])),
            'consequent': sorted(list(r['consequents'])),
            'support': float(r['support']),
            'confidence': float(r['confidence']),
            'lift': float(r['lift'])
        })
    MODEL['frequent_itemsets'] = frequent_itemsets.to_dict(orient='records')
    MODEL['rules'] = saved_rules


@app.on_event("startup")
def startup_event():
    transactions = load_dataset()
    train_apriori(transactions)
    print(f"Apriori model trained with {len(MODEL['rules'])} rules.")


@app.get('/rules')
def get_rules(limit: int = 50):
    return {'count': len(MODEL['rules']), 'rules': MODEL['rules'][:limit]}

@app.post('/predict')
def predict(req: PredictRequest):
    input_symptoms = set([s.strip().lower() for s in req.symptoms if s.strip()])
    candidates = []
    for r in MODEL['rules']:
        if set(r['antecedent']).issubset(input_symptoms):
            candidates.append(r)
    if not candidates:
        scored = []
        for r in MODEL['rules']:
            overlap = len(set(r['antecedent']) & input_symptoms)
            if overlap > 0:
                score = (overlap / max(1, len(r['antecedent']))) * r['confidence']
                scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        candidates = [r for s, r in scored[:5]]
    
    # Filter only diseases
    disease_set = set()
    for r in MODEL['rules']:
        disease_set.update([c.lower() for c in r['consequents']])
    
    predictions = {}
    for c in candidates:
        disease_items = [item for item in c['consequents'] if item in disease_set]
        if not disease_items:
            continue
        key = ", ".join(disease_items)
        if key not in predictions:
            predictions[key] = {'antecedent': c['antecedent'], 'consequent': disease_items, 'confidence': c['confidence'], 'support': c['support'], 'lift': c['lift']}

    return {'input': list(input_symptoms), 'predictions': list(predictions.values())}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)

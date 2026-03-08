#!/usr/bin/env python3
"""
SchemaLabs v1 PoC: Sector-Agnostic Detection from Column Names + Cell Values
"""
import json, os, random, math, numpy as np
from pathlib import Path
from collections import Counter, defaultdict

METADATA_PATH = Path(os.path.expanduser("~/Desktop/schemalabsai/data/poc_metadata.json"))

# ============================================================
# 1. DOMAIN KEYWORDS (semantic features)
# ============================================================
DOMAIN_KEYWORDS = {
    "healthcare": ["age","patient","diagnosis","blood","heart","medical","disease","glucose",
                   "bmi","pressure","cholesterol","clinical","hospital","diabetes","stroke",
                   "anaemia","platelets","creatinine","sodium","ejection","smoking","death"],
    "finance": ["price","amount","credit","loan","interest","transaction","payment","fraud",
                "income","debt","revenue","profit","stock","trading","insurance","premium",
                "claim","policy","charges","capital","budget","v1","v2","v3"],
    "ecommerce": ["product","order","cart","customer","invoice","quantity","shipping","purchase",
                  "item","stock","discount","seller","buyer","description","unitprice","country"],
    "realestate": ["property","house","apartment","rent","sqft","bedroom","bathroom","floor",
                   "building","zoning","lot","acre","land","bed","bath","broker","condo","price"],
    "telecom": ["mobile","churn","call","data","network","roaming","recharge","arpu","mou",
                "sms","bandwidth","subscriber","plan","usage","telecom","sim","broadband"],
    "energy": ["power","solar","wind","electricity","consumption","generation","load","kwh",
               "renewable","fossil","nuclear","turbine","biomass","emission","carbon","fuel"],
    "agriculture": ["crop","farm","soil","irrigation","fertilizer","pesticide","yield","rainfall",
                    "harvest","seed","temperature","humidity","ph","nitrogen","potassium"],
    "crime": ["crime","offense","arrest","victim","incident","police","weapon","theft","robbery",
              "assault","homicide","latitude","longitude","district","beat","ward","severity"],
    "sports": ["team","player","goal","score","match","league","season","referee","shot","foul",
               "corner","halftime","home","away","fifa","football","basketball","cricket"],
    "cybersecurity": ["attack","malware","intrusion","protocol","flag","port","packet","threat",
                      "vulnerability","anomaly","normal","duration","src_bytes","dst_bytes","tcp"],
    "manufacturing": ["production","defect","maintenance","downtime","quality","machine","torque",
                      "temperature","rpm","tool","wear","failure","cycle","assembly","yield"],
    "weather": ["temperature","humidity","wind","pressure","precipitation","cloud","visibility",
                "uv","sunrise","sunset","moon","forecast","rain","snow","condition","celsius"],
    "entertainment": ["movie","show","title","director","cast","rating","genre","release",
                      "duration","season","episode","netflix","youtube","views","likes","channel"],
    "logistics": ["shipment","warehouse","delivery","inventory","tracking","freight","route",
                  "transit","cargo","fleet","delay","logistics","supply","demand","asset"],
    "marketing": ["campaign","impression","click","conversion","ad","spend","reach","engagement",
                  "ctr","cpc","roi","channel","social","influencer","tv","radio","revenue"],
    "education": ["student","grade","exam","course","score","teacher","school","university",
                  "gpa","attendance","semester","class","learning","education","performance"],
    "hr": ["employee","salary","attrition","department","job","hire","termination","performance",
           "satisfaction","overtime","promotion","manager","tenure","training","engagement"],
    "retail": ["product","category","inventory","sku","warranty","stock","color","size",
               "dimension","manufacturing","expiration","tag","rating","price"],
    "government": ["census","income","occupation","education","marital","race","sex","age",
                   "workclass","capital","gain","loss","hours","country","relationship"],
    "science": ["sepal","petal","species","length","width","measurement","observation",
                "sample","experiment","classification","cluster","feature"]
}

# ============================================================
# 2. FEATURE EXTRACTION
# ============================================================
def extract_features(dataset):
    """Column names + cell values → feature vector"""
    columns = [c.lower().strip() for c in dataset["columns"]]
    col_text = " ".join(columns)
    
    # A) Keyword match scores (20 sectors)
    keyword_scores = []
    for sector, keywords in DOMAIN_KEYWORDS.items():
        score = 0
        for kw in keywords:
            for col in columns:
                if kw in col:
                    score += 1
        keyword_scores.append(score)
    
    # B) Column name statistics
    n_cols = len(columns)
    avg_col_len = np.mean([len(c) for c in columns]) if columns else 0
    has_id = any("id" in c for c in columns)
    has_date = any(d in col_text for d in ["date","time","year","month","day","timestamp"])
    has_amount = any(a in col_text for a in ["amount","price","cost","revenue","salary","charges","budget"])
    has_rate = any(r in col_text for r in ["rate","ratio","percentage","pct","%"])
    has_geo = any(g in col_text for g in ["lat","lon","latitude","longitude","location","city","state","country"])
    has_name = any(n in col_text for n in ["name","title","description","text"])
    n_numeric_cols = 0
    n_categorical_cols = 0
    
    # C) Cell value analysis (from sample_rows)
    sample_rows = dataset.get("sample_rows", [])
    numeric_ratios = []
    if sample_rows:
        for col_idx in range(min(n_cols, len(sample_rows[0]))):
            vals = [row[col_idx] for row in sample_rows if col_idx < len(row)]
            numeric_count = sum(1 for v in vals if _is_numeric(v))
            ratio = numeric_count / max(len(vals), 1)
            numeric_ratios.append(ratio)
            if ratio > 0.5:
                n_numeric_cols += 1
            else:
                n_categorical_cols += 1
    
    numeric_ratio = np.mean(numeric_ratios) if numeric_ratios else 0
    
    # D) Value range features (for numeric columns)
    value_stats = []
    if sample_rows:
        for col_idx in range(min(n_cols, len(sample_rows[0]))):
            vals = []
            for row in sample_rows:
                if col_idx < len(row):
                    try:
                        vals.append(float(row[col_idx]))
                    except:
                        pass
            if vals:
                value_stats.append({
                    "mean": np.mean(vals),
                    "std": np.std(vals),
                    "min": np.min(vals),
                    "max": np.max(vals)
                })
    
    avg_mean = np.mean([v["mean"] for v in value_stats]) if value_stats else 0
    avg_std = np.mean([v["std"] for v in value_stats]) if value_stats else 0
    value_range = np.mean([v["max"] - v["min"] for v in value_stats]) if value_stats else 0
    
    # Combine all features
    features = keyword_scores + [
        n_cols / 100.0,
        avg_col_len / 50.0,
        float(has_id),
        float(has_date),
        float(has_amount),
        float(has_rate),
        float(has_geo),
        float(has_name),
        n_numeric_cols / max(n_cols, 1),
        n_categorical_cols / max(n_cols, 1),
        numeric_ratio,
        min(avg_mean / 1e6, 1.0),
        min(avg_std / 1e5, 1.0),
        min(value_range / 1e6, 1.0),
    ]
    
    return np.array(features, dtype=np.float32)

def _is_numeric(val):
    try:
        float(str(val).replace(",", ""))
        return True
    except:
        return False

# ============================================================
# 3. SIMPLE NEURAL NETWORK (Pure NumPy - no PyTorch needed)
# ============================================================
class SimpleNN:
    """2-layer MLP with ReLU, trained with SGD"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.01):
        # Xavier init
        self.W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0/input_dim)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = np.random.randn(hidden_dim, hidden_dim//2).astype(np.float32) * np.sqrt(2.0/hidden_dim)
        self.b2 = np.zeros(hidden_dim//2, dtype=np.float32)
        self.W3 = np.random.randn(hidden_dim//2, output_dim).astype(np.float32) * np.sqrt(2.0/(hidden_dim//2))
        self.b3 = np.zeros(output_dim, dtype=np.float32)
        self.lr = lr
    
    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = np.maximum(0, self.z2)  # ReLU
        self.z3 = self.a2 @ self.W3 + self.b3
        # Softmax
        exp_z = np.exp(self.z3 - np.max(self.z3, axis=-1, keepdims=True))
        self.probs = exp_z / np.sum(exp_z, axis=-1, keepdims=True)
        return self.probs
    
    def backward(self, X, y_onehot):
        m = X.shape[0]
        dz3 = (self.probs - y_onehot) / m
        dW3 = self.a2.T @ dz3
        db3 = np.sum(dz3, axis=0)
        
        da2 = dz3 @ self.W3.T
        dz2 = da2 * (self.z2 > 0).astype(np.float32)
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0)
        
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0).astype(np.float32)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)
        
        # SGD update
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
    
    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=-1)

# ============================================================
# 4. TRAINING & EVALUATION
# ============================================================
def train_and_evaluate(data):
    print("\n" + "="*60)
    print("PHASE 1: SEEN SECTOR TEST (train/test split)")
    print("="*60)
    
    # Build sector mapping
    sectors = sorted(set(d["sector"] for d in data))
    sector2idx = {s: i for i, s in enumerate(sectors)}
    n_sectors = len(sectors)
    print(f"Sectors: {n_sectors}")
    
    # Extract features
    X_all = []
    y_all = []
    for d in data:
        feat = extract_features(d)
        X_all.append(feat)
        y_all.append(sector2idx[d["sector"]])
    
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    
    # Normalize features
    mean = X_all.mean(axis=0)
    std = X_all.std(axis=0) + 1e-8
    X_norm = (X_all - mean) / std
    
    input_dim = X_norm.shape[1]
    print(f"Feature dim: {input_dim}")
    print(f"Samples: {len(X_norm)}")
    
    # Train/test split (80/20)
    indices = list(range(len(X_norm)))
    random.seed(42)
    random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_idx = indices[:split]
    test_idx = indices[split:]
    
    X_train, y_train = X_norm[train_idx], y_all[train_idx]
    X_test, y_test = X_norm[test_idx], y_all[test_idx]
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # One-hot encode
    y_train_oh = np.zeros((len(y_train), n_sectors), dtype=np.float32)
    for i, y in enumerate(y_train):
        y_train_oh[i, y] = 1.0
    
    # Train
    model = SimpleNN(input_dim, 64, n_sectors, lr=0.05)
    
    for epoch in range(200):
        model.forward(X_train)
        model.backward(X_train, y_train_oh)
        
        if (epoch+1) % 50 == 0:
            preds = model.predict(X_train)
            train_acc = (preds == y_train).mean()
            
            test_preds = model.predict(X_test)
            test_acc = (test_preds == y_test).mean()
            
            print(f"  Epoch {epoch+1:3d}: Train={train_acc*100:.1f}% Test={test_acc*100:.1f}%")
    
    # Final test results
    test_preds = model.predict(X_test)
    test_acc = (test_preds == y_test).mean()
    print(f"\n  FINAL TEST ACCURACY: {test_acc*100:.1f}%")
    
    # Per-sector accuracy
    print("\n  Per-sector results:")
    for s_idx, s_name in enumerate(sectors):
        mask = y_test == s_idx
        if mask.sum() > 0:
            s_acc = (test_preds[mask] == y_test[mask]).mean()
            print(f"    {s_name:20s}: {s_acc*100:.0f}% ({mask.sum()} samples)")
    
    # Confusion details for wrong predictions
    wrong = test_preds != y_test
    if wrong.sum() > 0:
        print(f"\n  Wrong predictions ({wrong.sum()}):")
        for i in np.where(wrong)[0]:
            actual = sectors[y_test[i]]
            predicted = sectors[test_preds[i]]
            folder = data[test_idx[i]]["folder"]
            print(f"    {folder[:40]:40s} | actual={actual:15s} predicted={predicted}")
    
    # ============================================================
    # PHASE 2: UNSEEN SECTOR TEST (leave-one-sector-out)
    # ============================================================
    print("\n" + "="*60)
    print("PHASE 2: UNSEEN SECTOR TEST (zero-shot)")
    print("="*60)
    print("Her sektörü sırayla train'den çıkarıp tahmin edebiliyor mu?\n")
    
    # For each sector with enough samples, hold it out
    unseen_results = {}
    for holdout_sector in sectors:
        holdout_data = [d for d in data if d["sector"] == holdout_sector]
        train_data = [d for d in data if d["sector"] != holdout_sector]
        
        if len(holdout_data) < 1:
            continue
        
        # Train model without holdout sector
        train_sectors = sorted(set(d["sector"] for d in train_data))
        # Include holdout in mapping so we can test
        all_sectors = sorted(set(d["sector"] for d in data))
        s2i = {s: i for i, s in enumerate(all_sectors)}
        n_s = len(all_sectors)
        
        X_tr = np.array([extract_features(d) for d in train_data])
        y_tr = np.array([s2i[d["sector"]] for d in train_data])
        X_tr_norm = (X_tr - mean) / std
        
        y_tr_oh = np.zeros((len(y_tr), n_s), dtype=np.float32)
        for i, y in enumerate(y_tr):
            y_tr_oh[i, y] = 1.0
        
        m = SimpleNN(input_dim, 64, n_s, lr=0.05)
        for epoch in range(200):
            m.forward(X_tr_norm)
            m.backward(X_tr_norm, y_tr_oh)
        
        # Test on holdout
        X_ho = np.array([extract_features(d) for d in holdout_data])
        X_ho_norm = (X_ho - mean) / std
        ho_preds = m.predict(X_ho_norm)
        
        # Check: did it predict the correct sector?
        correct = 0
        for i, d in enumerate(holdout_data):
            pred_sector = all_sectors[ho_preds[i]]
            actual_sector = d["sector"]
            is_correct = pred_sector == actual_sector
            if is_correct:
                correct += 1
            status = "✓" if is_correct else "✗"
            print(f"  {status} [{holdout_sector:15s}] {d['folder'][:35]:35s} → predicted: {pred_sector}")
        
        acc = correct / len(holdout_data) * 100
        unseen_results[holdout_sector] = acc
    
    print(f"\n{'='*60}")
    print("UNSEEN SECTOR SUMMARY:")
    print(f"{'='*60}")
    total_correct = 0
    total_samples = 0
    for s, acc in sorted(unseen_results.items()):
        count = sum(1 for d in data if d["sector"] == s)
        n_correct = int(acc * count / 100)
        total_correct += n_correct
        total_samples += count
        status = "✓" if acc >= 50 else "✗"
        print(f"  {status} {s:20s}: {acc:5.1f}% ({n_correct}/{count})")
    
    overall = total_correct / max(total_samples, 1) * 100
    print(f"\n  OVERALL UNSEEN ACCURACY: {overall:.1f}%")
    print(f"  (Model hiç görmediği sektörleri {overall:.0f}% doğrulukla tahmin etti)")
    
    if overall >= 70:
        print("\n  ✅ SECTOR-AGNOSTIC ÇALIŞIYOR! Büyük model için devam edilebilir.")
    elif overall >= 40:
        print("\n  ⚠️  KISMEN ÇALIŞIYOR. Feature engineering geliştirilebilir.")
    else:
        print("\n  ❌ YETERSİZ. Mimari değişiklik gerekli.")

# ============================================================
# 5. MAIN
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("SchemaLabs v1 PoC: Sector-Agnostic Detection")
    print("Column Names + Cell Values → Sector Prediction")
    print("="*60)
    
    data = load_metadata()
    
    np.random.seed(42)
    random.seed(42)
    
    train_and_evaluate(data)

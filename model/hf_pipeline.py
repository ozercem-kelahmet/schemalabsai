"""
HuggingFace Dataset Pipeline — sector labeling
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import load_dataset

SAVE_FILE = Path("data/sector_labels.json")

DATASETS = [
    # sports
    {"id": "maharshipandya/cricket-dataset",                "sector": "sports"},
    {"id": "rhchengit/nba-stats",                           "sector": "sports"},
    {"id": "benediktstroebl/fifa22-player-stats",           "sector": "sports"},
    # manufacturing
    {"id": "mstz/bearing-dataset",                          "sector": "manufacturing"},
    {"id": "phiresky/predictive-maintenance",               "sector": "manufacturing"},
    # telecom
    {"id": "atahmasbi/telco-churn",                         "sector": "telecom"},
    {"id": "scikit-learn/churn-prediction",                 "sector": "telecom"},
    # entertainment
    {"id": "ashraq/tmdb-movies",                            "sector": "entertainment"},
    {"id": "TemporalTrendTracker/netflix-dataset",          "sector": "entertainment"},
    {"id": "maharshipandya/spotify-tracks-dataset",         "sector": "entertainment"},
    # supplychain
    {"id": "gurobi-ml/supply-chain",                        "sector": "supplychain"},
    # healthcare
    {"id": "scikit-learn/breast-cancer",                    "sector": "healthcare"},
    {"id": "scikit-learn/diabetes",                         "sector": "healthcare"},
    # finance
    {"id": "scikit-learn/default-credit-card-clients",      "sector": "finance"},
    # education
    {"id": "scikit-learn/adult-census-income",              "sector": "education"},
    # finance
    {"id": "inGeniia/german-credit-risk_credit-scoring_mlp",  "sector": "finance"},
    {"id": "jyunyilin/credit-card-fraud-detection",            "sector": "finance"},
    {"id": "mstz/bank",                                        "sector": "finance"},
    # sports
    {"id": "Valarmathy/cricket_indvspak",                      "sector": "sports"},
    # realestate
    {"id": "electricsheepafrica/nigerian_realestate_property_tax",         "sector": "realestate"},
    {"id": "electricsheepafrica/nigerian_realestate_mortgage_applications", "sector": "realestate"},
    {"id": "electricsheepafrica/nigerian_realestate_land_use_zoning",      "sector": "realestate"},
    # energy
    {"id": "mstz/electricity",                                             "sector": "energy"},
    {"id": "electricsheepafrica/nigerian_electricity_grid_infrastructure",  "sector": "energy"},
    {"id": "electricsheepafrica/nigerian_electricity_disco_performance",    "sector": "energy"},
    # insurance
    {"id": "bdr-ai-org/insurance-motor-claims-decision-v1",               "sector": "insurance"},
    {"id": "electricsheepafrica/national-health-insurance-enrolment",      "sector": "insurance"},
    {"id": "electricsheepafrica/community-based-health-insurance",         "sector": "insurance"},
    # entertainment
    {"id": "maharshipandya/spotify-tracks-dataset",                        "sector": "entertainment"},
    {"id": "sfiore/spotify-tracks-dataset",                                "sector": "entertainment"},
    {"id": "ozefe/spotify_audio_features",                                 "sector": "entertainment"},
    # manufacturing
    {"id": "akash140500/Predictive_Maintenance_Dataset",    "sector": "manufacturing"},
    {"id": "EddyGiusepe/Modified_dataset_for_predictive_maintenance", "sector": "manufacturing"},
    {"id": "Fdddhhhill/industrial_iot_sensor_data.csv",     "sector": "manufacturing"},
    # supplychain
    {"id": "electricsheepafrica/nigerian_retail_and_ecommerce_supply_chain_logistics_data", "sector": "supplychain"},
    {"id": "alalfi/SupplyChainDataset",                     "sector": "supplychain"},
    {"id": "electricsheepafrica/nigerian_transport_and_logistics_delivery_routes", "sector": "supplychain"},
    # hr
    {"id": "eduvance/employee_attrition",                   "sector": "hr"},
    {"id": "Johnmahith/employee_attrition",                 "sector": "hr"},
    {"id": "in1t/employee_attrition",                       "sector": "hr"},
    # transportation
    {"id": "arasu12/Flight_Delay",                          "sector": "transportation"},
    {"id": "swhoyle/flight-delays",                         "sector": "transportation"},
    {"id": "mohamedababsa/NYC_Trips_Taxi_Dataset",           "sector": "transportation"},
    # entertainment
    {"id": "lilacai/lilac-the_movies_dataset",              "sector": "entertainment"},
    {"id": "drossi/EDA_on_IMDB_Movies_Dataset",             "sector": "entertainment"},
    {"id": "Jarbas/music_queries_metal_tracks",             "sector": "entertainment"},
]

MIN_ROWS = 100
MIN_COLS = 4

import numpy as np

def encode_stats_check(column_stats):
    matrix = np.zeros((50, 6), dtype=np.float32)
    for i, (col, s) in enumerate(list(column_stats.items())[:50]):
        def safe(v):
            try:
                v = float(v)
                return 0.0 if (np.isnan(v) or np.isinf(v)) else v
            except: return 0.0
        matrix[i][0] = safe(s.get("null_ratio", 0))
        matrix[i][1] = safe(s.get("unique_ratio", 0))
        matrix[i][2] = 1.0 if "mean" in s else 0.0
        matrix[i][3] = safe(s.get("mean", 0))
        matrix[i][4] = safe(s.get("std", 0))
        matrix[i][5] = safe(s.get("max", 0))
    return matrix

def extract_features(df):
    col_names = list(df.columns)
    col_stats = {}
    for col in col_names[:50]:
        s = {"dtype": str(df[col].dtype), "null_ratio": round(df[col].isna().mean(), 3),
             "unique_ratio": round(df[col].nunique() / max(len(df), 1), 3)}
        if pd.api.types.is_numeric_dtype(df[col]):
            s.update({"mean": round(float(df[col].mean()), 3),
                      "std":  round(float(df[col].std()), 3),
                      "min":  round(float(df[col].min()), 3),
                      "max":  round(float(df[col].max()), 3)})
        else:
            vals = df[col].dropna().astype(str).unique()[:5].tolist()
            s["sample_values"] = vals
        col_stats[col] = s
    return {"column_names": col_names, "n_columns": len(col_names),
            "n_rows": len(df), "column_stats": col_stats}

def main():
    existing = json.loads(SAVE_FILE.read_text()) if SAVE_FILE.exists() else {}
    print(f"Pipeline basliyor — {len(DATASETS)} dataset")
    print("=" * 60)

    ok = err = 0
    for i, item in enumerate(DATASETS):
        ds_id  = item["id"]
        sector = item["sector"]
        key    = f"hf/{ds_id}"

        if key in existing:
            print(f"[{i+1}/{len(DATASETS)}] SKIP: {ds_id}")
            continue

        print(f"\n[{i+1}/{len(DATASETS)}] {ds_id} → {sector}")
        try:
            ds = load_dataset(ds_id, split="train", trust_remote_code=True)
            df = ds.to_pandas()

            if len(df) < MIN_ROWS:
                print(f"  x Kalite hatasi: too few rows ({len(df)})")
                err += 1; continue
            if len(df.columns) < MIN_COLS:
                print(f"  x Kalite hatasi: too few columns ({len(df.columns)})")
                err += 1; continue

            df = df.head(5000)
            features = extract_features(df)
            existing[key] = {"sector": sector, "column_names": features["column_names"], "features": features}
            # NaN kontrol
            import numpy as np
            cs = encode_stats_check(features["column_stats"])
            if np.isnan(cs).any() or np.isinf(cs).any():
                print(f"  x NaN/Inf stats, atlanıyor")
                err += 1; continue
            SAVE_FILE.write_text(json.dumps(existing, indent=2))
            print(f"  ok — {len(df)} rows, {len(df.columns)} cols")
            ok += 1
        except Exception as e:
            print(f"  x Hata: {e}")
            err += 1

    print(f"\n{'='*60}")
    print(f"TAMAMLANDI — basarili: {ok}, hatali: {err}")
    from collections import Counter
    counts = Counter(v["sector"] for v in existing.values())
    print("\nSektor dagilimi:")
    for s, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {s:20s}: {c}")

if __name__ == "__main__":
    main()

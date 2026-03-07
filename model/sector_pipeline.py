"""
SchemaLabs - Sector Detection Data Pipeline
Kullanim: python sector_pipeline.py
"""

import os
import json
import time
import pandas as pd
from pathlib import Path
import kaggle

RAW_DIR      = Path("data/raw")
RESULTS_FILE = Path("data/sector_labels.json")
RAW_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    {"id": "uciml/pima-indians-diabetes-database", "sector": "healthcare"},
    {"id": "andrewmvd/heart-failure-clinical-data", "sector": "healthcare"},
    {"id": "uciml/breast-cancer-wisconsin-data", "sector": "healthcare"},
    {"id": "fedesoriano/stroke-prediction-dataset", "sector": "healthcare"},
    {"id": "mlg-ulb/creditcardfraud", "sector": "finance"},
    {"id": "uciml/adult-census-income", "sector": "finance"},
    {"id": "olistbr/brazilian-ecommerce", "sector": "ecommerce"},
    {"id": "retailrocket/ecommerce-dataset", "sector": "ecommerce"},
    {"id": "carrie1/ecommerce-data", "sector": "ecommerce"},
    {"id": "shivamb/machine-predictive-maintenance-classification", "sector": "manufacturing"},
    {"id": "spscientist/students-performance-in-exams", "sector": "education"},
    {"id": "aljarah/xAPI-Edu-Data", "sector": "education"},
    {"id": "pavansubhasht/ibm-hr-analytics-attrition-dataset", "sector": "hr"},
    {"id": "rhuebner/human-resources-data-set", "sector": "hr"},
    {"id": "berkeleyearth/climate-change-earth-surface-temperature-data", "sector": "climate"},
    {"id": "sampadab17/network-intrusion-detection", "sector": "cybersecurity"},
    {"id": "mirichoi0218/insurance", "sector": "insurance"},
    {"id": "harlfoxem/housesalesprediction", "sector": "realestate"},
    {"id": "blastchar/telco-customer-churn", "sector": "telecom"},
    {"id": "rounakbanik/the-movies-dataset", "sector": "entertainment"},
    {"id": "shivamb/netflix-shows", "sector": "entertainment"},
    {"id": "shashwatwork/dataco-smart-supply-chain-for-big-data-analysis", "sector": "supplychain"},
    {"id": "uciml/iris", "sector": "agriculture"},
    {"id": "datasnaek/youtube-new", "sector": "marketing"},
    {"id": "stefanoleone992/ea-sports-fc-24-complete-player-dataset", "sector": "sports"},
    {"id": "secareanualin/football-events", "sector": "sports"},
    {"id": "saife245/english-premier-league", "sector": "sports"},
    {"id": "denkuznetz/traffic-accident-prediction", "sector": "transportation"},
    {"id": "devansodariya/road-accident-united-kingdom-uk-dataset", "sector": "transportation"},
    {"id": "mcamera/brazil-highway-traffic-accidents", "sector": "transportation"},
    {"id": "paultimothymooney/denver-crime-data", "sector": "government"},
    {"id": "aliafzal9323/chicago-crime-dataset-2024-2026", "sector": "government"},
    {"id": "saurabhbadole/crime-incidents-in-los-angeles-2020-to-present", "sector": "government"},
    {"id": "atharvaingle/crop-recommendation-dataset", "sector": "agriculture"},
    {"id": "bhadramohit/agriculture-and-farming-dataset", "sector": "agriculture"},
    {"id": "waqi786/climate-change-impact-on-agriculture", "sector": "agriculture"},
    {"id": "samithsachidanandan/german-power-consumption", "sector": "energy"},
    {"id": "nelgiriyewithana/global-weather-repository",              "sector": "climate"},
    {"id": "sudalairajkumar/daily-temperature-of-major-cities",        "sector": "climate"},
    {"id": "nkongolo/ugransome-dataset",                               "sector": "cybersecurity"},
    {"id": "rishikumarrajvansh/cyber-security",                        "sector": "cybersecurity"},
    {"id": "litvinenko630/insurance-claims",                           "sector": "insurance"},
    {"id": "rupakroy/auto-insurance",                                  "sector": "insurance"},
    {"id": "nelgiriyewithana/new-york-housing-market",                 "sector": "realestate"},
    {"id": "ahmedshahriarsakib/usa-real-estate-dataset",               "sector": "realestate"},
    {"id": "oluwademiladeadeniyi/mtn-nigeria-customer-churn",          "sector": "telecom"},
    {"id": "ashishkumarsingh123/telecom-churn-dataset",                "sector": "telecom"},
    {"id": "ziya07/smart-logistics-supply-chain-dataset",              "sector": "supplychain"},
    {"id": "keyushnisar/global-product-inventory-dataset-2025",        "sector": "supplychain"},
    {"id": "sinderpreet/analyze-the-marketing-spending",               "sector": "marketing"},
    {"id": "mahmoudshaheen1134/sales-and-advertising-clean-dataset",   "sector": "marketing"},
    {"id": "anshtanwar/global-data-on-sustainable-energy",             "sector": "energy"},
    {"id": "ahmeduzaki/wind-and-solar-energy-production-dataset",      "sector": "energy"},
    {"id": "rabieelkharoua/predicting-manufacturing-defects-dataset",  "sector": "manufacturing"},

]

def quality_check(df):
    if df is None or len(df) < 100:
        return False, "too few rows"
    if len(df.columns) < 4:
        return False, "too few columns"
    meaningless = sum(1 for c in df.columns if c.lower() in ['col','x','unnamed'] or c.startswith('col_') or c.startswith('feature_'))
    if meaningless / len(df.columns) > 0.5:
        return False, "meaningless columns"
    return True, "ok"

def extract_features(df):
    features = {
        "column_names": list(df.columns),
        "n_columns": len(df.columns),
        "n_rows": len(df),
        "column_stats": {}
    }
    for col in df.columns[:20]:
        col_data = df[col].dropna()
        stats = {
            "dtype": str(df[col].dtype),
            "null_ratio": round(df[col].isna().mean(), 3),
            "unique_ratio": round(df[col].nunique() / max(len(df), 1), 3),
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            stats["mean"] = round(float(col_data.mean()), 3) if len(col_data) > 0 else 0
            stats["std"]  = round(float(col_data.std()),  3) if len(col_data) > 0 else 0
            stats["min"]  = round(float(col_data.min()),  3) if len(col_data) > 0 else 0
            stats["max"]  = round(float(col_data.max()),  3) if len(col_data) > 0 else 0
        else:
            stats["sample_values"] = [str(v)[:30] for v in col_data.head(3).tolist()]
        features["column_stats"][col] = stats
    return features

def run():
    results = {}
    if RESULTS_FILE.exists():
        results = json.loads(RESULTS_FILE.read_text())

    print(f"Pipeline basliyor — {len(DATASETS)} dataset")
    print("=" * 60)

    success = 0
    failed  = 0

    for i, ds in enumerate(DATASETS):
        ds_id   = ds["id"]
        sector  = ds["sector"]

        if ds_id in results:
            print(f"[{i+1}/{len(DATASETS)}] SKIP: {ds_id}")
            success += 1
            continue

        print(f"\n[{i+1}/{len(DATASETS)}] {ds_id} → {sector}")

        try:
            dl_path = RAW_DIR / ds_id.replace("/", "_")
            dl_path.mkdir(parents=True, exist_ok=True)
            kaggle.api.dataset_download_files(ds_id, path=str(dl_path), unzip=True, quiet=True)

            csv_files = list(dl_path.rglob("*.csv")) + list(dl_path.rglob("*.txt"))
            if not csv_files:
                print(f"  x CSV bulunamadi")
                failed += 1
                continue

            csv_path = max(csv_files, key=lambda f: f.stat().st_size)

            try:
                df = pd.read_csv(csv_path, nrows=5000, on_bad_lines='skip')
            except Exception:
                df = pd.read_csv(csv_path, nrows=5000, encoding='latin1', on_bad_lines='skip')

            ok, reason = quality_check(df)
            if not ok:
                print(f"  x Kalite hatasi: {reason}")
                failed += 1
                continue

            features = extract_features(df)

            results[ds_id] = {
                "sector":       sector,
                "n_rows":       features["n_rows"],
                "n_columns":    features["n_columns"],
                "column_names": features["column_names"],
                "csv_path":     str(csv_path),
                "features":     features
            }

            RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
            RESULTS_FILE.write_text(json.dumps(results, indent=2))

            print(f"  ok — {features['n_rows']} rows, {features['n_columns']} cols")
            success += 1
            time.sleep(0.3)

        except Exception as e:
            print(f"  x Hata: {e}")
            failed += 1
            continue

    print("\n" + "=" * 60)
    print(f"TAMAMLANDI — basarili: {success}, hatali: {failed}")

    sector_counts = {}
    for v in results.values():
        s = v["sector"]
        sector_counts[s] = sector_counts.get(s, 0) + 1

    print("\nSektor dagilimi:")
    for s, c in sorted(sector_counts.items(), key=lambda x: -x[1]):
        print(f"  {s:20s}: {c}")

if __name__ == "__main__":
    run()

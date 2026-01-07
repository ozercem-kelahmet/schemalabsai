import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

OUTPUT_DIR = Path('../training_50x50')
OUTPUT_DIR.mkdir(exist_ok=True)
SAMPLES_PER_SUBSECTOR = 50000

def get_classes(subsector):
    return [f"a_{subsector}_low", f"b_{subsector}_medlow", f"c_{subsector}_med", f"d_{subsector}_medhigh", f"e_{subsector}_high"]

def generate_subsector(sector, subsector, class_names, sector_idx, subsector_idx, n=SAMPLES_PER_SUBSECTOR):
    global_idx = sector_idx * 50 + subsector_idx
    base_offset = global_idx * 100
    sorted_classes = sorted(class_names)
    class_to_idx = {c: i for i, c in enumerate(sorted_classes)}
    n_classes = len(class_names)
    per_class = n // n_classes
    patterns = [
        (0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.88),
        (0.85, 0.95, 0.80, 0.75, 0.90, 0.70, 0.82),
        (0.80, 0.85, 0.95, 0.70, 0.85, 0.90, 0.78),
        (0.75, 0.80, 0.85, 0.95, 0.80, 0.85, 0.75),
        (0.70, 0.75, 0.80, 0.85, 0.95, 0.80, 0.72),
    ]
    data = []
    for cls in class_names:
        idx = class_to_idx[cls]
        pattern = patterns[idx % len(patterns)]
        base = base_offset + idx * 20
        ceiling = base + 12
        for _ in range(per_class):
            primary = np.random.uniform(base, ceiling)
            data.append({
                'primary_score': primary,
                'secondary_score': primary * pattern[0] + np.random.uniform(-3, 3),
                'tertiary_score': primary * pattern[1] + np.random.uniform(-3, 3),
                'risk_index': primary * pattern[2] + np.random.uniform(-3, 3),
                'severity_level': primary * pattern[3] + np.random.uniform(-3, 3),
                'duration_factor': primary * pattern[4] + np.random.uniform(-3, 3),
                'frequency_rate': primary * pattern[5] + np.random.uniform(-3, 3),
                'intensity_score': primary * pattern[6] + np.random.uniform(-3, 3),
                'recovery_index': 100 - primary * 0.1 + np.random.uniform(-2, 2),
                'response_rate': 100 - primary * 0.1 + np.random.uniform(-2, 2),
                'sector': sector,
                'subsector': subsector,
                'target': cls
            })
    return pd.DataFrame(data).sample(frac=1, random_state=42).reset_index(drop=True)

def validate_data(df, target_col):
    feature_cols = ['primary_score', 'secondary_score', 'tertiary_score', 'risk_index',
                    'severity_level', 'duration_factor', 'frequency_rate', 'intensity_score',
                    'recovery_index', 'response_rate']
    X = df[feature_cols].values
    y = pd.factorize(df[target_col])[0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = (y_pred == y_test).mean()
    f1 = f1_score(y_test, y_pred, average='weighted')
    passed = acc >= 0.70 and f1 >= 0.70
    return acc, f1, 0, 0, passed

SECTORS_PART3 = {
    "biotech": ["genomics", "proteomics", "metabolomics", "transcriptomics", "bioinformatics", "computational_bio", "synthetic_bio", "gene_therapy", "cell_therapy", "car_t", "stem_cells", "regenerative", "tissue_eng", "organ_bio", "crispr", "gene_editing", "sequencing", "pcr", "microarray", "mass_spec", "chromatography", "electrophoresis", "microscopy", "imaging_bio", "flow_cytometry", "cell_culture", "fermentation", "bioprocessing", "upstream_bio", "downstream_bio", "purification", "formulation_bio", "analytics_bio", "quality_bio", "regulatory_bio", "cmo", "cdmo", "cro", "preclinical_bio", "toxicology", "pharmacology", "biomarker_bio", "diagnostics_bio", "ivd", "poct", "molecular_dx", "immunoassay", "pathology_bio", "histology", "cytology"],
    "construction": ["residential_const", "single_family", "multi_family", "condo", "townhouse", "commercial_const", "office_const", "retail_const", "hospitality_const", "healthcare_const", "education_const", "industrial_const", "warehouse_const", "manufacturing_const", "infrastructure", "roads", "bridges", "tunnels", "railways_const", "airports_const", "ports_const", "water_const", "wastewater_const", "power_const", "telecom_const", "oil_gas_const", "mining_const", "renovation_const", "restoration", "demolition", "excavation", "foundation", "structural_const", "framing", "roofing", "siding", "windows_const", "doors_const", "flooring_const", "ceiling", "drywall", "painting_const", "plumbing_const", "electrical_const", "hvac_const", "fire_protection", "elevators", "escalators", "landscaping", "paving"],
    "mining": ["gold_mining", "silver_mining", "platinum_mining", "palladium_mining", "copper_mining", "iron_ore", "bauxite", "zinc_mining", "lead_mining", "nickel_mining", "tin_mining", "titanium_mining", "lithium_mining", "cobalt_mining", "rare_earths", "uranium_mining", "coal_mining_min", "potash", "phosphate", "salt_mining", "sand_gravel", "limestone", "marble", "granite", "slate", "gypsum", "clay", "silica", "talc", "mica", "graphite", "diamonds", "gemstones", "exploration_min", "drilling_min", "blasting", "extraction", "crushing", "grinding_min", "flotation", "leaching", "smelting", "refining_min", "tailings", "reclamation", "mine_safety", "mine_ventilation", "mine_dewatering", "mine_equipment", "mine_services"],
    "oil_gas": ["exploration_og", "seismic", "drilling_og", "completion", "production_og", "workover", "stimulation", "fracturing", "acidizing", "cementing", "logging", "testing_og", "artificial_lift", "esp", "rod_pump", "gas_lift", "plunger", "gathering", "processing_og", "treating", "compression", "pipeline_og", "storage_og", "terminaling", "loading", "shipping_og", "trading_og", "marketing_og", "refining_og", "distillation", "cracking", "reforming", "blending", "additives", "lubricants_og", "asphalt", "petrochemicals", "olefins", "aromatics", "polymers", "plastics_og", "fertilizers_og", "lng_og", "lpg", "cng", "ngl", "condensate", "crude_grades", "benchmarks", "hedging"],
    "utilities": ["electric_utility", "gas_utility", "water_utility", "wastewater_utility", "steam_utility", "district_cooling", "district_heating", "generation", "transmission_util", "distribution_util", "retail_util", "wholesale_util", "merchant_power", "ipp", "regulated_util", "deregulated", "municipal_util", "cooperative", "public_power", "renewable_util", "nuclear_util", "fossil_util", "hydro_util", "solar_util", "wind_util", "battery_util", "smart_meter", "ami", "grid_modernization", "demand_mgmt", "energy_eff", "weatherization", "low_income", "rate_design", "tariffs", "regulatory_util", "puc", "ferc", "iso_rto", "capacity_market", "energy_market", "ancillary", "congestion", "curtailment", "interconnection", "wheeling", "net_metering", "community_solar", "green_tariff", "ppa"],
    "transport": ["trucking", "ltl", "ftl", "drayage_trans", "intermodal_trans", "rail_freight", "class_1", "short_line", "passenger_rail", "commuter_rail", "light_rail", "subway", "streetcar", "bus_transit", "brt", "paratransit", "rideshare_trans", "carshare_trans", "bikeshare", "scooter", "taxi", "limo_trans", "charter_trans", "shuttle", "school_bus", "motorcoach", "airlines", "cargo_air", "charter_air", "fractional", "fbo", "ground_handling", "shipping_trans", "container_ship", "bulk_carrier", "tanker_ship", "roro_ship", "ferry", "cruise_trans", "yacht", "barge", "tugboat", "port_ops", "terminal_ops", "stevedoring", "freight_broker", "customs_broker", "nvocc", "freight_audit", "fleet_trans"],
    "media": ["broadcast_tv", "cable_tv", "streaming_media", "ott", "svod", "avod", "tvod", "live_tv", "news_media", "sports_media_m", "entertainment_media", "documentary", "reality_tv", "scripted", "animation_media", "radio_media", "terrestrial", "satellite_radio", "internet_radio", "podcast_media", "newspapers", "magazines", "digital_pub", "newsletters", "wire_services", "photo_agency", "video_agency", "content_studio", "production_media", "post_production", "distribution_media", "syndication", "licensing_media", "rights_mgmt", "ad_sales", "programmatic", "native_ads", "branded_content", "influencer", "social_media", "ugc", "community_mgmt", "moderation", "analytics_media", "measurement", "attribution", "ad_tech", "mar_tech", "pr_media", "corp_comm"],
    "advertising": ["creative_agency", "media_agency", "digital_agency", "social_agency", "pr_agency", "branding", "design_agency", "production_agency", "experiential", "shopper_mkt", "trade_marketing", "direct_marketing", "crm_agency", "loyalty_agency", "promotional", "sponsorship", "event_mkt", "sports_mkt", "entertainment_mkt", "cause_mkt", "influencer_mkt", "content_mkt", "seo", "sem", "ppc", "display", "video_ads", "audio_ads", "native_agency", "programmatic_agency", "dsp", "ssp", "dmp", "ad_network", "ad_exchange", "ad_server", "verification", "viewability", "brand_safety", "fraud_detection", "attribution_agency", "analytics_agency", "research_agency", "insights", "strategy_agency", "consulting_agency", "tech_agency", "data_agency", "media_buying", "planning"],
    "legal": ["corporate_law", "securities_law", "ma_law", "private_equity_law", "venture_law", "banking_law", "finance_law", "restructuring", "bankruptcy_law", "tax_law", "estate_planning", "trusts", "probate", "real_estate_law", "construction_law", "environmental_law", "energy_law", "ip_law", "patent", "trademark", "copyright_law", "trade_secret", "licensing_law", "litigation", "trial", "appellate", "arbitration", "mediation", "class_action", "mass_tort", "product_liability_law", "medical_malpractice", "personal_injury", "insurance_law", "employment_law", "labor_law", "immigration_law", "criminal_law", "white_collar", "regulatory_law", "compliance_law", "antitrust", "trade_law", "international_law", "government", "lobbying", "public_policy", "family_law", "elder_law", "civil_rights"],
    "consulting": ["strategy_consulting", "management_consulting", "operations_consulting", "it_consulting", "digital_consulting", "technology_consulting", "data_consulting", "analytics_consulting", "ai_consulting", "cybersecurity_consulting", "cloud_consulting", "transformation", "change_consulting", "org_consulting", "hr_consulting", "talent_consulting", "leadership_consulting", "financial_consulting", "risk_consulting", "compliance_consulting", "audit_consulting", "tax_consulting", "valuation", "transaction_consulting", "restructuring_consulting", "turnaround", "supply_chain_consulting", "procurement_consulting", "manufacturing_consulting", "sustainability_consulting", "esg", "healthcare_consulting", "pharma_consulting", "life_sciences", "energy_consulting", "utilities_consulting", "telecom_consulting", "media_consulting", "retail_consulting", "consumer_consulting", "fsi", "banking_consulting", "insurance_consulting", "wealth_consulting", "public_sector", "defense_consulting", "aerospace_consulting", "travel_consulting", "hospitality_consulting", "education_consulting"],
}

def process_sector(sector_name, subsectors, sector_idx):
    print(f"\n{'='*60}")
    print(f"PROCESSING: {sector_name.upper()} (sector_idx={sector_idx})")
    print(f"{'='*60}")
    
    all_dfs = []
    stats = {'passed': 0, 'failed': 0}
    
    for subsector_idx, subsector in enumerate(subsectors):
        classes = get_classes(subsector)
        df = generate_subsector(sector_name, subsector, classes, sector_idx, subsector_idx)
        acc, f1, corr, imp, passed = validate_data(df, 'target')
        
        status = "✓" if passed else "✗"
        if subsector_idx % 10 == 0:
            print(f"  {status} {subsector}: acc={acc:.2f} f1={f1:.2f}")
        
        if passed:
            stats['passed'] += 1
        else:
            stats['failed'] += 1
        
        all_dfs.append(df)
    
    sector_df = pd.concat(all_dfs, ignore_index=True)
    output_file = OUTPUT_DIR / f'{sector_name}.parquet'
    sector_df.to_parquet(output_file, index=False)
    
    print(f"\n  SAVED: {output_file}")
    print(f"  Rows: {len(sector_df):,}")
    print(f"  Passed: {stats['passed']}/{stats['passed']+stats['failed']}")
    
    return stats

if __name__ == "__main__":
    print("=" * 70)
    print("PART 3: GENERATING 10 SECTORS x 50 SUBSECTORS")
    print("Sectors: biotech, construction, mining, oil_gas, utilities,")
    print("         transport, media, advertising, legal, consulting")
    print("=" * 70)
    
    total_stats = {'passed': 0, 'failed': 0}
    
    for local_idx, (sector_name, subsectors) in enumerate(SECTORS_PART3.items()):
        sector_idx = local_idx + 20
        stats = process_sector(sector_name, subsectors, sector_idx)
        total_stats['passed'] += stats['passed']
        total_stats['failed'] += stats['failed']
    
    print("\n" + "=" * 70)
    print("PART 3 COMPLETE")
    print("=" * 70)
    print(f"Total Passed: {total_stats['passed']}")
    print(f"Total Failed: {total_stats['failed']}")

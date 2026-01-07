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

SECTORS_PART1 = {
    "healthcare": ["dental", "cardiology", "oncology", "neurology", "orthopedics", "dermatology", "ophthalmology", "pediatrics", "geriatrics", "psychiatry", "radiology", "pathology", "anesthesiology", "emergency", "surgery", "urology", "nephrology", "pulmonology", "gastroenterology", "endocrinology", "rheumatology", "immunology", "infectious", "genetics", "nutrition", "rehabilitation", "palliative", "sports_med", "occupational", "preventive", "family_med", "internal_med", "plastic_surgery", "vascular", "thoracic", "transplant", "bariatric", "fertility", "neonatal", "maternal", "allergy", "sleep_med", "pain_mgmt", "wound_care", "dialysis", "home_health", "telemedicine", "pharmacy", "laboratory", "blood_bank"],
    "finance": ["retail_banking", "corporate_banking", "investment_banking", "private_banking", "credit_cards", "mortgages", "auto_loans", "personal_loans", "student_loans", "microfinance", "wealth_mgmt", "asset_mgmt", "hedge_funds", "mutual_funds", "etf", "pension_funds", "insurance_fin", "reinsurance", "stock_trading", "forex", "commodities", "derivatives", "bonds", "venture_capital", "private_equity", "crowdfunding", "payments", "remittance", "treasury", "risk_mgmt", "compliance_fin", "audit", "tax_services", "accounting", "financial_planning", "credit_scoring", "collections", "bankruptcy", "mergers", "ipo", "underwriting", "brokerage", "custody", "clearing", "settlement", "fintech", "crypto", "blockchain_fin", "regtech", "insurtech"],
    "manufacturing": ["automotive_mfg", "aerospace_mfg", "electronics_mfg", "semiconductors", "machinery", "chemicals_mfg", "pharmaceuticals_mfg", "food_processing", "beverages_mfg", "textiles_mfg", "apparel_mfg", "furniture_mfg", "paper_mfg", "plastics_mfg", "rubber_mfg", "glass_mfg", "ceramics_mfg", "cement_mfg", "steel_mfg", "aluminum_mfg", "copper_mfg", "metals_mfg", "wood_mfg", "printing_mfg", "packaging_mfg", "medical_devices", "consumer_goods", "industrial_equip", "defense_mfg", "shipbuilding", "rail_equip", "power_equip", "hvac_mfg", "lighting_mfg", "batteries_mfg", "solar_mfg", "wind_mfg", "robotics_mfg", "automation_mfg", "3d_printing", "composites", "coatings", "adhesives", "lubricants", "fertilizers", "pesticides", "cosmetics_mfg", "cleaning_mfg", "pet_food", "tobacco"],
    "logistics": ["freight_road", "freight_rail", "freight_air", "freight_sea", "courier", "express", "postal_log", "warehousing", "cold_storage", "hazmat", "bulk_log", "container", "tanker_log", "roro", "breakbulk", "project_cargo", "last_mile", "reverse_logistics", "cross_docking", "fulfillment", "inventory_mgmt", "procurement", "sourcing", "customs", "freight_forward", "3pl", "4pl", "drayage", "intermodal", "transload", "consolidation", "deconsolidation", "pick_pack", "kitting", "labeling", "packaging_log", "palletizing", "crating", "load_planning", "route_optimization", "fleet_mgmt", "telematics_log", "track_trace", "yard_mgmt", "dock_scheduling", "wms", "tms", "oms", "demand_planning", "supply_planning"],
    "retail": ["grocery", "supermarket", "hypermarket", "convenience", "discount", "dollar_store", "warehouse_club", "specialty_food", "organic", "gourmet", "liquor", "tobacco_ret", "pharmacy_ret", "health_beauty", "cosmetics_ret", "personal_care", "apparel_ret", "footwear", "accessories", "jewelry", "watches", "eyewear", "sporting_goods", "outdoor", "fitness_ret", "toys", "games", "hobbies", "books", "music_ret", "movies", "electronics_ret", "appliances", "furniture_ret", "home_decor", "bedding", "bath", "kitchen", "garden", "hardware", "paint", "flooring", "lighting_ret", "pet_supplies", "automotive_ret", "office_supplies", "art_supplies", "craft", "party", "seasonal"],
    "telecom": ["mobile_carrier", "fixed_line", "broadband", "fiber", "cable", "satellite_tel", "voip", "unified_comm", "contact_center", "pbx", "sip_trunk", "toll_free", "local_number", "long_distance", "international", "roaming", "mvno", "tower_infra", "small_cell", "das", "wifi", "private_network", "sd_wan", "mpls", "vpn_tel", "ethernet", "wavelength", "dark_fiber", "colocation", "interconnect", "peering", "transit", "cdn", "edge_compute", "iot_platform", "m2m", "nb_iot", "lte_m", "5g_infra", "network_security", "ddos", "firewall_tel", "siem", "noc", "soc", "oss", "bss", "billing_tel", "crm_tel", "workforce_mgmt"],
    "insurance": ["life_ins", "term_life", "whole_life", "universal_life", "variable_life", "annuities", "health_ins", "medical_ins", "dental_ins", "vision_ins", "disability", "long_term_care", "medicare", "medicaid", "auto_ins", "homeowners", "renters_ins", "flood_ins", "earthquake_ins", "umbrella", "liability", "professional_liability", "malpractice", "directors_officers", "errors_omissions", "cyber_ins", "property_ins", "commercial_property", "business_interruption", "workers_comp", "general_liability", "product_liability", "marine_ins", "cargo_ins", "hull_ins", "aviation_ins", "travel_ins", "pet_ins", "wedding_ins", "event_ins", "title_ins", "mortgage_ins", "credit_ins", "warranty_ins", "gap_ins", "reinsurance_ins", "captive_ins", "risk_retention", "surplus_lines", "claims_mgmt"],
    "hr": ["recruiting", "talent_acquisition", "sourcing_hr", "screening", "interviewing", "onboarding", "orientation", "training_hr", "learning_dev", "leadership_dev", "coaching", "mentoring", "performance_mgmt", "goal_setting", "reviews", "feedback", "recognition", "rewards", "compensation", "salary_planning", "bonus", "commission", "equity_comp", "benefits_admin", "health_benefits", "retirement", "wellness", "eap", "leave_mgmt", "time_attendance", "scheduling_hr", "payroll", "tax_hr", "compliance_hr", "labor_relations", "unions", "grievance", "discipline", "termination", "severance", "outplacement", "alumni", "employer_brand", "evp", "hris", "workforce_planning", "analytics_hr", "org_design", "change_mgmt", "remote_work"],
    "realestate": ["residential_sale", "residential_rent", "commercial_sale", "commercial_rent", "industrial_sale", "industrial_rent", "land_re", "development_re", "construction_re", "renovation_re", "interior_design", "architecture_re", "urban_planning", "zoning", "permitting", "title_re", "escrow", "mortgage_re", "refinancing", "foreclosure", "auction_re", "appraisal_re", "valuation", "inspection", "surveying", "property_mgmt", "facility_mgmt", "maintenance_re", "security_re", "leasing", "tenant_rep", "landlord_rep", "investment_re", "reit", "syndication", "crowdfunding_re", "senior_living", "student_housing", "affordable", "luxury_re", "vacation_rental", "timeshare_re", "coworking", "coliving", "flex_space", "mixed_use", "retail_re", "office_re", "hotel_re", "hospital_re"],
    "ecommerce": ["b2c", "b2b_ec", "c2c", "d2c", "marketplace_ec", "aggregator", "comparison", "cashback", "coupons", "deals", "flash_sales", "group_buying", "auction_ec", "classifieds", "listings", "dropshipping", "private_label", "white_label", "subscription_ec", "rental_ec", "resale_ec", "wholesale_ec", "food_delivery", "grocery_delivery", "pharmacy_ec", "alcohol_delivery", "flower_delivery", "gift_delivery", "furniture_ec", "fashion_ec", "electronics_ec", "beauty_ec", "health_ec", "sports_ec", "toys_ec", "books_ec", "music_ec", "video_ec", "gaming_ec", "software_ec", "digital_goods", "nft", "virtual_goods", "services_ec", "freelance", "gig", "travel_ec", "ticketing", "booking", "reservations"],
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
    print("PART 1: GENERATING 10 SECTORS x 50 SUBSECTORS")
    print("Sectors: healthcare, finance, manufacturing, logistics, retail,")
    print("         telecom, insurance, hr, realestate, ecommerce")
    print("=" * 70)
    
    total_stats = {'passed': 0, 'failed': 0}
    
    for sector_idx, (sector_name, subsectors) in enumerate(SECTORS_PART1.items()):
        stats = process_sector(sector_name, subsectors, sector_idx)
        total_stats['passed'] += stats['passed']
        total_stats['failed'] += stats['failed']
    
    print("\n" + "=" * 70)
    print("PART 1 COMPLETE")
    print("=" * 70)
    print(f"Total Passed: {total_stats['passed']}")
    print(f"Total Failed: {total_stats['failed']}")

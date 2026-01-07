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

SECTORS_PART2 = {
    "sports": ["football", "basketball", "baseball", "soccer", "hockey", "tennis", "golf", "boxing", "mma", "wrestling", "volleyball", "swimming", "track_field", "gymnastics", "cycling", "motorsports", "horse_racing", "skiing", "snowboarding", "skateboarding", "surfing", "cricket", "rugby", "lacrosse", "field_hockey", "badminton", "table_tennis", "archery", "fencing", "rowing", "sailing", "kayaking", "climbing", "triathlon", "marathon", "crossfit", "weightlifting", "powerlifting", "bodybuilding", "yoga", "pilates", "martial_arts", "esports", "fantasy_sports", "sports_betting", "sports_media", "sports_marketing", "sports_medicine", "sports_tech", "sports_facility"],
    "technology": ["software_dev", "web_dev", "mobile_dev", "cloud_computing", "devops", "cybersecurity", "data_science", "machine_learning", "ai_tech", "blockchain", "iot_tech", "ar_vr", "quantum_computing", "edge_computing", "serverless", "microservices", "api_dev", "database_tech", "networking_tech", "system_admin", "it_support", "help_desk", "managed_services", "consulting_tech", "integration", "erp", "crm_tech", "hrms", "scm_tech", "plm", "bi_analytics", "data_warehouse", "etl", "data_governance", "master_data", "testing_qa", "automation_tech", "rpa", "low_code", "no_code", "saas", "paas", "iaas", "fintech_tech", "healthtech", "edtech", "proptech", "agtech", "cleantech", "martech"],
    "energy": ["oil_exploration", "oil_production", "oil_refining", "natural_gas", "lng", "pipeline", "oil_trading", "gas_trading", "coal_mining", "coal_trading", "nuclear_power", "solar_energy", "wind_energy", "hydro_energy", "geothermal", "biomass", "biofuel", "hydrogen_energy", "fuel_cells", "battery_storage", "grid_storage", "smart_grid", "power_generation", "power_transmission", "power_distribution", "utility_retail", "energy_trading", "energy_risk", "energy_efficiency", "demand_response", "microgrids", "distributed_gen", "ev_charging", "carbon_capture", "carbon_trading", "emissions", "renewable_cert", "energy_consulting", "energy_finance", "energy_law", "energy_policy", "energy_research", "offshore_energy", "onshore_energy", "upstream", "midstream", "downstream", "oilfield_services", "drilling", "well_services"],
    "agriculture": ["crop_farming", "livestock", "dairy_farming", "poultry", "aquaculture", "horticulture", "viticulture", "organic_farming", "vertical_farming", "hydroponics", "aeroponics", "precision_ag", "agritech", "farm_equipment", "irrigation", "fertilizers_ag", "pesticides_ag", "seeds", "animal_feed", "veterinary", "farm_insurance", "farm_finance", "commodities_ag", "grain_trading", "livestock_trading", "cold_chain", "food_processing_ag", "food_packaging", "food_safety", "food_testing", "organic_cert", "fair_trade", "agri_consulting", "farm_mgmt", "land_mgmt", "soil_science", "crop_science", "animal_science", "ag_research", "ag_extension", "rural_dev", "ag_policy", "ag_subsidies", "ag_exports", "ag_imports", "farm_labor", "seasonal_work", "ag_coop", "farmers_market", "csa"],
    "education": ["k12", "elementary", "middle_school", "high_school", "charter_school", "private_school", "homeschool", "special_ed", "gifted_ed", "esl", "higher_ed", "community_college", "university", "graduate_school", "professional_school", "vocational", "trade_school", "online_learning", "mooc", "bootcamp", "tutoring", "test_prep", "college_counseling", "career_counseling", "ed_tech", "lms", "student_info", "assessment", "curriculum", "textbooks", "educational_toys", "stem_ed", "arts_ed", "music_ed", "physical_ed", "language_ed", "early_childhood", "daycare", "preschool", "afterschool", "summer_camp", "study_abroad", "student_exchange", "scholarships", "student_loans_ed", "ed_nonprofit", "ed_policy", "ed_research", "accreditation", "ed_consulting"],
    "entertainment": ["film_production", "tv_production", "streaming", "theatrical", "home_video", "animation", "vfx", "post_prod", "music_prod", "recording", "live_music", "concerts", "festivals", "dj", "radio_ent", "podcast", "audiobook", "gaming_ent", "console_gaming", "pc_gaming", "mobile_gaming", "vr_gaming", "ar_gaming", "esports_ent", "casino", "gambling", "lottery", "sports_betting_ent", "theme_parks", "water_parks", "zoos", "aquariums", "museums", "theaters", "comedy", "magic", "circus", "dance", "ballet", "opera", "symphony", "broadway", "talent_mgmt", "booking_ent", "promotion", "pr_ent", "marketing_ent", "licensing_ent", "merchandising_ent", "fan_engagement"],
    "hospitality": ["luxury_hotel", "boutique_hotel", "business_hotel", "resort", "motel", "hostel", "bnb", "vacation_rental", "timeshare_hosp", "extended_stay", "casino_hotel", "spa_hotel", "golf_resort", "ski_resort", "beach_resort", "eco_resort", "fine_dining", "casual_dining", "fast_casual", "qsr", "cafe", "bakery", "bar", "nightclub", "lounge", "food_truck", "catering", "banquet", "room_service", "concierge", "housekeeping", "front_desk", "reservations_hosp", "revenue_mgmt", "loyalty_hosp", "mice", "conference", "convention", "exhibition", "wedding", "event_planning", "tour_operator", "travel_agent", "cruise", "airline_hosp", "car_rental", "limo", "charter_hosp", "adventure_travel", "cultural_tourism"],
    "automotive": ["oem", "tier1", "tier2", "tier3", "aftermarket", "dealership", "used_cars", "cpo", "fleet_auto", "rental_auto", "leasing_auto", "financing_auto", "insurance_auto", "warranty_auto", "extended_warranty", "service_auto", "repair_auto", "body_shop", "paint_auto", "detailing", "tires", "wheels", "brakes", "suspension", "steering", "transmission", "engine_auto", "exhaust", "electrical_auto", "electronics_auto", "interior_auto", "exterior_auto", "glass_auto", "mirrors", "lighting_auto", "hvac_auto", "fuel_system", "cooling_auto", "battery_auto", "ev_auto", "hybrid_auto", "autonomous", "connected", "telematics_auto", "infotainment", "navigation_auto", "safety_auto", "adas", "motorsports_auto", "racing"],
    "aerospace": ["commercial_aircraft", "business_jets", "regional_jets", "helicopters", "uav_aero", "military_aircraft", "fighters", "bombers", "transports", "tankers", "trainers", "spacecraft", "satellites_aero", "launch_vehicles", "space_station", "lunar_aero", "mars_aero", "engines_aero", "turbines_aero", "propulsion_aero", "airframes", "wings", "fuselage", "empennage", "landing_gear", "avionics", "flight_controls", "navigation_aero", "communication_aero", "radar_aero", "sensors_aero", "weapons_aero", "missiles", "countermeasures", "interiors_aero", "seats_aero", "galleys", "lavatories", "cargo_aero", "mro", "maintenance_aero", "repair_aero", "overhaul", "parts_aero", "testing_aero", "certification_aero", "simulation_aero", "training_aero", "ground_support", "airport_equip"],
    "pharma": ["drug_discovery", "preclinical", "clinical_trials", "phase_1", "phase_2", "phase_3", "phase_4", "regulatory_pharma", "fda", "ema", "manufacturing_pharma", "api", "formulation", "packaging_pharma", "quality_pharma", "validation", "pharmacovigilance", "medical_affairs", "msl", "commercial_pharma", "sales_pharma", "marketing_pharma", "market_access", "pricing_pharma", "reimbursement", "health_economics", "real_world", "registry", "biomarker", "companion_dx", "precision_med", "oncology_pharma", "immunology_pharma", "neurology_pharma", "cardiology_pharma", "metabolic", "rare_disease", "orphan", "pediatric_pharma", "geriatric_pharma", "womens_health", "mens_health", "infectious_pharma", "vaccines", "biologics", "biosimilars", "generics", "otc", "supplements", "distribution_pharma"],
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
    print("PART 2: GENERATING 10 SECTORS x 50 SUBSECTORS")
    print("Sectors: sports, technology, energy, agriculture, education,")
    print("         entertainment, hospitality, automotive, aerospace, pharma")
    print("=" * 70)
    
    total_stats = {'passed': 0, 'failed': 0}
    
    for local_idx, (sector_name, subsectors) in enumerate(SECTORS_PART2.items()):
        sector_idx = local_idx + 10
        stats = process_sector(sector_name, subsectors, sector_idx)
        total_stats['passed'] += stats['passed']
        total_stats['failed'] += stats['failed']
    
    print("\n" + "=" * 70)
    print("PART 2 COMPLETE")
    print("=" * 70)
    print(f"Total Passed: {total_stats['passed']}")
    print(f"Total Failed: {total_stats['failed']}")

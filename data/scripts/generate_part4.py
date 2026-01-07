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

SECTORS_PART4 = {
    "security": ["physical_security", "access_control", "video_surveillance", "intrusion_detection", "alarm_systems", "guard_services", "patrol", "executive_protection", "event_security", "facility_security", "campus_security", "hospital_security", "retail_security", "banking_security", "critical_infra", "government_security", "military_security", "intelligence", "counterterrorism", "cybersecurity_sec", "network_security", "endpoint_security", "cloud_security", "app_security", "data_security", "identity_mgmt", "authentication", "encryption", "firewall_sec", "ids_ips", "siem_sec", "soc_sec", "incident_response", "forensics", "pen_testing", "vuln_mgmt", "threat_intel", "security_consulting", "risk_assessment", "compliance_sec", "audit_sec", "training_sec", "awareness", "background_check", "investigations", "loss_prevention", "fraud_sec", "aml", "sanctions", "privacy"],
    "environmental": ["environmental_consulting", "eia", "permitting_env", "compliance_env", "remediation", "site_assessment", "contamination", "groundwater", "soil_env", "air_quality", "emissions_env", "monitoring_env", "sampling", "testing_env", "analysis_env", "reporting_env", "auditing_env", "due_diligence", "brownfield", "superfund", "rcra", "cercla", "clean_air", "clean_water", "endangered_species", "wetlands", "coastal", "marine_env", "forestry", "wildlife", "conservation_env", "restoration_env", "mitigation", "offset", "carbon_env", "ghg", "sustainability_env", "esg_env", "reporting_esg", "rating", "certification_env", "leed", "well", "energy_star", "iso14001", "waste_env", "recycling_env", "circular", "lifecycle", "footprint"],
    "food_beverage": ["dairy_fb", "milk", "cheese", "yogurt", "ice_cream", "butter", "meat_fb", "beef", "pork", "poultry_fb", "lamb", "processed_meat", "seafood_fb", "fish", "shellfish", "frozen_seafood", "canned_seafood", "bakery", "bread", "pastry", "cakes", "cookies", "crackers", "snacks", "chips", "nuts", "dried_fruit", "candy", "chocolate", "gum", "beverages_fb", "soft_drinks", "juice", "water_fb", "coffee", "tea_fb", "energy_drinks", "sports_drinks", "alcoholic", "beer", "wine", "spirits", "ready_to_drink", "frozen_foods", "meals", "pizza", "vegetables_fb", "fruits_fb", "condiments", "sauces"],
    "textiles": ["cotton", "wool", "silk", "linen", "hemp_tex", "jute", "synthetic_fiber", "polyester", "nylon", "acrylic", "spandex", "rayon", "viscose", "modal", "lyocell", "yarn", "thread", "fabric", "woven", "knit", "nonwoven", "denim", "twill", "satin", "velvet", "lace", "embroidery", "printing_tex", "dyeing", "finishing", "coating_tex", "laminating", "technical_tex", "industrial_tex", "automotive_tex", "medical_tex", "geotextiles", "agrotextiles", "home_tex", "bedding_tex", "towels", "curtains", "upholstery", "carpet", "rugs", "apparel_tex", "sportswear", "workwear", "uniforms", "ppe_tex"],
    "chemicals": ["basic_chemicals", "petrochemicals_chem", "olefins_chem", "aromatics_chem", "methanol", "ammonia", "chlor_alkali", "industrial_gases", "specialty_chemicals", "adhesives_chem", "sealants", "coatings_chem", "paints", "inks", "dyes", "pigments", "catalysts", "surfactants", "polymers_chem", "plastics_chem", "rubber_chem", "elastomers", "resins", "composites_chem", "fibers_chem", "films", "additives_chem", "plasticizers", "flame_retardants", "stabilizers", "lubricants_chem", "coolants", "solvents", "cleaning_chem", "water_treatment", "oil_treatment", "mining_chem", "construction_chem", "electronics_chem", "pharma_chem", "agro_chem", "fertilizers_chem", "pesticides_chem", "herbicides", "fungicides", "seeds_chem", "biotech_chem", "fine_chemicals", "flavors", "fragrances"],
    "metals": ["iron_steel", "carbon_steel", "stainless", "alloy_steel", "tool_steel", "structural_steel", "flat_steel", "long_steel", "tubular", "wire_rod", "aluminum_met", "primary_aluminum", "secondary_aluminum", "rolled_aluminum", "extruded_aluminum", "cast_aluminum", "copper_met", "brass", "bronze", "nickel_met", "zinc_met", "lead_met", "tin_met", "titanium_met", "magnesium", "precious_metals", "gold_met", "silver_met", "platinum_met", "palladium_met", "rare_earths_met", "ferroalloys", "ferrochrome", "ferromanganese", "ferrosilicon", "scrap", "recycling_met", "foundry", "casting", "forging", "stamping", "machining", "welding", "heat_treatment", "surface_treatment", "plating", "coating_met", "testing_met", "certification_met", "trading_met"],
    "electronics": ["semiconductors_elec", "memory", "logic", "analog", "discrete", "optoelectronics", "sensors_elec", "mems", "displays", "lcd", "oled", "led_elec", "pcb", "rigid_pcb", "flex_pcb", "components", "passive", "active", "connectors", "switches", "relays", "capacitors", "resistors", "inductors", "transformers", "batteries_elec", "power_supplies", "ups", "inverters", "converters", "motors_elec", "generators_elec", "actuators", "control_systems", "plc", "scada_elec", "hmi", "instrumentation", "test_equipment", "measurement", "automation_elec", "robotics_elec", "consumer_elec", "mobile_elec", "computing", "networking_elec", "telecom_elec", "audio", "video_elec", "imaging"],
    "machinery": ["agricultural_machinery", "tractors", "harvesters", "planters", "sprayers", "construction_machinery", "excavators", "loaders", "bulldozers", "cranes", "forklifts", "mining_machinery", "drilling_machinery", "crushing_machinery", "conveying", "material_handling", "pumps", "compressors", "fans_blowers", "turbines", "engines_mach", "gearboxes_mach", "bearings", "seals", "hydraulics", "pneumatics", "valves", "actuators_mach", "machine_tools", "lathes", "mills", "grinders", "presses", "cutting", "forming", "joining", "additive_mach", "packaging_machinery", "filling", "sealing_mach", "labeling_mach", "wrapping", "palletizing_mach", "food_machinery", "textile_machinery", "printing_machinery", "paper_machinery", "plastics_machinery", "rubber_machinery", "woodworking"],
    "defense": ["land_systems", "tanks", "armored_vehicles", "artillery", "small_arms", "ammunition", "explosives", "missiles_def", "rockets", "guided_munitions", "air_defense", "naval_systems", "ships_def", "submarines", "torpedoes", "naval_weapons", "aircraft_def", "fighters_def", "bombers_def", "transports_def", "helicopters_def", "drones_def", "space_def", "satellites_def", "launch_def", "cyber_def", "electronic_warfare", "signals_intel", "communications_def", "c4isr", "surveillance_def", "reconnaissance", "targeting", "simulation_def", "training_def", "logistics_def", "maintenance_def", "support_services", "security_services", "consulting_def", "engineering_def", "testing_def", "certification_def", "export_def", "offset_def", "industrial_participation", "technology_transfer", "r_and_d", "innovation_def", "modernization"],
    "maritime": ["shipbuilding", "commercial_ships", "tankers_mar", "bulk_carriers_mar", "container_ships", "lng_carriers", "offshore_vessels", "tugboats_mar", "ferries_mar", "cruise_ships", "yachts_mar", "workboats", "naval_ships", "ship_repair", "ship_conversion", "ship_recycling", "marine_equipment", "propulsion_mar", "navigation_mar", "communication_mar", "safety_mar", "deck_equipment", "cargo_handling", "offshore_equipment", "drilling_mar", "production_mar", "fpso", "subsea", "marine_services", "crewing", "ship_management", "technical_mgmt", "commercial_mgmt", "chartering", "brokerage_mar", "finance_mar", "insurance_mar", "classification", "surveying_mar", "inspection_mar", "port_services", "pilotage", "towage", "mooring", "bunkering", "provisioning", "waste_mgmt_mar", "pollution_mar", "salvage", "law_mar"],
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
    print("PART 4: GENERATING 10 SECTORS x 50 SUBSECTORS")
    print("Sectors: security, environmental, food_beverage, textiles, chemicals,")
    print("         metals, electronics, machinery, defense, maritime")
    print("=" * 70)
    
    total_stats = {'passed': 0, 'failed': 0}
    
    for local_idx, (sector_name, subsectors) in enumerate(SECTORS_PART4.items()):
        sector_idx = local_idx + 30
        stats = process_sector(sector_name, subsectors, sector_idx)
        total_stats['passed'] += stats['passed']
        total_stats['failed'] += stats['failed']
    
    print("\n" + "=" * 70)
    print("PART 4 COMPLETE")
    print("=" * 70)
    print(f"Total Passed: {total_stats['passed']}")
    print(f"Total Failed: {total_stats['failed']}")

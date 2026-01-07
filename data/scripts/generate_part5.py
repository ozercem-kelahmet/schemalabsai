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

SECTORS_PART5 = {
    "aviation": ["commercial_aviation", "regional_aviation", "business_aviation", "general_aviation", "cargo_aviation", "charter_aviation", "fractional_aviation", "airlines_av", "legacy_carriers", "low_cost", "ultra_low_cost", "regional_carriers", "cargo_carriers", "airport_ops", "airport_mgmt", "terminal_ops", "ground_handling_av", "fueling", "catering_av", "cleaning_av", "security_av", "customs_av", "immigration_av", "air_traffic", "navigation_av", "communication_av", "surveillance_av", "weather_av", "flight_planning", "dispatch", "crew_scheduling", "maintenance_av", "line_maintenance", "heavy_maintenance", "component_repair", "engine_repair", "avionics_repair", "interior_refurb", "paint_av", "parts_av", "logistics_av", "gse", "training_av", "pilot_training", "cabin_crew", "maintenance_training", "safety_av", "compliance_av", "quality_av", "environmental_av"],
    "railways": ["passenger_rail", "high_speed", "intercity", "regional_rail", "commuter_rl", "urban_rail", "metro", "light_rail_rl", "streetcar_rl", "monorail", "freight_rail_rl", "intermodal_rl", "bulk_rail", "tank_cars", "auto_rack", "container_rail", "refrigerated_rail", "infrastructure_rl", "track", "switches", "signals_rl", "electrification", "stations", "yards", "terminals_rl", "rolling_stock", "locomotives", "passenger_cars", "freight_cars", "maintenance_rl", "mro_rl", "overhaul_rl", "modernization_rl", "operations_rl", "scheduling_rl", "dispatch_rl", "crew_mgmt_rl", "safety_rl", "positive_train", "crossing_safety", "security_rl", "technology_rl", "ticketing_rl", "reservations_rl", "loyalty_rl", "real_estate_rl", "development_rl", "concessions", "advertising_rl", "consulting_rl"],
    "postal": ["letter_mail", "priority_mail", "express_mail", "registered_mail", "certified_mail", "insured_mail", "international_mail", "parcel_post", "package_delivery", "express_parcel", "standard_parcel", "economy_parcel", "international_parcel", "ecommerce_postal", "fulfillment_postal", "returns_postal", "last_mile_postal", "rural_delivery", "po_boxes", "mail_forwarding", "change_address", "address_validation", "geocoding", "mail_processing", "sorting", "automation_postal", "transportation_postal", "air_postal", "ground_postal", "rail_postal", "retail_postal", "post_offices", "contract_units", "village_post", "self_service", "stamps", "packaging_postal", "mailing_supplies", "print_mail", "direct_mail", "marketing_mail", "business_mail", "publications", "nonprofit_mail", "political_mail", "government_mail", "secure_mail", "track_trace_postal", "proof_delivery", "signature"],
    "warehousing": ["distribution_center", "fulfillment_center", "cold_storage_wh", "frozen_storage", "climate_controlled", "hazmat_wh", "bonded_wh", "foreign_trade", "public_wh", "contract_wh", "dedicated_wh", "shared_wh", "cross_dock_wh", "transload_wh", "consolidation_wh", "deconsolidation_wh", "pick_pack_wh", "kitting_wh", "assembly_wh", "value_added", "inventory_wh", "cycle_counting", "physical_inventory", "slotting", "replenishment", "wave_planning", "labor_mgmt", "wms_wh", "automation_wh", "conveyor", "sortation", "as_rs", "agv", "amr", "goods_to_person", "pick_to_light", "voice_picking", "rf_scanning", "dock_mgmt", "yard_mgmt_wh", "appointment_scheduling", "trailer_tracking", "gate_mgmt", "security_wh", "fire_protection_wh", "pest_control", "sanitation", "maintenance_wh", "real_estate_wh", "development_wh"],
    "packaging": ["corrugated", "folding_carton", "rigid_box", "flexible_pkg", "pouches", "bags_pkg", "wraps", "shrink_film", "stretch_film", "blister_pkg", "clamshell", "thermoform", "injection_molded", "blow_molded", "glass_pkg", "bottles_pkg", "jars", "metal_pkg", "cans", "aerosol", "drums", "pails", "tubes_pkg", "closures", "caps", "lids", "labels_pkg", "pressure_sensitive", "shrink_sleeve", "in_mold", "rfid_pkg", "smart_pkg", "active_pkg", "modified_atmosphere", "vacuum_pkg", "aseptic", "retort", "sustainable_pkg", "recyclable", "compostable", "biodegradable", "reusable", "refillable", "lightweighting", "source_reduction", "recycled_content", "design_pkg", "structural_pkg", "graphics_pkg", "prototyping"],
    "printing": ["offset_printing", "sheetfed", "web_offset", "heatset", "coldset", "digital_printing", "inkjet", "electrophotography", "wide_format", "grand_format", "flexography", "narrow_web", "wide_web", "corrugated_print", "gravure", "publication", "packaging_print", "screen_printing", "textile_print", "industrial_print", "specialty_print", "security_print", "banknotes", "passports", "id_cards", "labels_print", "tags", "tickets", "forms", "envelopes", "direct_mail_print", "transactional", "transpromo", "books", "magazines_print", "catalogs", "newspapers", "inserts", "flyers", "brochures", "posters", "signage_print", "displays", "pop", "banners", "vehicle_wrap", "wall_graphics", "floor_graphics", "3d_print", "prepress"],
    "recycling": ["paper_recycling", "cardboard", "mixed_paper", "office_paper", "newsprint", "plastic_recycling", "pet", "hdpe", "ldpe", "pp", "ps", "mixed_plastics", "metal_recycling", "aluminum_rec", "steel_rec", "copper_rec", "brass_rec", "precious_rec", "ewaste", "computers", "phones", "tvs", "appliances_rec", "batteries_rec", "lead_acid", "lithium_rec", "glass_recycling", "container_glass", "flat_glass", "textile_recycling", "clothing", "shoes", "carpet_rec", "organic_recycling", "food_waste", "yard_waste", "wood_recycling", "pallets", "lumber", "tire_recycling", "crumb_rubber", "tire_derived", "construction_rec", "concrete", "asphalt", "drywall", "automotive_rec", "catalytic", "oil_rec", "coolant"],
    "water": ["water_supply", "surface_water", "groundwater_water", "desalination_water", "water_treatment_w", "filtration_water", "disinfection_water", "fluoridation_water", "softening_water", "distribution_water", "transmission_water", "pumping_water", "storage_water", "tanks_water", "reservoirs_water", "towers_water", "metering_water", "billing_water", "customer_water", "conservation_water", "efficiency_water", "recycling_water", "reuse_water", "reclamation_water", "wastewater_w", "collection_water", "sewers_water", "lift_stations", "treatment_ww", "primary_ww", "secondary_ww", "tertiary_ww", "biosolids", "effluent", "discharge_water", "stormwater", "drainage_water", "detention", "retention_water", "green_infra", "lids_water", "monitoring_water", "scada_water", "gis_water", "asset_water", "maintenance_water", "regulatory_water", "compliance_water", "testing_water", "consulting_water"],
    "waste_mgmt": ["municipal_waste", "residential_waste", "commercial_waste", "industrial_waste", "construction_waste", "demolition_waste", "hazardous_wm", "medical_wm", "pharmaceutical_wm", "radioactive_wm", "electronic_wm", "universal_wm", "special_wm", "collection_wm", "curbside", "dumpster_wm", "compactor_wm", "roll_off", "front_load", "rear_load", "automated_wm", "transfer_wm", "stations_wm", "mrf_wm", "sorting_wm", "processing_wm", "baling_wm", "shredding_wm", "grinding_wm", "composting_wm", "anaerobic_wm", "digestion_wm", "landfill_wm", "sanitary_wm", "subtitle_d", "subtitle_c", "cells_wm", "liners_wm", "leachate", "gas_wm", "flares_wm", "lfg", "wte", "incineration_wm", "mass_burn", "rdf", "pyrolysis_wm", "gasification_wm", "plasma_wm", "recycling_wm"],
    "renewable": ["solar_ren", "pv", "monocrystalline", "polycrystalline", "thin_film_ren", "bifacial", "tracking_ren", "fixed_tilt", "rooftop_ren", "ground_mount", "floating_ren", "utility_scale", "commercial_ren", "residential_ren", "community_ren", "wind_ren", "onshore_ren", "offshore_ren", "turbines_ren", "blades_ren", "towers_ren", "nacelles", "gearboxes_ren", "generators_ren", "hydro_ren", "run_of_river", "reservoir_ren", "pumped_storage", "micro_hydro", "geothermal_ren", "flash_ren", "binary_ren", "egs", "biomass_ren", "wood_ren", "pellets_ren", "chips_ren", "agricultural_ren", "biogas_ren", "landfill_ren", "digester_ren", "biofuel_ren", "ethanol_ren", "biodiesel_ren", "renewable_diesel", "saf_ren", "hydrogen_ren", "green_h2", "blue_h2", "electrolysis"],
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
    return stats

if __name__ == "__main__":
    print("=" * 70)
    print("PART 5: GENERATING 10 SECTORS x 50 SUBSECTORS")
    print("=" * 70)
    total_stats = {'passed': 0, 'failed': 0}
    for local_idx, (sector_name, subsectors) in enumerate(SECTORS_PART5.items()):
        sector_idx = local_idx + 40
        stats = process_sector(sector_name, subsectors, sector_idx)
        total_stats['passed'] += stats['passed']
        total_stats['failed'] += stats['failed']
    print("\n" + "=" * 70)
    print("PART 5 COMPLETE")
    print(f"Total Passed: {total_stats['passed']}")
    print(f"Total Failed: {total_stats['failed']}")

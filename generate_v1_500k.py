#!/usr/bin/env python3
"""
SchemaV1 Production Data Generator
500K datasets (10000 sectors x 50 each)
Realistic column names, varied distributions, missing data, imbalance
"""
import json, os, random, math, time
from collections import Counter
import numpy as np
from pathlib import Path

random.seed(42)
np.random.seed(42)

BASE = Path(os.path.expanduser("~/Desktop/schemalabsai"))
OUTPUT = BASE / "data" / "v1_production_500k.json"

# Load sector hierarchy
with open(BASE / "data" / "sector_list_10000.json") as f:
    sector_data = json.load(f)
    ALL_SECTORS = sector_data["sectors"]
    HIERARCHY = sector_data["hierarchy"]

print(f"Total sectors: {len(ALL_SECTORS)}")
print(f"Main sectors: {len(HIERARCHY)}")

# ============================================================
# COLUMN NAME POOLS (sector-specific, realistic)
# ============================================================
GENERIC_COLUMNS = [
    "id", "date", "timestamp", "created_at", "updated_at", "name",
    "description", "status", "type", "category", "region", "country",
    "city", "state", "zip_code", "latitude", "longitude", "amount",
    "quantity", "price", "cost", "revenue", "profit", "margin",
    "rate", "ratio", "percentage", "score", "rating", "rank",
    "count", "total", "average", "median", "minimum", "maximum",
    "duration", "frequency", "weight", "height", "length", "width",
    "area", "volume", "temperature", "pressure", "velocity", "density"
]

SECTOR_COLUMN_POOLS = {
    "healthcare": [
        "patient_id", "patient_age", "gender", "blood_pressure_systolic",
        "blood_pressure_diastolic", "heart_rate", "bmi", "cholesterol",
        "glucose_level", "hemoglobin", "white_blood_cell_count", "platelet_count",
        "creatinine", "sodium", "potassium", "diagnosis_code", "icd_code",
        "procedure_code", "admission_date", "discharge_date", "length_of_stay",
        "readmission_flag", "mortality_risk", "treatment_cost", "insurance_type",
        "primary_diagnosis", "secondary_diagnosis", "medication", "dosage",
        "lab_result", "imaging_result", "surgery_type", "anesthesia_type",
        "recovery_time", "pain_score", "satisfaction_score", "follow_up_date",
        "emergency_flag", "icu_stay", "ventilator_days", "infection_flag",
        "allergy", "smoking_status", "alcohol_use", "exercise_frequency",
        "family_history", "chronic_condition", "vaccination_status"
    ],
    "finance": [
        "transaction_id", "transaction_amount", "transaction_date", "account_id",
        "account_balance", "credit_score", "credit_limit", "debt_to_income",
        "loan_amount", "interest_rate", "monthly_payment", "loan_term",
        "default_flag", "delinquency_days", "payment_history", "annual_income",
        "employment_years", "home_ownership", "mortgage_balance", "investment_value",
        "portfolio_return", "dividend_yield", "pe_ratio", "market_cap",
        "trading_volume", "bid_price", "ask_price", "spread",
        "volatility", "beta", "sharpe_ratio", "risk_score",
        "fraud_flag", "suspicious_activity", "kyc_status", "aml_flag",
        "wire_transfer_amount", "atm_withdrawal", "online_purchase",
        "recurring_payment", "overdraft_count", "nsf_count",
        "savings_rate", "retirement_contribution", "tax_bracket",
        "net_worth", "liquid_assets", "fixed_assets"
    ],
    "manufacturing": [
        "machine_id", "production_line", "batch_number", "units_produced",
        "defect_count", "defect_rate", "yield_rate", "cycle_time",
        "setup_time", "downtime_minutes", "maintenance_cost", "energy_consumption",
        "raw_material_cost", "labor_hours", "operator_id", "shift",
        "temperature_celsius", "humidity_percent", "vibration_level", "rpm",
        "torque_nm", "pressure_psi", "flow_rate", "ph_level",
        "viscosity", "hardness", "tensile_strength", "surface_roughness",
        "tolerance_deviation", "scrap_rate", "rework_count", "oee",
        "mtbf_hours", "mttr_hours", "availability_rate", "performance_rate",
        "quality_rate", "inventory_level", "lead_time_days", "order_backlog",
        "supplier_rating", "inspection_result", "calibration_date",
        "tool_wear_level", "coolant_temperature", "chip_load",
        "spindle_speed", "feed_rate"
    ],
    "retail": [
        "product_id", "product_name", "sku", "category", "subcategory",
        "brand", "price", "cost_price", "markup_percent", "discount_percent",
        "quantity_sold", "quantity_in_stock", "reorder_point", "reorder_quantity",
        "units_per_transaction", "basket_size", "average_order_value",
        "customer_id", "customer_segment", "loyalty_points", "lifetime_value",
        "purchase_frequency", "recency_days", "churn_risk",
        "store_id", "store_location", "store_size_sqft", "foot_traffic",
        "conversion_rate", "sales_per_sqft", "shrinkage_rate",
        "return_rate", "return_reason", "refund_amount",
        "promotion_id", "promotion_type", "promotion_lift",
        "channel", "online_flag", "delivery_time_days",
        "customer_satisfaction", "nps_score", "review_rating",
        "competitor_price", "price_elasticity", "margin_percent",
        "seasonal_index", "weather_impact"
    ],
    "energy": [
        "plant_id", "generation_mw", "capacity_mw", "capacity_factor",
        "fuel_type", "fuel_consumption", "fuel_cost", "heat_rate",
        "efficiency_percent", "emissions_co2", "emissions_nox", "emissions_sox",
        "electricity_price", "wholesale_price", "retail_price",
        "demand_mw", "peak_demand", "load_factor", "reserve_margin",
        "renewable_share", "solar_generation", "wind_generation",
        "hydro_generation", "nuclear_generation", "thermal_generation",
        "storage_capacity_mwh", "charge_rate", "discharge_rate",
        "grid_frequency", "voltage_level", "power_factor",
        "transmission_loss", "distribution_loss", "outage_duration",
        "outage_frequency", "saidi", "saifi", "caidi",
        "smart_meter_reading", "consumption_kwh", "billing_amount",
        "carbon_credit", "carbon_tax", "lcoe",
        "irradiance", "wind_speed", "ambient_temperature"
    ],
    "education": [
        "student_id", "enrollment_date", "graduation_date", "gpa",
        "credits_completed", "credits_attempted", "major", "minor",
        "course_id", "course_grade", "course_credits", "instructor_rating",
        "attendance_rate", "assignment_score", "exam_score", "quiz_score",
        "participation_score", "final_grade", "pass_fail",
        "tuition_amount", "financial_aid", "scholarship_amount", "loan_amount",
        "sat_score", "act_score", "gre_score", "gmat_score",
        "application_status", "admission_decision", "yield_rate",
        "retention_rate", "dropout_risk", "transfer_flag",
        "class_size", "student_teacher_ratio", "faculty_count",
        "research_output", "publication_count", "citation_count",
        "ranking_position", "endowment_value", "alumni_donation",
        "campus_size_acres", "dormitory_occupancy", "library_usage",
        "career_placement_rate", "starting_salary", "employer_satisfaction"
    ],
    "logistics": [
        "shipment_id", "tracking_number", "origin", "destination",
        "distance_km", "weight_kg", "volume_cbm", "pieces_count",
        "freight_cost", "freight_rate", "surcharge", "insurance_cost",
        "carrier_id", "carrier_name", "service_level", "mode_of_transport",
        "pickup_date", "delivery_date", "estimated_delivery", "transit_days",
        "actual_transit_days", "on_time_flag", "delay_hours", "delay_reason",
        "warehouse_id", "storage_days", "handling_cost", "pick_pack_time",
        "order_accuracy", "damage_flag", "claim_amount",
        "route_id", "stops_count", "fuel_consumption", "co2_emissions",
        "vehicle_id", "vehicle_type", "capacity_utilization",
        "customs_clearance_days", "duty_amount", "incoterm",
        "purchase_order", "invoice_number", "payment_terms",
        "inventory_turns", "fill_rate", "backorder_rate",
        "demand_forecast", "safety_stock", "reorder_point"
    ],
    "sports": [
        "match_id", "match_date", "home_team", "away_team",
        "home_score", "away_score", "attendance", "venue",
        "player_id", "player_name", "position", "age",
        "games_played", "minutes_played", "goals", "assists",
        "shots", "shots_on_target", "pass_accuracy", "tackles",
        "interceptions", "fouls_committed", "fouls_drawn", "yellow_cards",
        "red_cards", "saves", "clean_sheets", "goals_conceded",
        "expected_goals", "expected_assists", "possession_percent",
        "distance_covered_km", "sprint_count", "top_speed_kmh",
        "dribbles_completed", "crosses", "corners", "free_kicks",
        "penalties_scored", "penalties_missed", "offsides",
        "win_probability", "elo_rating", "transfer_value",
        "salary", "contract_years", "market_value",
        "injury_days", "fitness_score"
    ],
    "real_estate": [
        "property_id", "listing_date", "sale_date", "listing_price",
        "sale_price", "price_per_sqft", "property_type", "bedrooms",
        "bathrooms", "total_sqft", "lot_size_sqft", "year_built",
        "stories", "parking_spaces", "garage_flag", "pool_flag",
        "hoa_fee", "property_tax", "zoning_code", "neighborhood",
        "school_rating", "walk_score", "transit_score",
        "days_on_market", "price_reduction_count", "offers_received",
        "mortgage_rate", "down_payment_percent", "monthly_payment",
        "rental_price", "cap_rate", "gross_yield", "net_yield",
        "occupancy_rate", "vacancy_rate", "tenant_turnover",
        "maintenance_cost", "renovation_cost", "appreciation_rate",
        "comparable_sale_1", "comparable_sale_2", "comparable_sale_3",
        "appraisal_value", "assessed_value", "tax_assessment",
        "flood_zone", "crime_index", "noise_level"
    ],
    "information_technology": [
        "server_id", "cpu_usage_percent", "memory_usage_percent",
        "disk_usage_percent", "network_in_mbps", "network_out_mbps",
        "request_count", "response_time_ms", "error_rate",
        "uptime_percent", "latency_ms", "throughput_rps",
        "active_users", "concurrent_sessions", "page_views",
        "bounce_rate", "session_duration", "conversion_rate",
        "deployment_frequency", "lead_time_hours", "mttr_minutes",
        "change_failure_rate", "code_coverage_percent", "bug_count",
        "sprint_velocity", "story_points", "backlog_items",
        "ticket_id", "priority", "resolution_time_hours",
        "sla_compliance", "customer_satisfaction", "nps_score",
        "license_count", "license_cost", "cloud_spend",
        "storage_tb", "bandwidth_gb", "api_calls",
        "security_incidents", "vulnerability_count", "patch_compliance",
        "backup_success_rate", "recovery_time_hours"
    ],
    "marketing": [
        "campaign_id", "campaign_name", "channel", "platform",
        "budget", "spend", "impressions", "reach",
        "clicks", "click_through_rate", "cost_per_click",
        "conversions", "conversion_rate", "cost_per_conversion",
        "revenue_generated", "roas", "roi",
        "leads_generated", "lead_quality_score", "cost_per_lead",
        "email_sent", "email_opened", "open_rate", "click_rate",
        "unsubscribe_rate", "bounce_rate", "spam_rate",
        "social_followers", "social_engagement", "social_reach",
        "share_of_voice", "brand_awareness", "brand_sentiment",
        "customer_acquisition_cost", "customer_lifetime_value",
        "retention_rate", "churn_rate", "referral_rate",
        "ab_test_variant", "statistical_significance", "lift_percent",
        "audience_size", "frequency_cap", "viewability_rate",
        "video_completion_rate", "engagement_rate"
    ],
    "human_resources": [
        "employee_id", "hire_date", "termination_date", "tenure_years",
        "department", "job_title", "job_level", "manager_id",
        "salary", "bonus", "total_compensation", "pay_grade",
        "performance_rating", "performance_score", "goal_completion",
        "promotion_flag", "promotion_date", "lateral_move_count",
        "training_hours", "certification_count", "skill_score",
        "engagement_score", "satisfaction_score", "eNPS",
        "absence_days", "sick_days", "vacation_days_used",
        "overtime_hours", "work_life_balance_score",
        "attrition_risk", "flight_risk_score", "retention_flag",
        "recruitment_source", "time_to_fill_days", "cost_per_hire",
        "offer_acceptance_rate", "quality_of_hire_score",
        "diversity_flag", "remote_work_percent", "commute_distance",
        "age", "gender", "education_level",
        "team_size", "span_of_control", "succession_ready"
    ],
    "insurance": [
        "policy_id", "policy_type", "coverage_amount", "premium",
        "deductible", "copay", "coinsurance_percent",
        "policyholder_age", "policyholder_gender", "marital_status",
        "dependents_count", "occupation", "income_bracket",
        "claim_id", "claim_date", "claim_amount", "claim_status",
        "claim_type", "loss_date", "loss_cause", "loss_location",
        "adjuster_id", "settlement_amount", "settlement_days",
        "fraud_flag", "fraud_score", "investigation_flag",
        "underwriting_score", "risk_class", "risk_factor",
        "loss_ratio", "combined_ratio", "expense_ratio",
        "retention_rate", "lapse_rate", "renewal_rate",
        "agent_id", "agent_commission", "channel",
        "reinsurance_flag", "reserve_amount", "incurred_loss",
        "actuarial_estimate", "development_factor",
        "catastrophe_flag", "weather_related", "vehicle_age",
        "driving_record", "credit_score"
    ],
    "telecom": [
        "subscriber_id", "plan_type", "monthly_charge", "total_charges",
        "tenure_months", "contract_type", "payment_method",
        "data_usage_gb", "voice_minutes", "sms_count",
        "international_minutes", "roaming_charges",
        "download_speed_mbps", "upload_speed_mbps", "latency_ms",
        "signal_strength", "dropped_calls", "call_quality_score",
        "customer_service_calls", "complaint_count", "resolution_time",
        "churn_flag", "churn_reason", "churn_probability",
        "arpu", "mrr", "clv",
        "device_type", "device_age_months", "upgrade_flag",
        "bundle_flag", "add_on_services", "loyalty_points",
        "network_type", "cell_tower_id", "coverage_area",
        "bandwidth_allocation", "congestion_level",
        "ott_usage_hours", "streaming_quality",
        "sim_type", "esim_flag", "number_portability",
        "family_plan_flag", "lines_count",
        "nps_score", "satisfaction_score", "engagement_index"
    ],
    "entertainment": [
        "content_id", "title", "genre", "subgenre",
        "release_date", "runtime_minutes", "rating",
        "director", "producer", "studio",
        "budget_usd", "box_office_domestic", "box_office_international",
        "streaming_views", "unique_viewers", "completion_rate",
        "critic_score", "audience_score", "imdb_rating",
        "metacritic_score", "rotten_tomatoes",
        "subscriber_count", "monthly_active_users", "daily_active_users",
        "watch_time_hours", "sessions_per_user", "content_per_session",
        "recommendation_click_rate", "search_popularity",
        "social_mentions", "trending_score", "viral_coefficient",
        "ad_revenue", "subscription_revenue", "ppv_revenue",
        "production_cost", "marketing_spend", "roi",
        "awards_count", "nomination_count",
        "language", "country_of_origin", "age_rating",
        "sequel_flag", "franchise_id", "adaptation_source",
        "soundtrack_streams", "merchandise_revenue"
    ],
    "environmental": [
        "station_id", "measurement_date", "air_quality_index",
        "pm25_ugm3", "pm10_ugm3", "ozone_ppb", "no2_ppb",
        "so2_ppb", "co_ppm", "temperature_celsius", "humidity_percent",
        "wind_speed_ms", "wind_direction", "precipitation_mm",
        "uv_index", "visibility_km", "barometric_pressure",
        "water_quality_index", "dissolved_oxygen", "ph_level",
        "turbidity_ntu", "conductivity", "total_dissolved_solids",
        "nitrate_mgl", "phosphate_mgl", "lead_mgl", "mercury_mgl",
        "carbon_emissions_tons", "methane_emissions", "ghg_intensity",
        "deforestation_hectares", "biodiversity_index",
        "waste_generated_tons", "recycling_rate", "landfill_capacity",
        "energy_consumption_mwh", "renewable_percent",
        "sea_level_mm", "glacier_mass_change", "ocean_temperature",
        "coral_bleaching_percent", "species_count",
        "noise_level_db", "light_pollution_index",
        "soil_contamination", "groundwater_level"
    ],
    "automotive": [
        "vehicle_id", "make", "model", "year", "trim",
        "engine_type", "engine_displacement", "horsepower", "torque",
        "transmission", "drivetrain", "fuel_type", "mpg_city",
        "mpg_highway", "fuel_tank_gallons", "curb_weight_lbs",
        "wheelbase_inches", "length_inches", "width_inches",
        "msrp", "invoice_price", "dealer_cost", "incentive_amount",
        "days_to_sell", "inventory_age", "lot_location",
        "mileage", "condition_score", "accident_history",
        "service_records", "tire_condition", "brake_condition",
        "battery_health", "emission_test_result",
        "safety_rating", "crash_test_score", "airbag_count",
        "adas_features", "autonomous_level",
        "insurance_group", "depreciation_rate", "residual_value",
        "lease_payment", "finance_rate", "trade_in_value",
        "customer_rating", "reliability_score", "recall_count",
        "warranty_months", "extended_warranty_flag"
    ],
    "construction": [
        "project_id", "project_type", "project_phase", "start_date",
        "completion_date", "duration_days", "delay_days",
        "contract_value", "actual_cost", "cost_overrun_percent",
        "budget_variance", "earned_value", "planned_value",
        "labor_hours", "labor_cost", "material_cost", "equipment_cost",
        "subcontractor_cost", "overhead_cost", "profit_margin",
        "workers_count", "safety_incidents", "lost_time_injuries",
        "osha_violations", "near_miss_count",
        "concrete_volume_m3", "steel_weight_tons", "lumber_board_feet",
        "equipment_utilization", "crane_hours", "excavation_volume",
        "rfi_count", "change_order_count", "change_order_value",
        "inspection_pass_rate", "punch_list_items", "deficiency_count",
        "weather_delay_days", "permit_approval_days",
        "floor_area_sqft", "stories_count", "building_height",
        "energy_rating", "leed_score", "sustainability_index",
        "client_satisfaction", "warranty_claims"
    ],
    "pharmaceutical": [
        "drug_id", "drug_name", "therapeutic_area", "indication",
        "molecule_type", "mechanism_of_action", "route_of_administration",
        "dosage_form", "strength", "formulation",
        "phase", "trial_id", "enrollment_count", "randomization_ratio",
        "primary_endpoint", "secondary_endpoint", "p_value",
        "efficacy_rate", "response_rate", "remission_rate",
        "adverse_event_count", "serious_ae_count", "dropout_rate",
        "bioavailability", "half_life_hours", "clearance_rate",
        "volume_of_distribution", "protein_binding_percent",
        "manufacturing_cost", "api_cost", "packaging_cost",
        "wholesale_price", "retail_price", "reimbursement_rate",
        "market_share", "prescription_volume", "refill_rate",
        "patent_expiry_date", "exclusivity_months", "generic_competition",
        "r_and_d_spend", "time_to_market_months", "approval_probability",
        "pharmacovigilance_signals", "label_update_count",
        "recall_flag", "supply_chain_risk"
    ],
    "hospitality": [
        "property_id", "property_type", "star_rating", "room_count",
        "occupancy_rate", "adr", "revpar", "goppar",
        "booking_id", "check_in_date", "check_out_date", "length_of_stay",
        "room_type", "room_rate", "total_revenue", "ancillary_revenue",
        "food_beverage_revenue", "spa_revenue", "event_revenue",
        "guest_id", "guest_nationality", "loyalty_tier", "repeat_guest_flag",
        "booking_channel", "booking_lead_time", "cancellation_rate",
        "no_show_rate", "overbooking_rate",
        "review_score", "cleanliness_score", "service_score",
        "location_score", "value_score", "nps",
        "staff_count", "staff_to_room_ratio", "labor_cost_percent",
        "energy_cost", "maintenance_cost", "amenity_cost",
        "competitive_set_index", "market_penetration_index",
        "seasonal_demand_index", "group_business_percent",
        "corporate_rate_percent", "walk_in_percent"
    ],
    "government": [
        "agency_id", "department", "program_id", "fiscal_year",
        "budget_allocated", "budget_spent", "budget_variance",
        "revenue_collected", "tax_revenue", "fee_revenue",
        "expenditure_type", "personnel_cost", "operating_cost",
        "capital_expenditure", "grant_amount", "transfer_payment",
        "population_served", "service_requests", "response_time_days",
        "satisfaction_score", "complaint_count", "resolution_rate",
        "employee_count", "vacancy_rate", "turnover_rate",
        "procurement_value", "contract_count", "vendor_count",
        "audit_findings", "compliance_score", "risk_rating",
        "permit_applications", "permit_approved", "processing_days",
        "inspection_count", "violation_count", "enforcement_actions",
        "crime_rate", "clearance_rate", "recidivism_rate",
        "infrastructure_condition", "road_quality_index",
        "public_transit_ridership", "park_utilization",
        "voter_turnout", "public_meeting_attendance"
    ],
    "cybersecurity_sector": [
        "alert_id", "alert_severity", "alert_type", "timestamp",
        "source_ip", "destination_ip", "source_port", "destination_port",
        "protocol", "packet_size", "bytes_transferred",
        "session_duration", "connection_count", "failed_login_count",
        "malware_detected", "threat_category", "attack_vector",
        "vulnerability_id", "cvss_score", "exploit_available",
        "patch_status", "remediation_time_hours",
        "firewall_rule_hit", "ids_signature", "anomaly_score",
        "user_id", "privilege_level", "access_type",
        "data_classification", "encryption_status",
        "incident_id", "incident_severity", "impact_score",
        "containment_time", "eradication_time", "recovery_time",
        "false_positive_rate", "detection_rate", "response_time",
        "compliance_framework", "audit_score", "risk_score",
        "asset_criticality", "exposure_score",
        "phishing_attempts", "spam_rate", "dlp_violations"
    ],
    "weather_climate": [
        "station_id", "observation_date", "temperature_max",
        "temperature_min", "temperature_avg", "feels_like",
        "humidity_percent", "dew_point", "pressure_hpa",
        "wind_speed_kmh", "wind_gust_kmh", "wind_direction_deg",
        "precipitation_mm", "snowfall_cm", "snow_depth_cm",
        "visibility_km", "cloud_cover_percent", "uv_index",
        "sunshine_hours", "solar_radiation_wm2",
        "sea_surface_temperature", "wave_height_m", "tide_level",
        "air_quality_index", "pollen_count",
        "drought_index", "flood_risk_score", "fire_weather_index",
        "heating_degree_days", "cooling_degree_days",
        "growing_degree_days", "frost_flag", "ice_flag",
        "thunderstorm_flag", "tornado_risk", "hurricane_category",
        "historical_avg_temp", "anomaly_from_normal",
        "forecast_accuracy", "model_ensemble_spread",
        "climate_zone", "elevation_m", "land_use_type",
        "albedo", "evapotranspiration"
    ],
    "science": [
        "sample_id", "experiment_id", "measurement_date",
        "instrument_id", "calibration_date", "measurement_value",
        "measurement_unit", "uncertainty", "precision",
        "temperature_k", "pressure_atm", "ph",
        "concentration_mol", "absorbance", "wavelength_nm",
        "intensity", "fluorescence", "mass_dalton",
        "retention_time", "peak_area", "signal_to_noise",
        "sequence_length", "gc_content", "mutation_count",
        "expression_level", "fold_change", "p_value",
        "species", "genus", "family", "classification",
        "latitude", "longitude", "altitude_m", "depth_m",
        "habitat_type", "population_count", "density",
        "growth_rate", "survival_rate", "reproduction_rate",
        "biodiversity_index", "evenness", "richness",
        "correlation_coefficient", "r_squared", "chi_squared",
        "effect_size", "confidence_interval"
    ],
}

# Fill remaining sectors with adapted columns
DEFAULT_COLUMNS = GENERIC_COLUMNS + [
    "value_1", "value_2", "value_3", "metric_a", "metric_b",
    "indicator", "index_value", "benchmark", "target_value",
    "actual_value", "variance", "trend", "growth_rate",
    "year_over_year", "month_over_month", "cumulative",
    "normalized_score", "weighted_average", "percentile_rank"
]

# Map remaining main sectors to closest column pool
SECTOR_MAPPING = {
    "oil_gas": "energy",
    "mining": "manufacturing",
    "utilities": "energy",
    "wholesale_trade": "retail",
    "transportation": "logistics",
    "professional_services": "human_resources",
    "media": "entertainment",
}

def get_columns_for_sector(sector):
    if sector in SECTOR_COLUMN_POOLS:
        return SECTOR_COLUMN_POOLS[sector]
    mapped = SECTOR_MAPPING.get(sector)
    if mapped and mapped in SECTOR_COLUMN_POOLS:
        base = SECTOR_COLUMN_POOLS[mapped]
        # Add sector-specific prefix to some columns
        modified = [f"{sector.split('_')[0]}_{c}" if random.random() < 0.3 else c for c in base]
        return modified
    return DEFAULT_COLUMNS

# ============================================================
# VALUE GENERATORS
# ============================================================
def gen_normal(n, mean=50, std=15):
    return np.random.normal(mean, std, n)

def gen_skewed(n, a=5):
    return np.random.gamma(a, 10, n)

def gen_bimodal(n, m1=30, m2=70, s=8):
    half = n // 2
    return np.concatenate([np.random.normal(m1, s, half), np.random.normal(m2, s, n - half)])

def gen_uniform(n, low=0, high=100):
    return np.random.uniform(low, high, n)

def gen_powerlaw(n, a=2):
    return (np.random.pareto(a, n) + 1) * 10

def gen_integer(n, low=0, high=1000):
    return np.random.randint(low, high, n).astype(float)

def gen_binary(n, p=0.5):
    return np.random.binomial(1, p, n).astype(float)

def gen_categorical_numeric(n, categories=5):
    return np.random.randint(0, categories, n).astype(float)

GENERATORS = [gen_normal, gen_skewed, gen_bimodal, gen_uniform, gen_powerlaw, gen_integer]

# ============================================================
# TARGET GENERATORS
# ============================================================
def make_target_binary(X, n):
    # Use 2-3 features with threshold
    key_cols = random.sample(range(X.shape[1]), min(3, X.shape[1]))
    score = sum(X[:, c] for c in key_cols)
    threshold = np.percentile(score, random.randint(30, 70))
    return (score > threshold).astype(int)

def make_target_multiclass(X, n, n_classes):
    key_cols = random.sample(range(X.shape[1]), min(4, X.shape[1]))
    score = sum(X[:, c] for c in key_cols)
    percentiles = np.linspace(0, 100, n_classes + 1)[1:-1]
    thresholds = np.percentile(score, percentiles)
    target = np.digitize(score, thresholds)
    return target

def make_target_interaction(X, n, n_classes=3):
    if X.shape[1] < 2:
        return make_target_multiclass(X, n, n_classes)
    c1, c2 = random.sample(range(X.shape[1]), 2)
    interaction = X[:, c1] * X[:, c2]
    percentiles = np.linspace(0, 100, n_classes + 1)[1:-1]
    thresholds = np.percentile(interaction, percentiles)
    return np.digitize(interaction, thresholds)

# ============================================================
# DATASET GENERATOR
# ============================================================
def generate_dataset(sector, dataset_idx):
    columns_pool = get_columns_for_sector(sector)
    
    # Random number of columns (5-30)
    n_cols = random.randint(5, min(30, len(columns_pool)))
    columns = random.sample(columns_pool, n_cols)
    
    # Random number of rows (100-2000)
    n_rows = random.choice([100, 200, 300, 500, 750, 1000, 1500, 2000])
    
    # Generate feature matrix
    X = np.zeros((n_rows, n_cols))
    for c in range(n_cols):
        gen = random.choice(GENERATORS)
        params = {}
        if gen == gen_normal:
            params = {"mean": random.uniform(10, 1000), "std": random.uniform(1, 200)}
        elif gen == gen_uniform:
            low = random.uniform(0, 500)
            params = {"low": low, "high": low + random.uniform(10, 5000)}
        elif gen == gen_integer:
            params = {"low": 0, "high": random.randint(10, 10000)}
        X[:, c] = gen(n_rows, **params)
    
    # Add correlations between some features
    if n_cols >= 4:
        n_corr = random.randint(1, min(3, n_cols // 2))
        for _ in range(n_corr):
            src, dst = random.sample(range(n_cols), 2)
            noise = np.random.normal(0, np.std(X[:, src]) * random.uniform(0.1, 0.5), n_rows)
            X[:, dst] = X[:, src] * random.uniform(0.5, 1.5) + noise
    
    # Target
    n_classes = random.choice([2, 2, 2, 3, 3, 4, 5, 7, 10])
    if n_classes == 2:
        target = make_target_binary(X, n_rows)
    elif random.random() < 0.3:
        target = make_target_interaction(X, n_rows, n_classes)
    else:
        target = make_target_multiclass(X, n_rows, n_classes)
    
    # Imbalance
    balance_type = random.choices(
        ["balanced", "mild_imbalance", "heavy_imbalance"],
        weights=[0.6, 0.25, 0.15]
    )[0]
    
    # Missing data
    missing_ratio = random.choices(
        [0.0, random.uniform(0.05, 0.15), random.uniform(0.15, 0.30), random.uniform(0.30, 0.50)],
        weights=[0.3, 0.3, 0.2, 0.2]
    )[0]
    
    if missing_ratio > 0:
        mask = np.random.random(X.shape) < missing_ratio
        X[mask] = np.nan
    
    # Outliers
    if random.random() < 0.3:
        n_outliers = int(n_rows * random.uniform(0.02, 0.08))
        for _ in range(n_outliers):
            r = random.randint(0, n_rows - 1)
            c = random.randint(0, n_cols - 1)
            X[r, c] = X[r, c] * random.uniform(5, 20) if not np.isnan(X[r, c]) else X[r, c]
    
    # Sample rows for metadata (10 rows)
    sample_indices = random.sample(range(n_rows), min(10, n_rows))
    sample_rows = []
    for idx in sample_indices:
        row = []
        for c in range(n_cols):
            v = X[idx, c]
            if np.isnan(v):
                row.append("")
            elif columns[c].endswith("_flag") or columns[c].startswith("is_"):
                row.append(str(int(v)))
            else:
                row.append(str(round(float(v), 2)))
        row.append(str(int(target[idx])))
        sample_rows.append(row)
    
    columns_with_target = columns + ["target"]
    
    return {
        "columns": columns_with_target,
        "sample_rows": sample_rows,
        "sector": sector,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_classes": int(len(set(target))),
        "missing_ratio": round(missing_ratio, 3),
        "balance": balance_type,
        "folder": f"synthetic_{sector}_{dataset_idx}",
    }

# ============================================================
# MAIN GENERATION
# ============================================================
print("=" * 60)
print("GENERATING 500K DATASETS")
print(f"10000 sectors x 50 datasets each")
print("=" * 60)

all_datasets = []
start = time.time()

for sector_idx, sector in enumerate(ALL_SECTORS):
    # Find main sector for column pool
    main = None
    for m, subs in HIERARCHY.items():
        if sector == m or sector in subs:
            main = m
            break
    if not main:
        main = sector
    
    for di in range(50):
        ds = generate_dataset(main, sector_idx * 50 + di)
        ds["sector"] = sector  # Use exact 10000-sector label
        ds["main_sector"] = main
        all_datasets.append(ds)
    
    if (sector_idx + 1) % 100 == 0:
        elapsed = time.time() - start
        rate = (sector_idx + 1) / elapsed
        eta = (len(ALL_SECTORS) - sector_idx - 1) / rate
        print(f"  [{sector_idx+1:5d}/{len(ALL_SECTORS)}] {sector[:30]:30s} "
              f"datasets={len(all_datasets):,} elapsed={elapsed:.0f}s eta={eta:.0f}s")

elapsed = time.time() - start
print(f"\nGenerated {len(all_datasets):,} datasets in {elapsed:.0f}s")

# Stats
sectors_count = Counter(ds["main_sector"] for ds in all_datasets)
print(f"\nMain sector distribution:")
for s, c in sorted(sectors_count.items(), key=lambda x: -x[1])[:20]:
    print(f"  {s:25s}: {c:,}")

missing_counts = Counter(
    "clean" if ds["missing_ratio"] == 0 else
    "light" if ds["missing_ratio"] < 0.15 else
    "medium" if ds["missing_ratio"] < 0.30 else "heavy"
    for ds in all_datasets
)
print(f"\nMissing data distribution: {dict(missing_counts)}")

class_counts = Counter(ds["n_classes"] for ds in all_datasets)
print(f"Class distribution: {dict(sorted(class_counts.items()))}")

balance_counts = Counter(ds["balance"] for ds in all_datasets)
print(f"Balance distribution: {dict(balance_counts)}")

# Save
print(f"\nSaving to {OUTPUT}...")
with open(OUTPUT, "w") as f:
    json.dump(all_datasets, f)

size_mb = OUTPUT.stat().st_size / 1024 / 1024
print(f"Saved: {OUTPUT} ({size_mb:.1f} MB)")
print("DONE")

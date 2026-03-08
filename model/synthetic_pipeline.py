"""
Pure Python synthetic sector data — no API needed
Her sektör için template'lerden varyasyonlar üretir
"""
import json, random, numpy as np
from pathlib import Path
from collections import Counter

random.seed(42)
SAVE_FILE = Path("data/sector_labels.json")

TEMPLATES = {
"healthcare": [
    ["patient_id","age","gender","bmi","blood_pressure","diagnosis","medication","discharge_date","los","readmission"],
    ["patient_id","glucose","insulin","bmi","age","skin_thickness","pregnancies","outcome"],
    ["hospital_id","department","num_beds","occupancy_rate","avg_los","mortality_rate","readmission_rate"],
    ["drug_name","dosage","trial_phase","efficacy","side_effects","approval_status","manufacturer"],
    ["sample_id","hemoglobin","wbc","rbc","platelets","mcv","mch","mchc","result"],
    ["patient_id","systolic_bp","diastolic_bp","heart_rate","temperature","spo2","timestamp"],
    ["claim_id","procedure_code","diagnosis_code","claim_amount","approved_amount","denial_reason"],
    ["symptom","icd_code","prevalence","avg_age_onset","gender_ratio","treatment_cost"],
    ["surgery_type","duration_min","complications","surgeon_id","anesthesia_type","outcome"],
    ["vaccine","coverage_rate","efficacy_rate","doses_required","adverse_events","population"],
],
"finance": [
    ["loan_id","loan_amount","interest_rate","term","credit_score","annual_income","dti","default"],
    ["transaction_id","amount","merchant","category","timestamp","fraud_label","card_type"],
    ["ticker","open","high","low","close","volume","market_cap","pe_ratio","dividend_yield"],
    ["customer_id","account_balance","num_transactions","credit_limit","payment_history","churn"],
    ["portfolio_id","asset_class","weight","return_1y","volatility","sharpe_ratio","beta"],
    ["company","revenue","ebitda","net_income","total_assets","total_debt","roe","eps"],
    ["currency_pair","bid","ask","spread","volume","timestamp","exchange"],
    ["claim_amount","premium","policy_type","deductible","loss_ratio","reserve"],
    ["bond_id","issuer","coupon_rate","maturity_date","yield","rating","duration"],
    ["branch_id","deposits","withdrawals","new_accounts","loan_disbursements","npls"],
],
"sports": [
    ["player_id","name","team","position","goals","assists","matches","minutes","yellow_cards","red_cards"],
    ["match_id","home_team","away_team","home_score","away_score","possession","shots","date"],
    ["athlete_id","event","time_seconds","distance_m","wind_speed","rank","nationality","year"],
    ["player","season","pts","reb","ast","stl","blk","fg_pct","three_pct","ft_pct"],
    ["team","wins","losses","draws","points","goals_for","goals_against","xg","xga"],
    ["race_id","driver","constructor","grid_pos","finish_pos","fastest_lap","points","pit_stops"],
    ["player","batting_avg","home_runs","rbi","obp","slg","ops","strikeouts","walks"],
    ["set_number","player1","player2","score","aces","double_faults","winners","errors"],
    ["player","weight_class","wins","losses","ko_wins","submission_wins","decision_wins","reach"],
    ["horse","jockey","trainer","odds","finish_position","time","distance","track_condition"],
],
"manufacturing": [
    ["machine_id","temperature","vibration","pressure","rotation_speed","torque","tool_wear","failure"],
    ["product_id","defect_type","defect_rate","line_id","shift","operator_id","timestamp"],
    ["batch_id","raw_material","quantity","quality_score","yield_rate","waste_pct","cost"],
    ["sensor_id","timestamp","value","unit","threshold","alert","equipment_id"],
    ["order_id","product","quantity","lead_time","on_time","quality_pass","customer_id"],
    ["component","supplier","unit_cost","lead_time_days","defect_rate","stock_level","reorder_point"],
    ["oee_date","availability","performance","quality","oee_score","downtime_min","line_id"],
    ["inspection_id","product_id","inspector","result","defect_code","rework_required","timestamp"],
    ["energy_kwh","machine_id","shift","production_units","efficiency","timestamp"],
    ["maintenance_id","machine_id","type","duration_hr","cost","technician","next_due"],
],
"ecommerce": [
    ["order_id","customer_id","product_id","quantity","price","discount","shipping_cost","status"],
    ["product_id","name","category","price","rating","reviews","stock","seller_id"],
    ["customer_id","age","gender","country","total_orders","total_spent","last_purchase","churn"],
    ["session_id","user_id","pages_viewed","time_on_site","cart_added","purchase","source"],
    ["review_id","product_id","rating","sentiment","verified","helpful_votes","timestamp"],
    ["ad_id","platform","impressions","clicks","ctr","cpc","conversions","roas","spend"],
    ["return_id","order_id","reason","condition","refund_amount","resolution","timestamp"],
    ["sku","warehouse","stock_level","reorder_point","sales_velocity","days_of_supply"],
    ["coupon_code","discount_pct","usage_count","revenue_impact","expiry_date","category"],
    ["delivery_id","carrier","estimated_days","actual_days","on_time","tracking_events","cost"],
],
"education": [
    ["student_id","grade","math_score","reading_score","writing_score","lunch","test_prep","gender"],
    ["course_id","enrollment","completion_rate","avg_grade","drop_rate","instructor_rating"],
    ["student_id","attendance_rate","gpa","extracurricular","parent_education","income","outcome"],
    ["school_id","district","students","teachers","student_teacher_ratio","graduation_rate","budget"],
    ["question_id","difficulty","discrimination","correct_rate","topic","bloom_level"],
    ["employee_id","training_hours","pre_score","post_score","department","role","year"],
    ["university","rank","acceptance_rate","tuition","graduation_rate","research_output","endowment"],
    ["assignment_id","student_id","score","submission_time","late","plagiarism_score","feedback"],
    ["program","graduates","employed_6mo","median_salary","satisfaction","accreditation"],
    ["book_id","subject","grade_level","reading_level","adoption_rate","publisher","year"],
],
"energy": [
    ["timestamp","solar_irradiance","temperature","dc_power","ac_power","efficiency","plant_id"],
    ["turbine_id","wind_speed","rotor_speed","power_output","pitch_angle","temperature","status"],
    ["meter_id","consumption_kwh","timestamp","tariff","peak_demand","reactive_power","customer_type"],
    ["station_id","fuel_type","capacity_mw","generation_mwh","availability","heat_rate","emissions"],
    ["grid_id","load_mw","frequency","voltage","losses","timestamp","region"],
    ["ev_station_id","charging_sessions","avg_duration","energy_delivered","utilization","revenue"],
    ["building_id","hvac_kwh","lighting_kwh","equipment_kwh","total_kwh","area_sqm","benchmark"],
    ["oil_well_id","production_bbl","pressure","temperature","water_cut","gor","status"],
    ["pipeline_id","flow_rate","pressure_in","pressure_out","temperature","leak_detected","segment"],
    ["household_id","monthly_kwh","solar_installed","net_metering","income","region","tariff_plan"],
],
"transportation": [
    ["flight_id","origin","destination","distance","dep_delay","arr_delay","carrier","cancelled"],
    ["vehicle_id","timestamp","latitude","longitude","speed","heading","fuel_level","status"],
    ["route_id","stops","avg_delay","ridership","on_time_pct","incidents","operator"],
    ["shipment_id","origin","destination","weight_kg","volume_cbm","carrier","transit_days","cost"],
    ["accident_id","severity","road_type","weather","light_conditions","casualties","timestamp"],
    ["port_id","vessel","cargo_type","teu","berth_time","turnaround","origin","destination"],
    ["trip_id","driver_id","distance_km","duration_min","fare","surge_multiplier","rating"],
    ["train_id","route","departure","arrival","delay_min","passengers","punctuality","speed"],
    ["intersection_id","hour","volume","avg_speed","incidents","signal_timing","congestion_level"],
    ["toll_id","vehicle_class","timestamp","amount","payment_type","lane","direction"],
],
"agriculture": [
    ["field_id","crop","yield_ton","area_ha","rainfall","temperature","fertilizer_kg","pesticide_kg"],
    ["sample_id","nitrogen","phosphorus","potassium","ph","organic_matter","moisture","texture"],
    ["livestock_id","species","breed","weight_kg","age_days","feed_kg","health_status","location"],
    ["sensor_id","timestamp","soil_moisture","temperature","humidity","ndvi","field_id"],
    ["market_id","commodity","price","volume","date","region","quality_grade","origin"],
    ["farm_id","area_ha","irrigation","mechanization","credit_access","yield_index","income"],
    ["pest_id","crop","infestation_pct","treatment","effectiveness","cost_per_ha","timestamp"],
    ["weather_station","date","max_temp","min_temp","precipitation","wind_speed","evapotranspiration"],
    ["harvest_id","crop","quantity_ton","moisture_pct","grade","storage_loss","sale_price"],
    ["variety","crop","maturity_days","drought_tolerance","disease_resistance","avg_yield","region"],
],
"realestate": [
    ["property_id","price","sqft","bedrooms","bathrooms","garage","year_built","zip","school_rating"],
    ["listing_id","price","days_on_market","price_reduction","list_price","sale_price","agent_id"],
    ["property_id","rent","vacancy_rate","cap_rate","noi","expenses","property_type","location"],
    ["project_id","units","construction_cost","completion_date","presales_pct","irr","developer"],
    ["address","assessed_value","tax_rate","tax_amount","exemptions","appeal_status","year"],
    ["neighborhood","median_price","price_growth","crime_rate","walkability","transit_score","schools"],
    ["mortgage_id","principal","rate","term","ltv","monthly_payment","default_risk","lender"],
    ["building_id","floors","units","occupancy","common_charges","reserve_fund","amenities","age"],
    ["lease_id","tenant","start_date","end_date","monthly_rent","deposit","renewal","escalation"],
    ["investor_id","portfolio_value","num_properties","avg_cap_rate","leverage","cash_flow","irr"],
],
"hr": [
    ["employee_id","age","gender","department","salary","tenure","performance","attrition"],
    ["job_id","title","level","salary_band","openings","applications","hires","time_to_fill"],
    ["employee_id","training_hours","courses_completed","certification","performance_delta","cost"],
    ["survey_id","employee_id","engagement_score","satisfaction","manager_rating","eNPS"],
    ["applicant_id","source","applied_date","stage","hired","offer_amount","rejection_reason"],
    ["employee_id","absence_days","reason","department","rolling_12m","bradford_score"],
    ["review_id","employee_id","reviewer","goals_met","competency_score","rating","year"],
    ["employee_id","base_salary","bonus","equity","benefits_cost","total_comp","band","market_ratio"],
    ["incident_id","type","severity","department","resolution_days","repeat_offender","outcome"],
    ["employee_id","diversity_category","level","pay_gap","promotion_rate","tenure","location"],
],
"marketing": [
    ["campaign_id","channel","impressions","clicks","conversions","cost","revenue","roas","date"],
    ["customer_id","rfm_recency","rfm_frequency","rfm_monetary","segment","ltv","churn_prob"],
    ["email_id","subject","open_rate","click_rate","unsubscribe_rate","revenue","list_size","date"],
    ["keyword","impressions","clicks","ctr","cpc","quality_score","position","conversions"],
    ["influencer_id","followers","engagement_rate","reach","impressions","cost","conversions"],
    ["content_id","type","views","shares","comments","watch_time","ctr","revenue"],
    ["ab_test_id","variant","visitors","conversions","revenue","significance","uplift"],
    ["brand","awareness_pct","consideration_pct","preference_pct","nps","market_share","date"],
    ["store_id","footfall","conversion_rate","avg_basket","promo_lift","category","date"],
    ["lead_id","source","score","stage","days_to_close","deal_value","rep_id","outcome"],
],
"cybersecurity": [
    ["log_id","src_ip","dst_ip","protocol","src_port","dst_port","bytes","packets","label"],
    ["alert_id","severity","type","source","destination","timestamp","status","analyst"],
    ["scan_id","host","vulnerability","cvss_score","exploitable","patch_available","age_days"],
    ["user_id","login_time","ip","location","device","failed_attempts","mfa_used","anomaly"],
    ["malware_id","family","behavior","infection_vector","targets","detection_rate","timestamp"],
    ["asset_id","type","os","patch_level","criticality","exposure_score","last_scan"],
    ["incident_id","type","severity","detection_time","response_time","impact","cost","resolved"],
    ["email_id","sender","subject","attachments","links","spam_score","phishing_label","timestamp"],
    ["firewall_id","rule","action","src","dst","hits","blocked","timestamp"],
    ["endpoint_id","os","antivirus","last_update","threats_detected","encrypted","compliant"],
],
"government": [
    ["region_id","population","gdp_per_capita","unemployment","poverty_rate","literacy","hdi"],
    ["project_id","ministry","budget","spent","completion_pct","beneficiaries","outcome"],
    ["citizen_id","age","gender","region","income","tax_paid","benefits_received","employment"],
    ["election_id","region","candidate","party","votes","voter_turnout","margin","year"],
    ["case_id","court","type","filing_date","resolution_date","outcome","sentence","appeal"],
    ["permit_id","type","applicant","submitted","approved","denied","processing_days","fee"],
    ["school_id","region","students","teachers","budget","pass_rate","dropout_rate","facilities"],
    ["road_id","type","length_km","condition","last_maintenance","accidents","traffic_volume"],
    ["crime_id","type","region","timestamp","resolved","suspect_age","victim_age","weapon"],
    ["hospital_id","region","beds","doctors","nurses","budget","patients","mortality_rate"],
],
"insurance": [
    ["policy_id","type","premium","coverage","deductible","claims_count","claim_amount","renewal"],
    ["claim_id","policy_id","incident_date","claim_amount","approved","fraud_flag","settlement"],
    ["customer_id","age","gender","bmi","smoker","region","premium","plan_type","claims"],
    ["vehicle_id","make","model","year","value","driver_age","accidents","premium","claim_freq"],
    ["property_id","type","value","location","flood_zone","fire_risk","premium","claims"],
    ["life_policy_id","age","health_score","sum_assured","premium","beneficiary","term","status"],
    ["risk_id","category","probability","severity","exposure","mitigation","residual_risk"],
    ["agent_id","policies_sold","premium_volume","retention_rate","claims_ratio","commission"],
    ["reinsurance_id","cedant","treaty_type","premium","limit","retention","loss_ratio"],
    ["actuary_model","line","loss_ratio","expense_ratio","combined_ratio","reserve","rdp"],
],
"supplychain": [
    ["po_id","supplier","item","quantity","unit_cost","order_date","delivery_date","on_time","quality"],
    ["warehouse_id","sku","stock_level","reorder_point","lead_time","holding_cost","stockout_freq"],
    ["shipment_id","origin","destination","carrier","weight","cost","transit_days","damage_rate"],
    ["supplier_id","reliability","quality_score","lead_time","price_index","capacity","risk_score"],
    ["demand_id","sku","date","actual","forecast","error","bias","region"],
    ["production_id","sku","planned","actual","yield","scrap","bottleneck","shift"],
    ["return_id","sku","reason","quantity","condition","disposition","cost","supplier"],
    ["bom_id","parent_sku","component","quantity","unit_cost","lead_time","criticality"],
    ["logistics_id","mode","origin","destination","cost_per_kg","transit_days","co2_kg","carrier"],
    ["inventory_id","sku","location","quantity","value","turnover","days_supply","abc_class"],
],
"telecom": [
    ["customer_id","plan","monthly_charge","tenure","data_usage_gb","calls_min","sms_count","churn"],
    ["tower_id","latitude","longitude","technology","capacity","utilization","downtime","region"],
    ["call_id","caller","receiver","duration_sec","type","quality_score","timestamp","dropped"],
    ["ticket_id","customer_id","issue_type","priority","created","resolved","satisfaction","channel"],
    ["device_id","model","os","data_usage","signal_strength","battery","customer_id","timestamp"],
    ["network_id","segment","latency_ms","packet_loss","throughput_mbps","timestamp","alert"],
    ["invoice_id","customer_id","amount","due_date","paid_date","late","payment_method","plan"],
    ["roaming_id","customer_id","country","data_mb","calls_min","charges","timestamp"],
    ["spectrum_id","band","frequency","license","operator","coverage_pct","throughput","region"],
    ["fiber_id","node","homes_passed","connected","speed_tier","arpu","churn_rate","region"],
],
"entertainment": [
    ["movie_id","title","genre","budget","revenue","rating","runtime","director","year","votes"],
    ["track_id","title","artist","album","danceability","energy","valence","tempo","streams"],
    ["game_id","title","genre","platform","sales","metacritic","user_score","developer","year"],
    ["show_id","title","platform","seasons","episodes","rating","views","genre","year"],
    ["user_id","content_id","watch_time","rating","completed","device","timestamp","genre"],
    ["book_id","title","author","genre","sales","rating","reviews","pages","year","publisher"],
    ["event_id","type","venue","attendance","revenue","ticket_price","artist","city","date"],
    ["channel_id","subscribers","views","uploads","avg_duration","engagement_rate","monetized"],
    ["podcast_id","title","category","episodes","avg_duration","downloads","rating","host"],
    ["ad_id","platform","content_type","impressions","completion_rate","ctr","skipped","revenue"],
],
"climate": [
    ["station_id","date","max_temp","min_temp","precipitation","humidity","wind_speed","sunshine_hrs"],
    ["country","year","co2_emissions","gdp","population","energy_intensity","renewable_pct"],
    ["glacier_id","name","area_km2","volume_km3","retreat_rate","elevation","year"],
    ["ocean_id","latitude","longitude","sst","salinity","ph","oxygen","depth","timestamp"],
    ["fire_id","region","area_ha","duration_days","cause","temperature","humidity","wind"],
    ["city","year","urban_heat_island","green_cover_pct","albedo","pm25","population"],
    ["flood_id","region","area_km2","depth_m","duration_days","damage_usd","displaced"],
    ["crop","region","year","yield_baseline","yield_change_pct","temperature_anomaly","precipitation"],
    ["species_id","name","habitat","population_trend","temperature_sensitivity","range_shift_km"],
    ["policy_id","country","type","target","baseline_emissions","projected_reduction","cost_bn"],
],
}

def make_stats(col_name, dtype="auto"):
    """Kolon adına göre gerçekçi stats üret"""
    col = col_name.lower()
    
    # numeric mi?
    numeric_hints = ["age","score","rate","ratio","count","amount","price","cost","value",
                     "pct","percent","num","qty","quantity","size","weight","height","bmi",
                     "temp","speed","power","kwh","revenue","salary","income","gdp","km",
                     "hr","min","sec","days","hours","delay","lat","lon","duration"]
    is_numeric = any(h in col for h in numeric_hints)
    
    if dtype == "auto":
        is_numeric = any(h in col for h in numeric_hints)
    
    null_ratio   = round(random.uniform(0, 0.15), 3)
    unique_ratio = round(random.uniform(0.3, 1.0), 3)
    
    if is_numeric:
        # makul aralıklar
        if any(x in col for x in ["age","year"]):
            mean, std = random.uniform(25,55), random.uniform(5,20)
        elif any(x in col for x in ["rate","ratio","pct","percent"]):
            mean, std = random.uniform(0.1,0.9), random.uniform(0.05,0.3)
        elif any(x in col for x in ["price","cost","amount","revenue","salary","income","value"]):
            mean, std = random.uniform(1000,100000), random.uniform(500,50000)
        elif any(x in col for x in ["score","rating"]):
            mean, std = random.uniform(3,8), random.uniform(0.5,2)
        elif any(x in col for x in ["temp","temperature"]):
            mean, std = random.uniform(15,35), random.uniform(3,10)
        else:
            mean, std = random.uniform(10,1000), random.uniform(5,500)
        
        mn = max(0, mean - 2*std)
        mx = mean + 3*std
        return {
            "dtype": "float64",
            "null_ratio": null_ratio,
            "unique_ratio": unique_ratio,
            "mean": round(mean, 2),
            "std":  round(abs(std), 2),
            "min":  round(mn, 2),
            "max":  round(mx, 2),
        }
    else:
        return {
            "dtype": "object",
            "null_ratio": null_ratio,
            "unique_ratio": round(random.uniform(0.001, 0.5), 3),
        }

def augment_columns(cols):
    """Kolon adlarına küçük varyasyonlar uygula"""
    synonyms = {
        "id": ["_id","_key","_code","_no","_num"],
        "amount": ["_amount","_sum","_total","_value"],
        "date": ["_date","_time","_timestamp","_at"],
        "rate": ["_rate","_ratio","_pct","_percent"],
        "score": ["_score","_rating","_index","_grade"],
    }
    result = []
    for col in cols:
        if random.random() < 0.3:  # %30 ihtimalle varyasyon
            for key, variants in synonyms.items():
                if col.endswith(key) and random.random() < 0.5:
                    base = col[:-len(key)]
                    col  = base + random.choice(variants)
                    break
        result.append(col)
    return result

def main():
    existing = json.loads(SAVE_FILE.read_text()) if SAVE_FILE.exists() else {}
    target   = 100

    for sector, templates in TEMPLATES.items():
        existing_count = sum(1 for v in existing.values() if v["sector"] == sector)
        if existing_count >= target:
            print(f"✓ SKIP {sector}: {existing_count}")
            continue

        need  = target - existing_count
        added = 0
        print(f"\n{sector}: {existing_count} var → {target} hedef ({need} üretiliyor)")

        while added < need:
            pct = int((added/need)*40)
            bar = "█"*pct + "░"*(40-pct)
            print(f"\r  [{bar}] {added}/{need}", end="", flush=True)

            # template seç + augment
            template = random.choice(templates)
            cols     = augment_columns(template[:])
            
            # shuffle ile farklı sıra
            if random.random() < 0.4:
                random.shuffle(cols)
            
            # bazı kolonları çıkar/ekle
            if len(cols) > 5 and random.random() < 0.3:
                cols = random.sample(cols, random.randint(max(4,len(cols)-3), len(cols)))

            col_stats = {col: make_stats(col) for col in cols}
            key = f"synthetic/{sector}/{existing_count + added}"
            
            existing[key] = {
                "sector":       sector,
                "column_names": cols,
                "features": {
                    "column_names": cols,
                    "n_columns":    len(cols),
                    "n_rows":       random.randint(500, 100000),
                    "column_stats": col_stats,
                }
            }
            added += 1

        SAVE_FILE.write_text(json.dumps(existing, indent=2))
        print(f"\r  [{'█'*40}] {added}/{need} ✓")

    counts = Counter(v["sector"] for v in existing.values())
    print(f"\nTOPLAM: {len(existing)}")
    for s, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {s:20s}: {c}")

if __name__ == "__main__":
    main()

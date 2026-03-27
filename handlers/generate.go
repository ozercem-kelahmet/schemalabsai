package handlers

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"log"
	"io"
	"math"
	"math/rand"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
	"schemalabsai/services"
)

type GenerateRequest struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Rows        int    `json:"rows"`
	Columns     int    `json:"columns"`
	Vertical    string `json:"vertical"`
	Prompt      string `json:"prompt"`
	UsePython   bool   `json:"use_python"`
	PythonCode  string `json:"python_code"`
}

type ColSpec struct {
	Name      string
	DataType  string
	Min       float64
	Max       float64
	Options   []string
	TrueRate  float64
	DependsOn string
	DepType   string
}

func GenerateDatasetHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}
	var req GenerateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}
	if req.Rows <= 0 { req.Rows = 1000 }
	if req.Columns <= 0 { req.Columns = 10 }
	if req.Rows > 100000 { req.Rows = 100000 }
	if req.Columns > 100 { req.Columns = 100 }

	// Check quota
	creditCost := math.Round((0.50+float64(req.Rows)/1000.0*0.10+float64(req.Columns)/10.0*0.05)*100) / 100
	if creditCost < 0.50 { creditCost = 0.50 }
	if creditCost > 10.0 { creditCost = 10.0 }
var genErrors []string
if ok, reason := CheckCredits(userID, creditCost); !ok {
genErrors = append(genErrors, reason)
}
estimatedMB := float64(req.Rows) * float64(req.Columns) * 100 / (1024 * 1024)
if ok, reason := CheckStorage(userID, estimatedMB); !ok {
genErrors = append(genErrors, reason)
}
if len(genErrors) > 0 {
w.Header().Set("Content-Type", "application/json")
w.WriteHeader(http.StatusForbidden)
json.NewEncoder(w).Encode(map[string]string{"error": strings.Join(genErrors, " | ")})
return
}

	fileID := uuid.New().String()
	timestamp := time.Now().Format("20060102_150405")
	filename := fmt.Sprintf("%s_%s.csv", sanitizeFilename(req.Name), timestamp)
	destPath := filepath.Join("./uploads", filename)

	var err error
	if req.UsePython {
		err = runPythonGenerate(req, destPath)
	} else {
		err = generateDynamic(req, destPath)
	}
	if err != nil {
		http.Error(w, "Generation failed: "+err.Error(), http.StatusInternalServerError)
		return
	}

	fi, _ := os.Stat(destPath)
	size := int64(0)
	if fi != nil { size = fi.Size() }
	rowCount, colNames := csvStats(destPath)

	// Upload to cloud storage
	storageKey := filename
	if userID != "" {
		storageKey = services.UserKey(userID, "uploads", filename)
	}
	if sf, soerr := os.Open(destPath); soerr == nil {
		if suerr := services.DefaultStorage.Upload(storageKey, sf); suerr != nil {
			log.Printf("[STORAGE] Generate upload failed: %s: %v", storageKey, suerr)
		} else {
			log.Printf("[STORAGE] Generate uploaded: %s (%d bytes)", storageKey, size)
os.Remove(destPath)
		}
		sf.Close()
	}

	creditCost = math.Round((0.50+float64(req.Rows)/1000.0*0.10+float64(req.Columns)/10.0*0.05)*100) / 100
	if creditCost < 0.50 { creditCost = 0.50 }
	if creditCost > 10.0 { creditCost = 10.0 }

	if DB != nil {
		DB.Create(&UploadedFile{ID: fileID, Filename: filename, Path: storageKey, Size: size, UserID: userID, CreatedAt: time.Now(), Columns: colNames, RowCount: rowCount, Vertical: req.Vertical, Source: "generated"})
		var q UserQuota
		if DB.Where("user_id = ?", userID).First(&q).Error == nil { q.CreditsUsed += creditCost; DB.Save(&q) }
		genTokens := rowCount * len(strings.Split(colNames, ",")) * 3
		if genTokens < 500 { genTokens = 500 }
		DB.Create(&UsageLog{ID: uuid.New().String(), UserID: userID, EventType: "generate", EventName: "Synthetic Data Generation", ResourceID: fileID, ResourceName: filename, CreditsUsed: creditCost, TokensUsed: genTokens, ModelUsed: "synthetic", CreatedAt: time.Now()})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"status": "success", "file_id": fileID, "filename": filename, "rows": rowCount, "columns": colNames, "credits_used": creditCost, "download_url": "/api/download/" + fileID})
}

func runPythonGenerate(req GenerateRequest, destPath string) error {
	tmp := filepath.Join(os.TempDir(), "gen_"+uuid.New().String()+".py")
	code := fmt.Sprintf("import pandas as pd\nimport numpy as np\nfrom faker import Faker\nimport random\n%s\nif 'df' in dir():\n    df.to_csv('%s', index=False)\n", req.PythonCode, destPath)
	if err := os.WriteFile(tmp, []byte(code), 0644); err != nil { return err }
	defer os.Remove(tmp)
	out, err := exec.Command("python3", tmp).CombinedOutput()
	if err != nil { return fmt.Errorf("python: %s - %s", err, string(out)) }
	return nil
}

func generateDynamic(req GenerateRequest, destPath string) error {
	rand.Seed(time.Now().UnixNano())
	specs := buildSpecs(req.Prompt, req.Vertical, req.Columns)

	f, err := os.Create(destPath)
	if err != nil { return err }
	defer f.Close()
	w := csv.NewWriter(f)
	defer w.Flush()

	hdrs := make([]string, len(specs))
	for i, s := range specs { hdrs[i] = s.Name }
	w.Write(hdrs)

	for i := 0; i < req.Rows; i++ {
		row := make([]string, len(specs))
		vals := map[string]float64{}
		for j, col := range specs {
			v, n := cellValue(col, i, vals)
			row[j] = v
			vals[col.Name] = n
		}
		w.Write(row)
	}
	return nil
}

func buildSpecs(desc string, vert string, numCols int) []ColSpec {
	var specs []ColSpec
	names := extractCols(desc)
	for _, n := range names {
		if len(specs) >= numCols { break }
		specs = append(specs, toSpec(n, vert))
	}
	for _, d := range vertDefaults(vert) {
		if len(specs) >= numCols { break }
		dup := false
		for _, s := range specs { if s.Name == d.Name { dup = true; break } }
		if !dup { specs = append(specs, d) }
	}
	idx := 1
	for len(specs) < numCols {
		specs = append(specs, ColSpec{Name: fmt.Sprintf("feature_%d", idx), DataType: "float", Min: 0, Max: 100})
		idx++
	}
	specs = setCorrelations(specs)
	return specs[:numCols]
}

func extractCols(desc string) []string {
	if desc == "" { return nil }
	lo := strings.ToLower(desc)
	re := regexp.MustCompile(`(?:columns?|fields?)\s*[:=]?\s*([^.]+?)(?:\.|$)`)
	if m := re.FindStringSubmatch(lo); len(m) > 1 {
		parts := splitCols(m[1])
		if len(parts) > 0 { return parts }
	}
	return extractKW(lo)
}

func splitCols(s string) []string {
	var r []string
	for _, p := range strings.Split(s, ",") {
		p = strings.TrimSpace(p)
		p = strings.ReplaceAll(p, " ", "_")
		p = regexp.MustCompile(`[^a-z0-9_]`).ReplaceAllString(p, "")
		if len(p) > 1 { r = append(r, p) }
	}
	return r
}

func extractKW(text string) []string {
	skip := map[string]bool{"generate":true,"create":true,"data":true,"with":true,"and":true,"the":true,"for":true,"that":true,"this":true,"from":true,"have":true,"each":true,"like":true,"include":true,"column":true,"columns":true,"row":true,"rows":true,"should":true,"would":true,"could":true,"also":true,"some":true,"any":true,"all":true,"between":true,"about":true,"example":true,"such":true,"more":true,"other":true,"make":true,"need":true,"want":true,"use":true,"using":true,"dataset":true,"table":true,"values":true,"types":true,"specific":true,"patterns":true,"distributions":true,"describe":true,"including":true,"where":true,"which":true,"will":true,"can":true,"are":true,"was":true,"been":true,"being":true,"has":true,"please":true,"add":true,"set":true,"get":true,"new":true,"based":true,"real":true,"realistic":true,"sample":true}
	words := regexp.MustCompile(`[a-z_]+`).FindAllString(text, -1)
	seen := map[string]bool{}
	var r []string
	for _, w := range words {
		if len(w) > 2 && !skip[w] && !seen[w] { seen[w] = true; r = append(r, w) }
	}
	return r
}

func cAny(s string, kws []string) bool {
	for _, k := range kws { if strings.Contains(s, k) { return true } }
	return false
}

func toSpec(name string, vert string) ColSpec {
	n := strings.ToLower(name)
	s := ColSpec{Name: name}
	if n == "id" || strings.HasSuffix(n, "_id") { s.DataType = "id"; return s }
	if cAny(n, []string{"date","time","timestamp","created","updated","born","signup","admission","discharge","hire","delivery","enrolled"}) { s.DataType = "date"; return s }
	if strings.Contains(n, "email") { s.DataType = "email"; return s }
	if (strings.Contains(n, "name") || n == "first_name" || n == "last_name") && !cAny(n, []string{"file","model"}) { s.DataType = "name"; return s }
	if cAny(n, []string{"phone","mobile"}) { s.DataType = "phone"; return s }
	if cAny(n, []string{"address","street"}) { s.DataType = "address"; return s }
	if strings.HasPrefix(n, "is_") || strings.HasPrefix(n, "has_") { s.DataType = "bool"; s.TrueRate = 0.3; return s }
	for _, kw := range []string{"churned","returned","fraud","active","verified","on_time","readmission","remote","scholarship","peak","cancelled","sold","injured","resolved","premium","recurring","completed","escalated","spam","anomaly","deductible","repeat"} {
		if strings.Contains(n, kw) {
			s.DataType = "bool"
			switch { case cAny(n,[]string{"fraud","anomaly","spam"}): s.TrueRate=0.03; case cAny(n,[]string{"active","on_time","verified","deductible"}): s.TrueRate=0.85; case cAny(n,[]string{"churn","return","cancel"}): s.TrueRate=0.12; default: s.TrueRate=0.3 }
			return s
		}
	}
	if cAny(n, []string{"price","cost","amount","revenue","salary","spend","budget","fee","payment","income","profit","total","tuition","balance","premium","mrr","billing","charge"}) {
		s.DataType = "float"
		switch { case cAny(n,[]string{"salary","income"}): s.Min,s.Max=30000,200000; case cAny(n,[]string{"balance"}): s.Min,s.Max=0,100000; case cAny(n,[]string{"total","revenue","billing"}): s.Min,s.Max=10,50000; case cAny(n,[]string{"mrr"}): s.Min,s.Max=0,10000; default: s.Min,s.Max=5,10000 }
		return s
	}
	if cAny(n, []string{"score","rating","satisfaction","performance","gpa","nps"}) {
		s.DataType = "float"
		switch { case strings.Contains(n,"gpa"): s.Min,s.Max=1,4; case cAny(n,[]string{"satisfaction","rating"}): s.Min,s.Max=1,5; case strings.Contains(n,"nps"): s.Min,s.Max=-100,100; default: s.Min,s.Max=0,100 }
		return s
	}
	if cAny(n, []string{"rate","ratio","pct","probability","risk","ctr","conversion_rate","roi","efficiency","attendance","engagement","confidence","readability"}) {
		s.DataType = "float"; s.Min = 0; s.Max = 1; return s
	}
	if n == "age" { s.DataType = "int"; s.Min = 18; s.Max = 85; return s }
	if cAny(n, []string{"year_built","graduation_year","model_year"}) { s.DataType = "int"; s.Min = 1960; s.Max = 2026; return s }
	if cAny(n, []string{"count","quantity","qty","visits","clicks","views","impressions","orders","conversions","training_hours","credits_completed","tenure","days_on_market","bedrooms","bathrooms","parking","support_tickets","sessions","achievements","goals","assists","minutes_played","velocity","word_count","entities","uptime"}) {
		s.DataType = "int"
		switch { case cAny(n,[]string{"impression","view","follower"}): s.Min,s.Max=100,1000000; case cAny(n,[]string{"click"}): s.Min,s.Max=0,50000; case cAny(n,[]string{"bedroom","bathroom","parking"}): s.Min,s.Max=0,6; case cAny(n,[]string{"tenure"}): s.Min,s.Max=1,240; default: s.Min,s.Max=0,500 }
		return s
	}
	if cAny(n, []string{"sqft","area","lot_size","size"}) { s.DataType = "float"; s.Min = 500; s.Max = 10000; return s }
	if cAny(n, []string{"weight","height","bmi"}) { s.DataType = "float"; if strings.Contains(n,"bmi") { s.Min,s.Max=16,40 } else { s.Min,s.Max=40,120 }; return s }
	if cAny(n, []string{"temperature","temp"}) { s.DataType = "float"; s.Min = -10; s.Max = 45; return s }
	if cAny(n, []string{"duration","processing_time","length_of_stay","resolution_time","watch_time","session_duration","delivery_time","cycle_time","response_time"}) { s.DataType = "float"; s.Min = 0.5; s.Max = 120; return s }
	if cAny(n, []string{"consumption","kwh","demand","generation","grid_price"}) { s.DataType = "float"; s.Min = 0; s.Max = 1000; return s }
	if cAny(n, []string{"humidity","occupancy","battery","brightness","quality"}) { s.DataType = "float"; s.Min = 0; s.Max = 100; return s }
	if cAny(n, []string{"discount","margin","defect","interest"}) { s.DataType = "float"; s.Min = 0; s.Max = 0.5; return s }
	if cAny(n, []string{"signal"}) { s.DataType = "float"; s.Min = -100; s.Max = 0; return s }
	if cAny(n, []string{"mileage"}) { s.DataType = "int"; s.Min = 0; s.Max = 200000; return s }
	if cAny(n, []string{"credit_score"}) { s.DataType = "int"; s.Min = 300; s.Max = 850; return s }
	if cAny(n, []string{"beneficiaries"}) { s.DataType = "int"; s.Min = 100; s.Max = 500000; return s }
	if cAny(n, []string{"rainfall","precipitation"}) { s.DataType = "float"; s.Min = 0; s.Max = 300; return s }
	if cAny(n, []string{"yield"}) { s.DataType = "float"; s.Min = 0.5; s.Max = 50; return s }
	if cAny(n, []string{"dosage"}) { s.DataType = "float"; s.Min = 5; s.Max = 500; return s }
	if cAny(n, []string{"pressure"}) { s.DataType = "float"; s.Min = 970; s.Max = 1050; return s }
	if cAny(n, []string{"wind"}) { s.DataType = "float"; s.Min = 0; s.Max = 150; return s }
	if cAny(n, []string{"uv_index"}) { s.DataType = "int"; s.Min = 0; s.Max = 11; return s }
	if cAny(n, []string{"visibility"}) { s.DataType = "float"; s.Min = 0.1; s.Max = 50; return s }
	if cAny(n, []string{"gas_fee","tip","cpc"}) { s.DataType = "float"; s.Min = 0; s.Max = 50; return s }
	if cAny(n, []string{"roas"}) { s.DataType = "float"; s.Min = 0; s.Max = 10; return s }
	if cAny(n, []string{"level"}) && !cAny(n, []string{"job","battery","signal"}) { s.DataType = "int"; s.Min = 1; s.Max = 100; return s }
	if n == "day_of_week" || n == "weekday" { s.DataType = "category"; s.Options = []string{"Mon","Tue","Wed","Thu","Fri","Sat","Sun"}; return s }
	if n == "month" { s.DataType = "category"; s.Options = []string{"Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"}; return s }
	if cAny(n, []string{"description","notes","comment","feedback","review","text","message"}) { s.DataType = "text"; return s }
	// Category detection
	cats := []string{"type","category","status","level","tier","plan","segment","channel","method","region","country","city","department","gender","priority","source","platform","device","class","grade","group","warehouse","supplier","blood","education","building","property","creative","insurance","shipping","merchant","currency","diagnosis","color","brand","material","language","role","sector","industry","condition","format","label","topic","trend","cuisine","carrier","fuel","soil","outcome","jurisdiction","phase","contract","shift","crop","coin","exchange","interaction","firmware","position","team","campaign","post_type","content_type","ad_format","booking_source","donor_type","program_area","age_group","event_type","severity","target_system","sensor_type","location","make"}
	for _, kw := range cats {
		if strings.Contains(n, kw) { s.DataType = "category"; s.Options = catOptions(n, vert); return s }
	}
	s.DataType = "float"; s.Min = 0; s.Max = 100; return s
}

func catOptions(name string, vert string) []string {
	n := strings.ToLower(name)
	m := map[string][]string{
		"gender":{"Male","Female","Non-binary"},"status":{"active","inactive","pending","completed"},"priority":{"low","medium","high","critical"},"tier":{"free","basic","pro","enterprise"},"plan":{"free","starter","pro","enterprise"},"level":{"junior","mid","senior","lead","manager"},"channel":{"email","social","search","display","referral","direct"},"payment":{"credit_card","debit_card","bank_transfer","paypal"},"device":{"desktop","mobile","tablet"},"platform":{"web","ios","android"},"region":{"North","South","East","West","Central"},"country":{"US","UK","DE","FR","JP","BR","IN","CA","AU"},"city":{"New York","Los Angeles","Chicago","Houston","Phoenix","San Francisco","Seattle","Miami"},"department":{"engineering","sales","marketing","hr","finance","operations","support"},"source":{"organic","paid","referral","direct","social"},"segment":{"new","returning","vip","at_risk","dormant"},"shipping":{"standard","express","same_day","pickup"},"insurance":{"private","public","none"},"class":{"economy","business","first"},"blood":{"A+","A-","B+","B-","AB+","AB-","O+","O-"},"education":{"high_school","bachelors","masters","phd"},"building":{"residential","commercial","industrial","mixed"},"property":{"house","condo","apartment","townhouse","land"},"warehouse":{"WH-A","WH-B","WH-C","WH-D"},"supplier":{"SUP-001","SUP-002","SUP-003","SUP-004","SUP-005"},"merchant":{"Amazon","Walmart","Target","Costco","BestBuy","Starbucks"},"currency":{"USD","EUR","GBP","JPY","CAD"},"grade":{"A","B","C","D","F"},"brand":{"Apple","Samsung","Sony","LG","Dell","Nike","Adidas"},"material":{"cotton","polyester","leather","wood","metal","plastic"},"language":{"en","es","fr","de","zh","ja"},"role":{"admin","editor","viewer","member"},"sector":{"technology","finance","healthcare","retail","energy","education"},"industry":{"tech","finance","healthcare","retail","manufacturing","energy"},"color":{"red","blue","green","black","white"},"condition":{"new","excellent","good","fair","poor"},"fuel":{"gasoline","diesel","electric","hybrid"},"soil":{"poor","fair","good","excellent"},"outcome":{"won","lost","settled","dismissed"},"jurisdiction":{"federal","state","local"},"phase":{"Phase_I","Phase_II","Phase_III","Phase_IV"},"contract":{"monthly","annual","two_year"},"shift":{"morning","afternoon","night"},"crop":{"wheat","corn","rice","soybean","cotton"},"coin":{"BTC","ETH","SOL","ADA","DOT"},"exchange":{"Binance","Coinbase","Kraken"},"interaction":{"view","click","purchase","wishlist"},"firmware":{"v1.0","v1.1","v2.0","v2.1"},"position":{"forward","midfielder","defender","goalkeeper"},"team":{"Team_A","Team_B","Team_C","Team_D","Team_E"},"cuisine":{"italian","chinese","mexican","japanese","american","indian"},"carrier":{"FedEx","UPS","DHL","USPS"},"make":{"Toyota","Honda","Ford","BMW","Tesla","Mercedes"},"diagnosis":{"J06","I10","E11","M54","F32","K21"},"content_type":{"article","video","podcast","social_post"},"post_type":{"image","video","text","story","reel","carousel"},"ad_format":{"display","video","search","native","social"},"booking_source":{"direct","ota","travel_agent","corporate"},"donor_type":{"individual","corporate","foundation"},"program_area":{"education","health","environment","poverty"},"age_group":{"18-24","25-34","35-44","45-54","55+"},"event_type":{"intrusion","malware","phishing","ddos","ransomware"},"severity":{"low","medium","high","critical"},"target_system":{"web_server","database","email","firewall","endpoint"},"sensor_type":{"temperature","humidity","pressure","motion","light"},"location":{"floor_1","floor_2","floor_3","outdoor","basement"},"topic":{"product","service","pricing","support","general"},"trend":{"increasing","decreasing","stable"},"label":{"cat","dog","car","person","building","nature"},"format":{"jpg","png","webp","tiff"},"campaign":{"Brand_Awareness","Product_Launch","Retargeting","Seasonal"},
	}
	// Vertical-specific overrides for "category"
	vc := map[string][]string{"finance":{"retail","food","travel","utilities","entertainment"},"healthcare":{"cardiology","neurology","oncology","orthopedics","pediatrics"},"e-commerce":{"electronics","clothing","home","beauty","sports","books"},"marketing":{"awareness","consideration","conversion","retention"},"hr":{"engineering","sales","marketing","hr","finance"},"operations":{"raw_material","finished_goods","packaging","spare_parts"},"real-estate":{"residential","commercial","industrial","land"},"education":{"STEM","humanities","business","arts"},"energy":{"solar","wind","hydro","natural_gas","nuclear"},"sports":{"offense","defense","special_teams"},"media":{"news","entertainment","sports","tech","lifestyle"},"travel":{"business","leisure","adventure","luxury"},"food":{"appetizer","main","dessert","beverage"},"automotive":{"sedan","suv","truck","sports","van"},"agriculture":{"grain","fruit","vegetable","livestock"},"pharma":{"analgesic","antibiotic","antiviral","vaccine"},"legal":{"civil","criminal","corporate","family"},"government":{"infrastructure","social","defense","education"},"nonprofit":{"education","health","environment","arts"},"cybersecurity":{"network","endpoint","cloud","identity"},"iot":{"home","industrial","wearable","automotive"},"gaming":{"action","rpg","strategy","sports","puzzle"},"crypto":{"DeFi","NFT","Layer1","Layer2","stablecoin"},"saas":{"CRM","analytics","security","communication"},"customer-service":{"billing","technical","account","shipping"},"sales":{"inbound","outbound","partnership","referral"},"advertising":{"display","video","search","native"},"social-media":{"organic","paid","influencer","ugc"},"weather":{"sunny","cloudy","rainy","snowy","stormy"},"demographics":{"urban","suburban","rural"},"survey":{"satisfied","neutral","dissatisfied"},"sentiment":{"positive","neutral","negative"},"fraud":{"legitimate","suspicious","fraudulent"},"churn":{"retained","at_risk","churned"},"recommendation":{"collaborative","content_based","hybrid"},"time-series":{"trend","seasonal","residual"},"nlp":{"news","review","social","academic"},"image":{"cat","dog","car","person","building"},"retail":{"grocery","apparel","electronics","home","pharmacy"},"manufacturing":{"assembly","machining","welding","painting"},"logistics":{"ground","air","sea","rail"},"insurance":{"auto","home","life","health","travel"},"banking":{"checking","savings","credit","loan"},"telecom":{"prepaid","postpaid","broadband","enterprise"},}
	for key, opts := range m {
		if strings.Contains(n, key) { return opts }
	}
	if n == "category" || strings.HasSuffix(n, "_category") {
		if o, ok := vc[vert]; ok { return o }
	}
	return []string{"type_a","type_b","type_c","type_d","type_e"}
}

func setCorrelations(specs []ColSpec) []ColSpec {
	idx := map[string]int{}
	for i, s := range specs { idx[s.Name] = i }
	for i, s := range specs {
		n := strings.ToLower(s.Name)
		if strings.Contains(n, "total") { if _, ok := idx["quantity"]; ok { if _, ok2 := idx["unit_price"]; ok2 { specs[i].DependsOn = "quantity"; specs[i].DepType = "mul_price" } } }
		if strings.Contains(n, "revenue") { if _, ok := idx["spend"]; ok { specs[i].DependsOn = "spend"; specs[i].DepType = "mul_roi" } }
		if strings.Contains(n, "click") && s.DataType == "int" { if _, ok := idx["impressions"]; ok { specs[i].DependsOn = "impressions"; specs[i].DepType = "ctr" } }
		if strings.Contains(n, "conversion") && s.DataType == "int" { if _, ok := idx["clicks"]; ok { specs[i].DependsOn = "clicks"; specs[i].DepType = "cvr" } }
	}
	return specs
}

func cellValue(col ColSpec, row int, vals map[string]float64) (string, float64) {
	if col.DependsOn != "" {
		if dv, ok := vals[col.DependsOn]; ok {
			switch col.DepType {
			case "mul_price": if pv, ok2 := vals["unit_price"]; ok2 { t := dv * pv; return fmt.Sprintf("%.2f", t), t }
			case "mul_roi": r := 0.8 + rand.Float64()*2.5; v := dv * r; return fmt.Sprintf("%.2f", v), v
			case "ctr": v := int(dv * (0.01 + rand.Float64()*0.08)); return strconv.Itoa(v), float64(v)
			case "cvr": v := int(dv * (0.02 + rand.Float64()*0.15)); if v < 0 { v = 0 }; return strconv.Itoa(v), float64(v)
			}
		}
	}
	switch col.DataType {
	case "id":
		p := strings.ToUpper(col.Name); if len(p) > 4 { p = p[:4] }; return fmt.Sprintf("%s_%06d", p, 10000+row), float64(row)
	case "int":
		rng := col.Max - col.Min; if rng <= 0 { rng = 100 }
		v := rand.NormFloat64()*(rng/6) + (col.Min+col.Max)/2
		if v < col.Min { v = col.Min }; if v > col.Max { v = col.Max }
		iv := int(math.Round(v)); return strconv.Itoa(iv), float64(iv)
	case "float":
		rng := col.Max - col.Min; if rng <= 0 { rng = 100 }
		v := rand.NormFloat64()*(rng/6) + (col.Min+col.Max)/2
		if v < col.Min { v = col.Min }; if v > col.Max { v = col.Max }
		return fmt.Sprintf("%.2f", v), v
	case "date":
		d := time.Now().AddDate(0, 0, -rand.Intn(730)); return d.Format("2006-01-02"), 0
	case "bool":
		if rand.Float64() < col.TrueRate { return "true", 1 }; return "false", 0
	case "category":
		if len(col.Options) > 0 { return col.Options[rand.Intn(len(col.Options))], 0 }; return "unknown", 0
	case "name":
		fn := []string{"James","Mary","John","Patricia","Robert","Jennifer","Michael","Linda","David","Elizabeth","Emma","Oliver","Sophia","Liam","Ava","Noah","Isabella","Lucas","Mia","Ethan"}
		ln := []string{"Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis","Rodriguez","Martinez","Wilson","Anderson","Taylor","Thomas","Moore","Lee"}
		return fn[rand.Intn(len(fn))]+" "+ln[rand.Intn(len(ln))], 0
	case "email":
		dm := []string{"gmail.com","yahoo.com","outlook.com","company.com"}; return fmt.Sprintf("user_%d@%s", 1000+row, dm[rand.Intn(len(dm))]), 0
	case "phone":
		return fmt.Sprintf("+1-%03d-%03d-%04d", 200+rand.Intn(800), 100+rand.Intn(900), 1000+rand.Intn(9000)), 0
	case "address":
		st := []string{"Main St","Oak Ave","Park Blvd","Broadway","Elm St","Cedar Ln","Maple Dr","Pine Rd"}; return fmt.Sprintf("%d %s", 100+rand.Intn(9900), st[rand.Intn(len(st))]), 0
	case "text":
		tx := []string{"Good experience","Needs improvement","Excellent service","Below expectations","Highly recommended","Average quality","Outstanding results","Could be better","Very satisfied","Great value"}
		return tx[rand.Intn(len(tx))], 0
	}
	return fmt.Sprintf("val_%d", row), 0
}

func vertDefaults(v string) []ColSpec {
	all := map[string][]ColSpec{
		"finance":{{Name:"transaction_id",DataType:"id"},{Name:"customer_id",DataType:"id"},{Name:"amount",DataType:"float",Min:10,Max:10000},{Name:"transaction_date",DataType:"date"},{Name:"category",DataType:"category",Options:[]string{"retail","food","travel","utilities","entertainment"}},{Name:"payment_method",DataType:"category",Options:[]string{"credit_card","debit_card","bank_transfer","cash"}},{Name:"is_fraud",DataType:"bool",TrueRate:0.03},{Name:"risk_score",DataType:"float",Min:0,Max:1},{Name:"balance",DataType:"float",Min:0,Max:100000},{Name:"region",DataType:"category",Options:[]string{"North","South","East","West"}}},
		"healthcare":{{Name:"patient_id",DataType:"id"},{Name:"age",DataType:"int",Min:18,Max:90},{Name:"gender",DataType:"category",Options:[]string{"Male","Female"}},{Name:"diagnosis_code",DataType:"category",Options:[]string{"J06","I10","E11","M54","F32"}},{Name:"admission_date",DataType:"date"},{Name:"treatment_cost",DataType:"float",Min:100,Max:50000},{Name:"insurance_type",DataType:"category",Options:[]string{"private","public","none"}},{Name:"readmission",DataType:"bool",TrueRate:0.15},{Name:"satisfaction_score",DataType:"float",Min:1,Max:5},{Name:"department",DataType:"category",Options:[]string{"cardiology","neurology","oncology","orthopedics"}}},
		"e-commerce":{{Name:"order_id",DataType:"id"},{Name:"customer_id",DataType:"id"},{Name:"product_category",DataType:"category",Options:[]string{"electronics","clothing","home","beauty","sports"}},{Name:"quantity",DataType:"int",Min:1,Max:10},{Name:"unit_price",DataType:"float",Min:5,Max:500},{Name:"total_amount",DataType:"float",Min:5,Max:5000},{Name:"order_date",DataType:"date"},{Name:"shipping_method",DataType:"category",Options:[]string{"standard","express","same_day"}},{Name:"returned",DataType:"bool",TrueRate:0.08},{Name:"rating",DataType:"float",Min:1,Max:5}},
		"marketing":{{Name:"campaign_id",DataType:"id"},{Name:"channel",DataType:"category",Options:[]string{"email","social","search","display","referral"}},{Name:"impressions",DataType:"int",Min:100,Max:100000},{Name:"clicks",DataType:"int",Min:0,Max:5000},{Name:"conversions",DataType:"int",Min:0,Max:500},{Name:"spend",DataType:"float",Min:10,Max:10000},{Name:"revenue",DataType:"float",Min:0,Max:50000},{Name:"campaign_date",DataType:"date"},{Name:"target_segment",DataType:"category",Options:[]string{"new","returning","vip","at_risk"}},{Name:"platform",DataType:"category",Options:[]string{"web","ios","android"}}},
		"hr":{{Name:"employee_id",DataType:"id"},{Name:"name",DataType:"name"},{Name:"department",DataType:"category",Options:[]string{"engineering","sales","marketing","hr","finance","operations"}},{Name:"job_level",DataType:"category",Options:[]string{"junior","mid","senior","lead","manager"}},{Name:"hire_date",DataType:"date"},{Name:"salary",DataType:"float",Min:30000,Max:200000},{Name:"performance_score",DataType:"float",Min:1,Max:5},{Name:"tenure_months",DataType:"int",Min:1,Max:240},{Name:"is_remote",DataType:"bool",TrueRate:0.3},{Name:"churned",DataType:"bool",TrueRate:0.12}},
		"operations":{{Name:"order_id",DataType:"id"},{Name:"warehouse",DataType:"category",Options:[]string{"WH-A","WH-B","WH-C","WH-D"}},{Name:"quantity",DataType:"int",Min:1,Max:1000},{Name:"processing_time_hours",DataType:"float",Min:0.5,Max:72},{Name:"shipping_cost",DataType:"float",Min:5,Max:500},{Name:"delivery_date",DataType:"date"},{Name:"on_time",DataType:"bool",TrueRate:0.92},{Name:"defect_rate",DataType:"float",Min:0,Max:0.05},{Name:"supplier",DataType:"category",Options:[]string{"SUP-001","SUP-002","SUP-003","SUP-004","SUP-005"}},{Name:"priority",DataType:"category",Options:[]string{"low","medium","high","critical"}}},
		"retail":{{Name:"transaction_id",DataType:"id"},{Name:"store_id",DataType:"id"},{Name:"product_category",DataType:"category",Options:[]string{"grocery","apparel","electronics","home","pharmacy"}},{Name:"quantity",DataType:"int",Min:1,Max:50},{Name:"unit_price",DataType:"float",Min:1,Max:500},{Name:"total_amount",DataType:"float",Min:1,Max:5000},{Name:"purchase_date",DataType:"date"},{Name:"payment_method",DataType:"category",Options:[]string{"cash","credit_card","debit_card","mobile_pay"}},{Name:"customer_segment",DataType:"category",Options:[]string{"regular","premium","new"}},{Name:"region",DataType:"category",Options:[]string{"North","South","East","West"}}},
		"manufacturing":{{Name:"batch_id",DataType:"id"},{Name:"machine_id",DataType:"id"},{Name:"product_type",DataType:"category",Options:[]string{"type_A","type_B","type_C"}},{Name:"production_date",DataType:"date"},{Name:"units_produced",DataType:"int",Min:50,Max:5000},{Name:"defect_count",DataType:"int",Min:0,Max:50},{Name:"cycle_time_min",DataType:"float",Min:1,Max:120},{Name:"oee_score",DataType:"float",Min:0.4,Max:1},{Name:"downtime_hours",DataType:"float",Min:0,Max:8},{Name:"shift",DataType:"category",Options:[]string{"morning","afternoon","night"}}},
		"logistics":{{Name:"shipment_id",DataType:"id"},{Name:"origin",DataType:"category",Options:[]string{"New York","Chicago","Dallas","LA","Atlanta"}},{Name:"destination",DataType:"category",Options:[]string{"Miami","Seattle","Boston","Denver","Phoenix"}},{Name:"weight_kg",DataType:"float",Min:0.5,Max:500},{Name:"shipping_date",DataType:"date"},{Name:"carrier",DataType:"category",Options:[]string{"FedEx","UPS","DHL","USPS"}},{Name:"cost",DataType:"float",Min:5,Max:2000},{Name:"on_time",DataType:"bool",TrueRate:0.88},{Name:"damage_reported",DataType:"bool",TrueRate:0.02},{Name:"priority",DataType:"category",Options:[]string{"standard","express","overnight"}}},
		"real-estate":{{Name:"property_id",DataType:"id"},{Name:"price",DataType:"float",Min:50000,Max:2000000},{Name:"bedrooms",DataType:"int",Min:1,Max:6},{Name:"bathrooms",DataType:"int",Min:1,Max:4},{Name:"sqft",DataType:"float",Min:500,Max:5000},{Name:"year_built",DataType:"int",Min:1950,Max:2026},{Name:"property_type",DataType:"category",Options:[]string{"house","condo","apartment","townhouse"}},{Name:"city",DataType:"category",Options:[]string{"New York","LA","Chicago","Houston","Phoenix"}},{Name:"days_on_market",DataType:"int",Min:1,Max:365},{Name:"status",DataType:"category",Options:[]string{"active","pending","sold"}}},
		"insurance":{{Name:"policy_id",DataType:"id"},{Name:"customer_id",DataType:"id"},{Name:"policy_type",DataType:"category",Options:[]string{"auto","home","life","health","travel"}},{Name:"premium",DataType:"float",Min:50,Max:5000},{Name:"claim_amount",DataType:"float",Min:0,Max:50000},{Name:"start_date",DataType:"date"},{Name:"age",DataType:"int",Min:18,Max:80},{Name:"has_claim",DataType:"bool",TrueRate:0.2},{Name:"risk_level",DataType:"category",Options:[]string{"low","medium","high"}},{Name:"region",DataType:"category",Options:[]string{"North","South","East","West"}}},
		"banking":{{Name:"account_id",DataType:"id"},{Name:"account_type",DataType:"category",Options:[]string{"checking","savings","credit","loan"}},{Name:"balance",DataType:"float",Min:0,Max:500000},{Name:"transaction_amount",DataType:"float",Min:1,Max:25000},{Name:"transaction_date",DataType:"date"},{Name:"credit_score",DataType:"int",Min:300,Max:850},{Name:"is_default",DataType:"bool",TrueRate:0.04},{Name:"loan_term_months",DataType:"int",Min:6,Max:360},{Name:"interest_rate",DataType:"float",Min:0.01,Max:0.25},{Name:"region",DataType:"category",Options:[]string{"North","South","East","West"}}},
		"telecom":{{Name:"customer_id",DataType:"id"},{Name:"plan_type",DataType:"category",Options:[]string{"basic","standard","premium","unlimited"}},{Name:"monthly_charge",DataType:"float",Min:20,Max:200},{Name:"data_usage_gb",DataType:"float",Min:0.5,Max:100},{Name:"call_minutes",DataType:"int",Min:0,Max:3000},{Name:"tenure_months",DataType:"int",Min:1,Max:72},{Name:"contract_type",DataType:"category",Options:[]string{"monthly","annual","two_year"}},{Name:"churned",DataType:"bool",TrueRate:0.15},{Name:"support_tickets",DataType:"int",Min:0,Max:10},{Name:"satisfaction_score",DataType:"float",Min:1,Max:5}},
		"energy":{{Name:"meter_id",DataType:"id"},{Name:"reading_date",DataType:"date"},{Name:"consumption_kwh",DataType:"float",Min:0,Max:500},{Name:"temperature",DataType:"float",Min:-10,Max:45},{Name:"humidity",DataType:"float",Min:10,Max:100},{Name:"building_type",DataType:"category",Options:[]string{"residential","commercial","industrial"}},{Name:"peak_demand",DataType:"float",Min:0,Max:1000},{Name:"is_peak_hour",DataType:"bool",TrueRate:0.35},{Name:"efficiency_score",DataType:"float",Min:0,Max:1},{Name:"region",DataType:"category",Options:[]string{"North","South","East","West"}}},
		"education":{{Name:"student_id",DataType:"id"},{Name:"name",DataType:"name"},{Name:"age",DataType:"int",Min:17,Max:30},{Name:"department",DataType:"category",Options:[]string{"STEM","humanities","business","arts"}},{Name:"gpa",DataType:"float",Min:1,Max:4},{Name:"enrollment_date",DataType:"date"},{Name:"credits_completed",DataType:"int",Min:0,Max:150},{Name:"attendance_rate",DataType:"float",Min:0.5,Max:1},{Name:"scholarship",DataType:"bool",TrueRate:0.2},{Name:"status",DataType:"category",Options:[]string{"enrolled","graduated","dropped"}}},
		"sports":{{Name:"player_id",DataType:"id"},{Name:"name",DataType:"name"},{Name:"team",DataType:"category",Options:[]string{"Team_A","Team_B","Team_C","Team_D"}},{Name:"position",DataType:"category",Options:[]string{"forward","midfielder","defender","goalkeeper"}},{Name:"age",DataType:"int",Min:18,Max:40},{Name:"goals_scored",DataType:"int",Min:0,Max:50},{Name:"assists",DataType:"int",Min:0,Max:30},{Name:"minutes_played",DataType:"int",Min:0,Max:3500},{Name:"rating",DataType:"float",Min:1,Max:10},{Name:"is_injured",DataType:"bool",TrueRate:0.1}},
		"media":{{Name:"content_id",DataType:"id"},{Name:"content_type",DataType:"category",Options:[]string{"article","video","podcast","social_post"}},{Name:"publish_date",DataType:"date"},{Name:"views",DataType:"int",Min:100,Max:1000000},{Name:"likes",DataType:"int",Min:0,Max:50000},{Name:"shares",DataType:"int",Min:0,Max:10000},{Name:"watch_time_min",DataType:"float",Min:0.5,Max:120},{Name:"engagement_rate",DataType:"float",Min:0,Max:0.3},{Name:"platform",DataType:"category",Options:[]string{"YouTube","Instagram","TikTok","Twitter","web"}},{Name:"category",DataType:"category",Options:[]string{"news","entertainment","sports","tech"}}},
		"travel":{{Name:"booking_id",DataType:"id"},{Name:"destination",DataType:"category",Options:[]string{"Paris","Tokyo","New York","London","Dubai","Bali"}},{Name:"booking_date",DataType:"date"},{Name:"check_in_date",DataType:"date"},{Name:"nights",DataType:"int",Min:1,Max:21},{Name:"total_cost",DataType:"float",Min:100,Max:10000},{Name:"booking_source",DataType:"category",Options:[]string{"direct","ota","travel_agent","corporate"}},{Name:"rating",DataType:"float",Min:1,Max:5},{Name:"is_cancelled",DataType:"bool",TrueRate:0.12},{Name:"traveler_type",DataType:"category",Options:[]string{"business","leisure","family"}}},
		"food":{{Name:"order_id",DataType:"id"},{Name:"cuisine_type",DataType:"category",Options:[]string{"italian","chinese","mexican","japanese","american","indian"}},{Name:"order_date",DataType:"date"},{Name:"total_amount",DataType:"float",Min:5,Max:200},{Name:"delivery_time_min",DataType:"int",Min:10,Max:90},{Name:"rating",DataType:"float",Min:1,Max:5},{Name:"order_type",DataType:"category",Options:[]string{"dine_in","takeout","delivery"}},{Name:"tip_amount",DataType:"float",Min:0,Max:30},{Name:"is_repeat_customer",DataType:"bool",TrueRate:0.4},{Name:"payment_method",DataType:"category",Options:[]string{"cash","card","mobile"}}},
		"automotive":{{Name:"vehicle_id",DataType:"id"},{Name:"make",DataType:"category",Options:[]string{"Toyota","Honda","Ford","BMW","Tesla","Mercedes"}},{Name:"model_year",DataType:"int",Min:2010,Max:2026},{Name:"mileage",DataType:"int",Min:0,Max:200000},{Name:"price",DataType:"float",Min:5000,Max:100000},{Name:"fuel_type",DataType:"category",Options:[]string{"gasoline","diesel","electric","hybrid"}},{Name:"condition",DataType:"category",Options:[]string{"new","excellent","good","fair"}},{Name:"days_in_inventory",DataType:"int",Min:1,Max:180},{Name:"is_sold",DataType:"bool",TrueRate:0.65},{Name:"region",DataType:"category",Options:[]string{"Northeast","Southeast","Midwest","West"}}},
		"agriculture":{{Name:"farm_id",DataType:"id"},{Name:"crop_type",DataType:"category",Options:[]string{"wheat","corn","rice","soybean","cotton"}},{Name:"planting_date",DataType:"date"},{Name:"harvest_date",DataType:"date"},{Name:"yield_tons",DataType:"float",Min:0.5,Max:50},{Name:"area_hectares",DataType:"float",Min:1,Max:500},{Name:"rainfall_mm",DataType:"float",Min:0,Max:300},{Name:"temperature_avg",DataType:"float",Min:5,Max:40},{Name:"fertilizer_kg",DataType:"float",Min:0,Max:500},{Name:"soil_quality",DataType:"category",Options:[]string{"poor","fair","good","excellent"}}},
		"pharma":{{Name:"trial_id",DataType:"id"},{Name:"patient_id",DataType:"id"},{Name:"drug_name",DataType:"category",Options:[]string{"Drug_A","Drug_B","Drug_C","Placebo"}},{Name:"dosage_mg",DataType:"float",Min:5,Max:500},{Name:"trial_phase",DataType:"category",Options:[]string{"Phase_I","Phase_II","Phase_III"}},{Name:"start_date",DataType:"date"},{Name:"efficacy_score",DataType:"float",Min:0,Max:1},{Name:"adverse_event",DataType:"bool",TrueRate:0.08},{Name:"age",DataType:"int",Min:18,Max:80},{Name:"gender",DataType:"category",Options:[]string{"Male","Female"}}},
		"legal":{{Name:"case_id",DataType:"id"},{Name:"case_type",DataType:"category",Options:[]string{"civil","criminal","corporate","family"}},{Name:"filing_date",DataType:"date"},{Name:"duration_days",DataType:"int",Min:7,Max:1000},{Name:"billing_amount",DataType:"float",Min:500,Max:100000},{Name:"outcome",DataType:"category",Options:[]string{"won","lost","settled","dismissed"}},{Name:"attorney_id",DataType:"id"},{Name:"jurisdiction",DataType:"category",Options:[]string{"federal","state","local"}},{Name:"is_appealed",DataType:"bool",TrueRate:0.15},{Name:"priority",DataType:"category",Options:[]string{"low","medium","high"}}},
		"government":{{Name:"project_id",DataType:"id"},{Name:"department",DataType:"category",Options:[]string{"transportation","education","health","defense","environment"}},{Name:"budget",DataType:"float",Min:10000,Max:10000000},{Name:"actual_spend",DataType:"float",Min:10000,Max:10000000},{Name:"start_date",DataType:"date"},{Name:"status",DataType:"category",Options:[]string{"planned","in_progress","completed","delayed"}},{Name:"region",DataType:"category",Options:[]string{"North","South","East","West","Central"}},{Name:"beneficiaries",DataType:"int",Min:100,Max:500000},{Name:"satisfaction_score",DataType:"float",Min:1,Max:5},{Name:"priority",DataType:"category",Options:[]string{"low","medium","high"}}},
		"nonprofit":{{Name:"donation_id",DataType:"id"},{Name:"amount",DataType:"float",Min:5,Max:50000},{Name:"donation_date",DataType:"date"},{Name:"campaign",DataType:"category",Options:[]string{"annual","emergency","capital"}},{Name:"channel",DataType:"category",Options:[]string{"online","mail","event","phone"}},{Name:"is_recurring",DataType:"bool",TrueRate:0.3},{Name:"donor_type",DataType:"category",Options:[]string{"individual","corporate","foundation"}},{Name:"program_area",DataType:"category",Options:[]string{"education","health","environment","poverty"}},{Name:"tax_deductible",DataType:"bool",TrueRate:0.85},{Name:"region",DataType:"category",Options:[]string{"North","South","East","West"}}},
		"cybersecurity":{{Name:"event_id",DataType:"id"},{Name:"event_type",DataType:"category",Options:[]string{"intrusion","malware","phishing","ddos","ransomware"}},{Name:"severity",DataType:"category",Options:[]string{"low","medium","high","critical"}},{Name:"timestamp",DataType:"date"},{Name:"target_system",DataType:"category",Options:[]string{"web_server","database","email","firewall"}},{Name:"response_time_min",DataType:"float",Min:1,Max:480},{Name:"is_resolved",DataType:"bool",TrueRate:0.75},{Name:"data_lost_gb",DataType:"float",Min:0,Max:100},{Name:"cost_impact",DataType:"float",Min:0,Max:1000000},{Name:"source_country",DataType:"category",Options:[]string{"CN","RU","US","KP","IR","unknown"}}},
		"iot":{{Name:"device_id",DataType:"id"},{Name:"sensor_type",DataType:"category",Options:[]string{"temperature","humidity","pressure","motion","light"}},{Name:"reading_value",DataType:"float",Min:0,Max:100},{Name:"timestamp",DataType:"date"},{Name:"battery_level",DataType:"float",Min:0,Max:100},{Name:"signal_strength",DataType:"float",Min:-100,Max:0},{Name:"location",DataType:"category",Options:[]string{"floor_1","floor_2","floor_3","outdoor"}},{Name:"is_anomaly",DataType:"bool",TrueRate:0.05},{Name:"firmware_version",DataType:"category",Options:[]string{"v1.0","v1.1","v2.0"}},{Name:"uptime_hours",DataType:"int",Min:0,Max:8760}},
		"gaming":{{Name:"player_id",DataType:"id"},{Name:"level",DataType:"int",Min:1,Max:100},{Name:"total_playtime_hours",DataType:"float",Min:0.5,Max:5000},{Name:"in_app_purchases",DataType:"float",Min:0,Max:1000},{Name:"daily_sessions",DataType:"int",Min:0,Max:20},{Name:"achievements_unlocked",DataType:"int",Min:0,Max:200},{Name:"platform",DataType:"category",Options:[]string{"PC","PlayStation","Xbox","mobile"}},{Name:"is_premium",DataType:"bool",TrueRate:0.15},{Name:"retention_day30",DataType:"bool",TrueRate:0.25},{Name:"genre",DataType:"category",Options:[]string{"action","rpg","strategy","sports"}}},
		"crypto":{{Name:"transaction_id",DataType:"id"},{Name:"coin",DataType:"category",Options:[]string{"BTC","ETH","SOL","ADA","DOT"}},{Name:"amount",DataType:"float",Min:0.001,Max:100},{Name:"price_usd",DataType:"float",Min:0.1,Max:100000},{Name:"transaction_date",DataType:"date"},{Name:"transaction_type",DataType:"category",Options:[]string{"buy","sell","transfer","stake"}},{Name:"gas_fee",DataType:"float",Min:0.01,Max:50},{Name:"exchange",DataType:"category",Options:[]string{"Binance","Coinbase","Kraken"}},{Name:"is_whale",DataType:"bool",TrueRate:0.02},{Name:"wallet_type",DataType:"category",Options:[]string{"hot","cold","custodial"}}},
		"saas":{{Name:"account_id",DataType:"id"},{Name:"plan",DataType:"category",Options:[]string{"free","starter","pro","enterprise"}},{Name:"mrr",DataType:"float",Min:0,Max:10000},{Name:"signup_date",DataType:"date"},{Name:"active_users",DataType:"int",Min:1,Max:500},{Name:"feature_usage_pct",DataType:"float",Min:0,Max:1},{Name:"support_tickets",DataType:"int",Min:0,Max:20},{Name:"nps_score",DataType:"int",Min:-100,Max:100},{Name:"churned",DataType:"bool",TrueRate:0.08},{Name:"industry",DataType:"category",Options:[]string{"tech","finance","healthcare","retail"}}},
		"customer-service":{{Name:"ticket_id",DataType:"id"},{Name:"category",DataType:"category",Options:[]string{"billing","technical","account","shipping","returns"}},{Name:"priority",DataType:"category",Options:[]string{"low","medium","high","urgent"}},{Name:"created_date",DataType:"date"},{Name:"resolution_time_hours",DataType:"float",Min:0.1,Max:168},{Name:"channel",DataType:"category",Options:[]string{"email","phone","chat","social"}},{Name:"satisfaction_score",DataType:"float",Min:1,Max:5},{Name:"is_escalated",DataType:"bool",TrueRate:0.15},{Name:"agent_id",DataType:"id"},{Name:"status",DataType:"category",Options:[]string{"open","in_progress","resolved","closed"}}},
		"sales":{{Name:"deal_id",DataType:"id"},{Name:"deal_value",DataType:"float",Min:1000,Max:500000},{Name:"stage",DataType:"category",Options:[]string{"prospect","qualified","proposal","negotiation","closed_won","closed_lost"}},{Name:"created_date",DataType:"date"},{Name:"sales_rep",DataType:"name"},{Name:"industry",DataType:"category",Options:[]string{"tech","finance","healthcare","retail","manufacturing"}},{Name:"deal_size",DataType:"category",Options:[]string{"small","medium","large","enterprise"}},{Name:"is_won",DataType:"bool",TrueRate:0.35},{Name:"close_date",DataType:"date"},{Name:"source",DataType:"category",Options:[]string{"inbound","outbound","referral"}}},
		"advertising":{{Name:"ad_id",DataType:"id"},{Name:"platform",DataType:"category",Options:[]string{"Google","Meta","TikTok","LinkedIn","Twitter"}},{Name:"ad_format",DataType:"category",Options:[]string{"display","video","search","native"}},{Name:"impressions",DataType:"int",Min:1000,Max:1000000},{Name:"clicks",DataType:"int",Min:10,Max:50000},{Name:"spend",DataType:"float",Min:10,Max:50000},{Name:"conversions",DataType:"int",Min:0,Max:1000},{Name:"cpc",DataType:"float",Min:0.1,Max:10},{Name:"roas",DataType:"float",Min:0,Max:10},{Name:"campaign",DataType:"category",Options:[]string{"Brand","Product","Retargeting","Seasonal"}}},
		"social-media":{{Name:"post_id",DataType:"id"},{Name:"platform",DataType:"category",Options:[]string{"Instagram","TikTok","Twitter","LinkedIn","YouTube"}},{Name:"post_type",DataType:"category",Options:[]string{"image","video","text","story","reel"}},{Name:"publish_date",DataType:"date"},{Name:"followers",DataType:"int",Min:100,Max:1000000},{Name:"likes",DataType:"int",Min:0,Max:100000},{Name:"comments",DataType:"int",Min:0,Max:5000},{Name:"shares",DataType:"int",Min:0,Max:10000},{Name:"engagement_rate",DataType:"float",Min:0,Max:0.2},{Name:"sentiment_score",DataType:"float",Min:-1,Max:1}},
		"weather":{{Name:"station_id",DataType:"id"},{Name:"date",DataType:"date"},{Name:"temperature_c",DataType:"float",Min:-30,Max:50},{Name:"humidity_pct",DataType:"float",Min:0,Max:100},{Name:"wind_speed_kmh",DataType:"float",Min:0,Max:150},{Name:"precipitation_mm",DataType:"float",Min:0,Max:100},{Name:"pressure_hpa",DataType:"float",Min:970,Max:1050},{Name:"condition",DataType:"category",Options:[]string{"sunny","cloudy","rainy","snowy","stormy"}},{Name:"uv_index",DataType:"int",Min:0,Max:11},{Name:"visibility_km",DataType:"float",Min:0.1,Max:50}},
		"demographics":{{Name:"person_id",DataType:"id"},{Name:"age",DataType:"int",Min:0,Max:100},{Name:"gender",DataType:"category",Options:[]string{"Male","Female","Non-binary"}},{Name:"income",DataType:"float",Min:10000,Max:300000},{Name:"education",DataType:"category",Options:[]string{"high_school","bachelors","masters","phd"}},{Name:"marital_status",DataType:"category",Options:[]string{"single","married","divorced"}},{Name:"city",DataType:"category",Options:[]string{"New York","LA","Chicago","Houston","Phoenix"}},{Name:"household_size",DataType:"int",Min:1,Max:8},{Name:"is_employed",DataType:"bool",TrueRate:0.65},{Name:"region",DataType:"category",Options:[]string{"North","South","East","West"}}},
		"survey":{{Name:"response_id",DataType:"id"},{Name:"survey_date",DataType:"date"},{Name:"q1_satisfaction",DataType:"int",Min:1,Max:5},{Name:"q2_recommend",DataType:"int",Min:0,Max:10},{Name:"q3_ease_of_use",DataType:"int",Min:1,Max:5},{Name:"q4_value",DataType:"int",Min:1,Max:5},{Name:"overall_score",DataType:"float",Min:1,Max:10},{Name:"age_group",DataType:"category",Options:[]string{"18-24","25-34","35-44","45-54","55+"}},{Name:"completed",DataType:"bool",TrueRate:0.7},{Name:"channel",DataType:"category",Options:[]string{"email","web","app","phone"}}},
		"sentiment":{{Name:"text_id",DataType:"id"},{Name:"source",DataType:"category",Options:[]string{"twitter","review","survey","email","chat"}},{Name:"sentiment_score",DataType:"float",Min:-1,Max:1},{Name:"sentiment_label",DataType:"category",Options:[]string{"positive","neutral","negative"}},{Name:"confidence",DataType:"float",Min:0.5,Max:1},{Name:"text_length",DataType:"int",Min:10,Max:500},{Name:"date",DataType:"date"},{Name:"topic",DataType:"category",Options:[]string{"product","service","pricing","support"}},{Name:"language",DataType:"category",Options:[]string{"en","es","fr","de","ja"}},{Name:"has_keywords",DataType:"bool",TrueRate:0.6}},
		"fraud":{{Name:"transaction_id",DataType:"id"},{Name:"amount",DataType:"float",Min:1,Max:25000},{Name:"transaction_time",DataType:"date"},{Name:"merchant_category",DataType:"category",Options:[]string{"retail","online","atm","gas","restaurant"}},{Name:"distance_from_home",DataType:"float",Min:0,Max:500},{Name:"is_foreign",DataType:"bool",TrueRate:0.1},{Name:"is_high_value",DataType:"bool",TrueRate:0.15},{Name:"device_type",DataType:"category",Options:[]string{"mobile","desktop","pos","atm"}},{Name:"velocity_24h",DataType:"int",Min:1,Max:30},{Name:"is_fraud",DataType:"bool",TrueRate:0.03}},
		"churn":{{Name:"customer_id",DataType:"id"},{Name:"tenure_months",DataType:"int",Min:1,Max:72},{Name:"monthly_spend",DataType:"float",Min:10,Max:500},{Name:"total_spend",DataType:"float",Min:10,Max:36000},{Name:"support_calls",DataType:"int",Min:0,Max:15},{Name:"last_activity_days",DataType:"int",Min:0,Max:90},{Name:"plan_type",DataType:"category",Options:[]string{"basic","standard","premium"}},{Name:"contract_type",DataType:"category",Options:[]string{"monthly","annual"}},{Name:"satisfaction_score",DataType:"float",Min:1,Max:5},{Name:"churned",DataType:"bool",TrueRate:0.15}},
		"recommendation":{{Name:"user_id",DataType:"id"},{Name:"item_id",DataType:"id"},{Name:"rating",DataType:"float",Min:1,Max:5},{Name:"timestamp",DataType:"date"},{Name:"item_category",DataType:"category",Options:[]string{"electronics","books","movies","music","games"}},{Name:"interaction_type",DataType:"category",Options:[]string{"view","click","purchase","wishlist"}},{Name:"session_duration_min",DataType:"float",Min:0.5,Max:60},{Name:"device",DataType:"category",Options:[]string{"mobile","desktop","tablet"}},{Name:"is_purchased",DataType:"bool",TrueRate:0.2},{Name:"age_group",DataType:"category",Options:[]string{"18-24","25-34","35-44","45-54","55+"}}},
		"time-series":{{Name:"timestamp",DataType:"date"},{Name:"value",DataType:"float",Min:0,Max:1000},{Name:"metric_name",DataType:"category",Options:[]string{"cpu","memory","requests","latency","errors"}},{Name:"host",DataType:"category",Options:[]string{"server_1","server_2","server_3","server_4"}},{Name:"moving_avg",DataType:"float",Min:0,Max:1000},{Name:"std_dev",DataType:"float",Min:0,Max:100},{Name:"is_anomaly",DataType:"bool",TrueRate:0.03},{Name:"trend",DataType:"category",Options:[]string{"increasing","decreasing","stable"}},{Name:"seasonality",DataType:"float",Min:-50,Max:50},{Name:"forecast",DataType:"float",Min:0,Max:1000}},
		"nlp":{{Name:"document_id",DataType:"id"},{Name:"text_length",DataType:"int",Min:10,Max:5000},{Name:"language",DataType:"category",Options:[]string{"en","es","fr","de","zh","ja"}},{Name:"category",DataType:"category",Options:[]string{"news","review","social","academic"}},{Name:"sentiment",DataType:"float",Min:-1,Max:1},{Name:"readability_score",DataType:"float",Min:0,Max:100},{Name:"named_entities",DataType:"int",Min:0,Max:50},{Name:"word_count",DataType:"int",Min:5,Max:1000},{Name:"is_spam",DataType:"bool",TrueRate:0.1},{Name:"topic_cluster",DataType:"category",Options:[]string{"cluster_0","cluster_1","cluster_2","cluster_3"}}},
		"image":{{Name:"image_id",DataType:"id"},{Name:"width",DataType:"int",Min:64,Max:4096},{Name:"height",DataType:"int",Min:64,Max:4096},{Name:"file_size_kb",DataType:"int",Min:10,Max:10000},{Name:"format",DataType:"category",Options:[]string{"jpg","png","webp","tiff"}},{Name:"label",DataType:"category",Options:[]string{"cat","dog","car","person","building","nature"}},{Name:"confidence",DataType:"float",Min:0.5,Max:1},{Name:"brightness",DataType:"float",Min:0,Max:255},{Name:"has_faces",DataType:"bool",TrueRate:0.4},{Name:"quality_score",DataType:"float",Min:0,Max:1}},
	}
	if d, ok := all[v]; ok { return d }
	return all["finance"]
}

func csvStats(p string) (int, string) {
	f, err := os.Open(p)
	if err != nil { return 0, "" }
	defer f.Close()
	r := csv.NewReader(f)
	recs, err := r.ReadAll()
	if err != nil || len(recs) == 0 { return 0, "" }
	return len(recs) - 1, strings.Join(recs[0], ",")
}

func DownloadFileHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet { http.Error(w, "Method not allowed", http.StatusMethodNotAllowed); return }
	userID := r.Header.Get("X-User-ID")
	if userID == "" { http.Error(w, "Unauthorized", http.StatusUnauthorized); return }
	fileID := strings.TrimPrefix(r.URL.Path, "/api/download/")
	if fileID == "" { http.Error(w, "File ID required", http.StatusBadRequest); return }
	var file UploadedFile
	if err := DB.Where("id = ? AND user_id = ?", fileID, userID).First(&file).Error; err != nil { http.Error(w, "File not found", http.StatusNotFound); return }
	// Try cloud storage first, fallback to local
	var f io.ReadCloser
	f, err := services.DefaultStorage.Download(file.Path)
	if err != nil {
		// Fallback: try local file
		f, err = os.Open(file.Path)
		if err != nil { http.Error(w, "File not found", http.StatusNotFound); return }
	}
	defer f.Close()
	w.Header().Set("Content-Disposition", fmt.Sprintf(`attachment; filename="%s"`, file.Filename))
	ext := strings.ToLower(filepath.Ext(file.Filename))
	ctype := "application/octet-stream"
	if ext == ".csv" { ctype = "text/csv" }
	if ext == ".json" || ext == ".jsonl" { ctype = "application/json" }
	if ext == ".xlsx" { ctype = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" }
	w.Header().Set("Content-Type", ctype)
	if file.Size > 0 { w.Header().Set("Content-Length", strconv.FormatInt(file.Size, 10)) }
	io.Copy(w, f)
}

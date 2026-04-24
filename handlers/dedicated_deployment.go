package handlers

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

type DedicatedBundle struct {
	ID               string    `gorm:"primaryKey" json:"id"`
	UserID           string    `gorm:"index" json:"user_id"`
	ModelID          string    `gorm:"index" json:"model_id"`
	ModelName        string    `json:"model_name"`
	ModelVersion     int       `json:"model_version"`
	CheckpointPath   string    `json:"checkpoint_path"`
	EncryptedPath    string    `json:"encrypted_path"`
	EncryptedSize    int64     `json:"encrypted_size"`
	KeyID            string    `gorm:"index" json:"key_id"`
	CiphertextSHA256 string    `json:"ciphertext_sha256"`
	DeploymentTarget string    `json:"deployment_target"`
	Status           string    `gorm:"default:created" json:"status"`
	DownloadCount    int       `gorm:"default:0" json:"download_count"`
	LastDownloadedAt time.Time `json:"last_downloaded_at"`
	RevokedAt        time.Time `json:"revoked_at"`
	CreatedAt        time.Time `json:"created_at"`
	UpdatedAt        time.Time `json:"updated_at"`
}

type DedicatedBundleKey struct {
	ID            string    `gorm:"primaryKey" json:"id"`
	BundleID      string    `gorm:"index" json:"bundle_id"`
	UserID        string    `gorm:"index" json:"user_id"`
	KeyCiphertext string    `json:"-"`
	KeyFingerprint string   `json:"key_fingerprint"`
	RotatedFromID string    `json:"rotated_from_id,omitempty"`
	Active        bool      `gorm:"default:true" json:"active"`
	CreatedAt     time.Time `json:"created_at"`
	ExpiresAt     time.Time `json:"expires_at"`
}

type DedicatedBundleAudit struct {
	ID         string    `gorm:"primaryKey" json:"id"`
	BundleID   string    `gorm:"index" json:"bundle_id"`
	UserID     string    `gorm:"index" json:"user_id"`
	Action     string    `json:"action"`
	IPAddress  string    `json:"ip_address"`
	UserAgent  string    `json:"user_agent"`
	Details    string    `json:"details"`
	CreatedAt  time.Time `json:"created_at"`
}

type DedicatedDownloadToken struct {
	Token     string    `gorm:"primaryKey" json:"token"`
	BundleID  string    `gorm:"index" json:"bundle_id"`
	UserID    string    `gorm:"index" json:"user_id"`
	ExpiresAt time.Time `json:"expires_at"`
	UsedAt    time.Time `json:"used_at"`
	CreatedAt time.Time `json:"created_at"`
}

func getMasterKey() ([]byte, error) {
	raw := os.Getenv("DEDICATED_MASTER_KEY")
	if raw == "" {
		return nil, fmt.Errorf("DEDICATED_MASTER_KEY not set")
	}
	key, err := hex.DecodeString(raw)
	if err != nil {
		return nil, fmt.Errorf("DEDICATED_MASTER_KEY must be 64-char hex: %w", err)
	}
	if len(key) != 32 {
		return nil, fmt.Errorf("DEDICATED_MASTER_KEY must be 32 bytes (256 bits)")
	}
	return key, nil
}

func generateBundleKey() ([]byte, string, error) {
	key := make([]byte, 32)
	if _, err := rand.Read(key); err != nil {
		return nil, "", err
	}
	h := sha256.Sum256(key)
	return key, hex.EncodeToString(h[:8]), nil
}

func encryptBundleKey(plainKey []byte) (string, error) {
	master, err := getMasterKey()
	if err != nil {
		return "", err
	}
	block, err := aes.NewCipher(master)
	if err != nil {
		return "", err
	}
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return "", err
	}
	nonce := make([]byte, gcm.NonceSize())
	if _, err := rand.Read(nonce); err != nil {
		return "", err
	}
	ct := gcm.Seal(nonce, nonce, plainKey, nil)
	return base64.StdEncoding.EncodeToString(ct), nil
}

func decryptBundleKey(ciphertext string) ([]byte, error) {
	master, err := getMasterKey()
	if err != nil {
		return nil, err
	}
	data, err := base64.StdEncoding.DecodeString(ciphertext)
	if err != nil {
		return nil, err
	}
	block, err := aes.NewCipher(master)
	if err != nil {
		return nil, err
	}
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}
	ns := gcm.NonceSize()
	if len(data) < ns {
		return nil, fmt.Errorf("ciphertext too short")
	}
	nonce, ct := data[:ns], data[ns:]
	return gcm.Open(nil, nonce, ct, nil)
}

func encryptCheckpointAES256GCM(plainPath, cipherPath string, key []byte) (int64, string, error) {
	plain, err := os.ReadFile(plainPath)
	if err != nil {
		return 0, "", fmt.Errorf("read checkpoint: %w", err)
	}
	block, err := aes.NewCipher(key)
	if err != nil {
		return 0, "", err
	}
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return 0, "", err
	}
	nonce := make([]byte, gcm.NonceSize())
	if _, err := rand.Read(nonce); err != nil {
		return 0, "", err
	}
	ct := gcm.Seal(nonce, nonce, plain, nil)

	if err := os.MkdirAll(filepath.Dir(cipherPath), 0o755); err != nil {
		return 0, "", err
	}
	if err := os.WriteFile(cipherPath, ct, 0o600); err != nil {
		return 0, "", err
	}
	h := sha256.Sum256(ct)
	return int64(len(ct)), hex.EncodeToString(h[:]), nil
}

func logBundleAudit(bundleID, userID, action, ip, ua, details string) {
	entry := DedicatedBundleAudit{
		ID:        fmt.Sprintf("aud-%d-%s", time.Now().UnixNano(), bundleID[:min(8, len(bundleID))]),
		BundleID:  bundleID,
		UserID:    userID,
		Action:    action,
		IPAddress: ip,
		UserAgent: ua,
		Details:   details,
		CreatedAt: time.Now(),
	}
	DB.Create(&entry)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func clientIP(r *http.Request) string {
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		return strings.SplitN(xff, ",", 2)[0]
	}
	return r.RemoteAddr
}

func requireEnterprise(userID string) (*UserQuota, error) {
	quota, err := GetOrCreateQuota(userID)
	if err != nil {
		return nil, err
	}
	if quota.Plan != "enterprise" && !IsUnlimitedPlan(quota.Plan) {
		return nil, fmt.Errorf("dedicated deployment requires enterprise plan")
	}
	return quota, nil
}

func CreateDedicatedBundleHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}
	if _, err := requireEnterprise(userID); err != nil {
		http.Error(w, err.Error(), http.StatusForbidden)
		return
	}

	var req struct {
		ModelID          string `json:"model_id"`
		DeploymentTarget string `json:"deployment_target"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}
	if req.ModelID == "" {
		http.Error(w, "model_id required", http.StatusBadRequest)
		return
	}

	var model FineTunedModel
	if err := DB.Where("id = ? AND user_id = ?", req.ModelID, userID).First(&model).Error; err != nil {
		http.Error(w, "Model not found", http.StatusNotFound)
		return
	}
	if model.ModelPath == "" {
		http.Error(w, "Model checkpoint not available", http.StatusBadRequest)
		return
	}

	plainKey, fp, err := generateBundleKey()
	if err != nil {
		http.Error(w, "Key generation failed", http.StatusInternalServerError)
		return
	}
	encKey, err := encryptBundleKey(plainKey)
	if err != nil {
		log.Printf("[DEDICATED] key encrypt failed: %v", err)
		http.Error(w, "Key encryption failed", http.StatusInternalServerError)
		return
	}

	bundleID := fmt.Sprintf("bundle-%d-%s", time.Now().Unix(), fp[:8])
	keyID := fmt.Sprintf("key-%s", fp)
	encPath := filepath.Join("./dedicated_bundles", userID, bundleID+".enc")

	size, digest, err := encryptCheckpointAES256GCM(model.ModelPath, encPath, plainKey)
	if err != nil {
		log.Printf("[DEDICATED] encrypt checkpoint failed: %v", err)
		http.Error(w, "Encryption failed: "+err.Error(), http.StatusInternalServerError)
		return
	}

	bundle := DedicatedBundle{
		ID:               bundleID,
		UserID:           userID,
		ModelID:          model.ID,
		ModelName:        model.Name,
		ModelVersion:     model.Version,
		CheckpointPath:   model.ModelPath,
		EncryptedPath:    encPath,
		EncryptedSize:    size,
		KeyID:            keyID,
		CiphertextSHA256: digest,
		DeploymentTarget: req.DeploymentTarget,
		Status:           "ready",
		CreatedAt:        time.Now(),
		UpdatedAt:        time.Now(),
	}
	if err := DB.Create(&bundle).Error; err != nil {
		http.Error(w, "DB error", http.StatusInternalServerError)
		return
	}

	keyRow := DedicatedBundleKey{
		ID:             keyID,
		BundleID:       bundleID,
		UserID:         userID,
		KeyCiphertext:  encKey,
		KeyFingerprint: fp,
		Active:         true,
		CreatedAt:      time.Now(),
		ExpiresAt:      time.Now().AddDate(1, 0, 0),
	}
	DB.Create(&keyRow)

	logBundleAudit(bundleID, userID, "bundle_created", clientIP(r), r.UserAgent(),
		fmt.Sprintf("model=%s target=%s size=%d sha256=%s", model.Name, req.DeploymentTarget, size, digest))

	for i := range plainKey {
		plainKey[i] = 0
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(bundle)
}

func ListDedicatedBundlesHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}
	if _, err := requireEnterprise(userID); err != nil {
		http.Error(w, err.Error(), http.StatusForbidden)
		return
	}
	var bundles []DedicatedBundle
	DB.Where("user_id = ?", userID).Order("created_at DESC").Find(&bundles)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(bundles)
}

func IssueDownloadTokenHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}
	if _, err := requireEnterprise(userID); err != nil {
		http.Error(w, err.Error(), http.StatusForbidden)
		return
	}

	var req struct {
		BundleID string `json:"bundle_id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	var bundle DedicatedBundle
	if err := DB.Where("id = ? AND user_id = ?", req.BundleID, userID).First(&bundle).Error; err != nil {
		http.Error(w, "Bundle not found", http.StatusNotFound)
		return
	}
	if !bundle.RevokedAt.IsZero() {
		http.Error(w, "Bundle revoked", http.StatusGone)
		return
	}

	rawToken := make([]byte, 32)
	if _, err := rand.Read(rawToken); err != nil {
		http.Error(w, "Token gen failed", http.StatusInternalServerError)
		return
	}
	token := hex.EncodeToString(rawToken)

	dt := DedicatedDownloadToken{
		Token:     token,
		BundleID:  bundle.ID,
		UserID:    userID,
		ExpiresAt: time.Now().Add(15 * time.Minute),
		CreatedAt: time.Now(),
	}
	DB.Create(&dt)
	logBundleAudit(bundle.ID, userID, "download_token_issued", clientIP(r), r.UserAgent(), "expires_in=15min")

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"token":      token,
		"url":        "/api/dedicated/download/" + token,
		"expires_at": dt.ExpiresAt.Format(time.RFC3339),
	})
}

func DownloadDedicatedBundleHandler(w http.ResponseWriter, r *http.Request) {
	token := strings.TrimPrefix(r.URL.Path, "/api/dedicated/download/")
	if token == "" {
		http.Error(w, "Token required", http.StatusBadRequest)
		return
	}

	var dt DedicatedDownloadToken
	if err := DB.Where("token = ?", token).First(&dt).Error; err != nil {
		http.Error(w, "Invalid token", http.StatusNotFound)
		return
	}
	if !dt.UsedAt.IsZero() {
		http.Error(w, "Token already used", http.StatusGone)
		return
	}
	if time.Now().After(dt.ExpiresAt) {
		http.Error(w, "Token expired", http.StatusGone)
		return
	}

	var bundle DedicatedBundle
	if err := DB.Where("id = ?", dt.BundleID).First(&bundle).Error; err != nil {
		http.Error(w, "Bundle not found", http.StatusNotFound)
		return
	}
	if !bundle.RevokedAt.IsZero() {
		http.Error(w, "Bundle revoked", http.StatusGone)
		return
	}

	f, err := os.Open(bundle.EncryptedPath)
	if err != nil {
		log.Printf("[DEDICATED] open encrypted failed: %v", err)
		http.Error(w, "File not available", http.StatusInternalServerError)
		return
	}
	defer f.Close()

	filename := fmt.Sprintf("%s_v%d.enc", strings.ReplaceAll(bundle.ModelName, " ", "_"), bundle.ModelVersion)
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("Content-Disposition", fmt.Sprintf(`attachment; filename="%s"`, filename))
	w.Header().Set("Content-Length", fmt.Sprintf("%d", bundle.EncryptedSize))
	w.Header().Set("X-Bundle-SHA256", bundle.CiphertextSHA256)
	w.Header().Set("X-Bundle-Key-ID", bundle.KeyID)

	if _, err := io.Copy(w, f); err != nil {
		log.Printf("[DEDICATED] stream failed: %v", err)
		return
	}

	dt.UsedAt = time.Now()
	DB.Save(&dt)
	bundle.DownloadCount++
	bundle.LastDownloadedAt = time.Now()
	DB.Save(&bundle)
	logBundleAudit(bundle.ID, dt.UserID, "bundle_downloaded", clientIP(r), r.UserAgent(),
		fmt.Sprintf("token=%s download_count=%d", token[:12], bundle.DownloadCount))
}

func RevealBundleKeyHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}
	if _, err := requireEnterprise(userID); err != nil {
		http.Error(w, err.Error(), http.StatusForbidden)
		return
	}

	var req struct {
		BundleID string `json:"bundle_id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	var bundle DedicatedBundle
	if err := DB.Where("id = ? AND user_id = ?", req.BundleID, userID).First(&bundle).Error; err != nil {
		http.Error(w, "Bundle not found", http.StatusNotFound)
		return
	}

	var keyRow DedicatedBundleKey
	if err := DB.Where("id = ? AND active = ?", bundle.KeyID, true).First(&keyRow).Error; err != nil {
		http.Error(w, "Active key not found", http.StatusNotFound)
		return
	}

	plainKey, err := decryptBundleKey(keyRow.KeyCiphertext)
	if err != nil {
		log.Printf("[DEDICATED] key decrypt failed: %v", err)
		http.Error(w, "Key decryption failed", http.StatusInternalServerError)
		return
	}
	plainHex := hex.EncodeToString(plainKey)
	for i := range plainKey {
		plainKey[i] = 0
	}
	logBundleAudit(bundle.ID, userID, "key_revealed", clientIP(r), r.UserAgent(), "fingerprint="+keyRow.KeyFingerprint)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"key_hex":     plainHex,
		"fingerprint": keyRow.KeyFingerprint,
		"algorithm":   "AES-256-GCM",
		"key_id":      keyRow.ID,
		"warning":     "Transmit this key only over encrypted out-of-band channel. Never commit to repository or send over unencrypted email.",
	})
}

func RotateBundleKeyHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}
	if _, err := requireEnterprise(userID); err != nil {
		http.Error(w, err.Error(), http.StatusForbidden)
		return
	}

	var req struct {
		BundleID string `json:"bundle_id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	var bundle DedicatedBundle
	if err := DB.Where("id = ? AND user_id = ?", req.BundleID, userID).First(&bundle).Error; err != nil {
		http.Error(w, "Bundle not found", http.StatusNotFound)
		return
	}

	newKey, newFp, err := generateBundleKey()
	if err != nil {
		http.Error(w, "Key gen failed", http.StatusInternalServerError)
		return
	}
	newEncKey, err := encryptBundleKey(newKey)
	if err != nil {
		http.Error(w, "Key encryption failed", http.StatusInternalServerError)
		return
	}

	newEncPath := filepath.Join("./dedicated_bundles", userID, bundle.ID+"_rotated_"+newFp[:8]+".enc")
	size, digest, err := encryptCheckpointAES256GCM(bundle.CheckpointPath, newEncPath, newKey)
	if err != nil {
		http.Error(w, "Re-encrypt failed: "+err.Error(), http.StatusInternalServerError)
		return
	}

	DB.Model(&DedicatedBundleKey{}).Where("bundle_id = ?", bundle.ID).Update("active", false)

	oldKeyID := bundle.KeyID
	newKeyID := fmt.Sprintf("key-%s", newFp)

	keyRow := DedicatedBundleKey{
		ID:             newKeyID,
		BundleID:       bundle.ID,
		UserID:         userID,
		KeyCiphertext:  newEncKey,
		KeyFingerprint: newFp,
		RotatedFromID:  oldKeyID,
		Active:         true,
		CreatedAt:      time.Now(),
		ExpiresAt:      time.Now().AddDate(1, 0, 0),
	}
	DB.Create(&keyRow)

	oldEncPath := bundle.EncryptedPath
	bundle.KeyID = newKeyID
	bundle.EncryptedPath = newEncPath
	bundle.EncryptedSize = size
	bundle.CiphertextSHA256 = digest
	bundle.UpdatedAt = time.Now()
	DB.Save(&bundle)
	_ = os.Remove(oldEncPath)

	for i := range newKey {
		newKey[i] = 0
	}
	logBundleAudit(bundle.ID, userID, "key_rotated", clientIP(r), r.UserAgent(),
		fmt.Sprintf("old=%s new=%s", oldKeyID, newKeyID))

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(bundle)
}

func RevokeBundleHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}
	if _, err := requireEnterprise(userID); err != nil {
		http.Error(w, err.Error(), http.StatusForbidden)
		return
	}

	var req struct {
		BundleID string `json:"bundle_id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	var bundle DedicatedBundle
	if err := DB.Where("id = ? AND user_id = ?", req.BundleID, userID).First(&bundle).Error; err != nil {
		http.Error(w, "Bundle not found", http.StatusNotFound)
		return
	}

	bundle.RevokedAt = time.Now()
	bundle.Status = "revoked"
	DB.Save(&bundle)
	DB.Model(&DedicatedBundleKey{}).Where("bundle_id = ?", bundle.ID).Update("active", false)
	DB.Model(&DedicatedDownloadToken{}).Where("bundle_id = ? AND used_at IS NULL", bundle.ID).
		Update("expires_at", time.Now())
	logBundleAudit(bundle.ID, userID, "bundle_revoked", clientIP(r), r.UserAgent(), "")

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "revoked"})
}

func ListBundleAuditHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}
	if _, err := requireEnterprise(userID); err != nil {
		http.Error(w, err.Error(), http.StatusForbidden)
		return
	}

	bundleID := r.URL.Query().Get("bundle_id")
	q := DB.Where("user_id = ?", userID).Order("created_at DESC").Limit(500)
	if bundleID != "" {
		q = q.Where("bundle_id = ?", bundleID)
	}
	var logs []DedicatedBundleAudit
	q.Find(&logs)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(logs)
}

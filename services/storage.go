package services

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"cloud.google.com/go/storage"
	"google.golang.org/api/iterator"
)

// StorageBackend - cloud agnostic storage interface
// Local → GCS → S3 geçişi sadece env değişikliği
type StorageBackend interface {
	Upload(key string, r io.Reader) error
	Download(key string) (io.ReadCloser, error)
	Delete(key string) error
	Exists(key string) (bool, error)
	URL(key string) string
	SignedURL(key string, expiry time.Duration) (string, error)
	Size(key string) (int64, error)
	List(prefix string) ([]string, error)
}

var DefaultStorage StorageBackend

func InitStorage() {
	backend := os.Getenv("STORAGE_BACKEND")
	if backend == "" {
		backend = "local"
	}
	switch backend {
	case "gcs":
		bucket := os.Getenv("GCS_BUCKET")
		if bucket == "" {
			bucket = "schemalabs-prod-us-central1"
		}
		gcs, err := NewGCSStorage(bucket)
		if err != nil {
			fmt.Printf("[STORAGE] GCS init failed: %v, falling back to local\n", err)
			DefaultStorage = &LocalStorage{BasePath: "./uploads"}
			return
		}
		DefaultStorage = gcs
	case "s3":
		bucket := os.Getenv("S3_BUCKET")
		region := os.Getenv("S3_REGION")
		if region == "" {
			region = "us-east-1"
		}
		DefaultStorage = &S3Storage{Bucket: bucket, Region: region}
	default:
		basePath := os.Getenv("STORAGE_LOCAL_PATH")
		if basePath == "" {
			basePath = "./uploads"
		}
		DefaultStorage = &LocalStorage{BasePath: basePath}
	}
	fmt.Printf("[STORAGE] Backend: %s\n", backend)
}

// ============================================================
// USER-SCOPED KEY HELPERS
// ============================================================

func UserKey(userID, category, filename string) string {
	return fmt.Sprintf("users/%s/%s/%s", userID, category, sanitizeForStorage(filename))
}

func SharedKey(category, filename string) string {
	return fmt.Sprintf("shared/%s/%s", category, sanitizeForStorage(filename))
}

func SystemKey(category, filename string) string {
	return fmt.Sprintf("system/%s/%s", category, sanitizeForStorage(filename))
}

func UserStorageSize(userID string) (int64, error) {
	prefix := fmt.Sprintf("users/%s/", userID)
	keys, err := DefaultStorage.List(prefix)
	if err != nil {
		return 0, err
	}
	var total int64
	for _, key := range keys {
		size, err := DefaultStorage.Size(key)
		if err != nil {
			continue
		}
		total += size
	}
	return total, nil
}

func StorageKey(fileID, filename string) string {
	return fileID + "_" + sanitizeForStorage(filename)
}

func sanitizeForStorage(name string) string {
	name = filepath.Base(name)
	replacer := strings.NewReplacer(
		" ", "_", "..", "", "/", "", "\\", "",
		"'", "", "\"", "", "#", "", "&", "",
	)
	return replacer.Replace(name)
}

// ============================================================
// LOCAL STORAGE
// ============================================================

type LocalStorage struct {
	BasePath string
}

func (l *LocalStorage) Upload(key string, r io.Reader) error {
	path := filepath.Join(l.BasePath, key)
	os.MkdirAll(filepath.Dir(path), 0755)
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = io.Copy(f, r)
	return err
}

func (l *LocalStorage) Download(key string) (io.ReadCloser, error) {
	return os.Open(filepath.Join(l.BasePath, key))
}

func (l *LocalStorage) Delete(key string) error {
	return os.Remove(filepath.Join(l.BasePath, key))
}

func (l *LocalStorage) Exists(key string) (bool, error) {
	_, err := os.Stat(filepath.Join(l.BasePath, key))
	if os.IsNotExist(err) {
		return false, nil
	}
	return err == nil, err
}

func (l *LocalStorage) URL(key string) string {
	return filepath.Join(l.BasePath, key)
}

func (l *LocalStorage) SignedURL(key string, expiry time.Duration) (string, error) {
	return filepath.Join(l.BasePath, key), nil
}

func (l *LocalStorage) Size(key string) (int64, error) {
	info, err := os.Stat(filepath.Join(l.BasePath, key))
	if err != nil {
		return 0, err
	}
	return info.Size(), nil
}

func (l *LocalStorage) List(prefix string) ([]string, error) {
	var keys []string
	root := filepath.Join(l.BasePath, prefix)
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		if !info.IsDir() {
			rel, _ := filepath.Rel(l.BasePath, path)
			keys = append(keys, rel)
		}
		return nil
	})
	return keys, err
}

// ============================================================
// GCS STORAGE — PRODUCTION
// ============================================================

type GCSStorage struct {
	Bucket string
	client *storage.Client
}

func NewGCSStorage(bucket string) (*GCSStorage, error) {
	ctx := context.Background()
	client, err := storage.NewClient(ctx)
	if err != nil {
		return nil, fmt.Errorf("gcs client init: %w", err)
	}
	_, err = client.Bucket(bucket).Attrs(ctx)
	if err != nil {
		client.Close()
		return nil, fmt.Errorf("gcs bucket %s not accessible: %w", bucket, err)
	}
	fmt.Printf("[STORAGE] GCS connected: gs://%s/\n", bucket)
	return &GCSStorage{Bucket: bucket, client: client}, nil
}

func (g *GCSStorage) Upload(key string, r io.Reader) error {
	ctx := context.Background()
	w := g.client.Bucket(g.Bucket).Object(key).NewWriter(ctx)
	w.ChunkSize = 8 * 1024 * 1024
	if _, err := io.Copy(w, r); err != nil {
		w.Close()
		return fmt.Errorf("gcs upload %s: %w", key, err)
	}
	if err := w.Close(); err != nil {
		return fmt.Errorf("gcs upload close %s: %w", key, err)
	}
	return nil
}

func (g *GCSStorage) Download(key string) (io.ReadCloser, error) {
	ctx := context.Background()
	r, err := g.client.Bucket(g.Bucket).Object(key).NewReader(ctx)
	if err != nil {
		return nil, fmt.Errorf("gcs download %s: %w", key, err)
	}
	return r, nil
}

func (g *GCSStorage) Delete(key string) error {
	ctx := context.Background()
	if err := g.client.Bucket(g.Bucket).Object(key).Delete(ctx); err != nil {
		return fmt.Errorf("gcs delete %s: %w", key, err)
	}
	return nil
}

func (g *GCSStorage) Exists(key string) (bool, error) {
	ctx := context.Background()
	_, err := g.client.Bucket(g.Bucket).Object(key).Attrs(ctx)
	if err == storage.ErrObjectNotExist {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("gcs exists %s: %w", key, err)
	}
	return true, nil
}

func (g *GCSStorage) URL(key string) string {
	return fmt.Sprintf("gs://%s/%s", g.Bucket, key)
}

func (g *GCSStorage) SignedURL(key string, expiry time.Duration) (string, error) {
	url, err := g.client.Bucket(g.Bucket).SignedURL(key, &storage.SignedURLOptions{
		Method:  "GET",
		Expires: time.Now().Add(expiry),
	})
	if err != nil {
		return "", fmt.Errorf("gcs signed url %s: %w", key, err)
	}
	return url, nil
}

func (g *GCSStorage) Size(key string) (int64, error) {
	ctx := context.Background()
	attrs, err := g.client.Bucket(g.Bucket).Object(key).Attrs(ctx)
	if err != nil {
		return 0, fmt.Errorf("gcs size %s: %w", key, err)
	}
	return attrs.Size, nil
}

func (g *GCSStorage) List(prefix string) ([]string, error) {
	ctx := context.Background()
	it := g.client.Bucket(g.Bucket).Objects(ctx, &storage.Query{Prefix: prefix})
	var keys []string
	for {
		attrs, err := it.Next()
		if err == iterator.Done {
			break
		}
		if err != nil {
			return keys, fmt.Errorf("gcs list %s: %w", prefix, err)
		}
		if strings.HasSuffix(attrs.Name, ".keep") {
			continue
		}
		keys = append(keys, attrs.Name)
	}
	return keys, nil
}

// ============================================================
// S3 STORAGE — AWS geçişinde aktif olacak
// ============================================================

type S3Storage struct {
	Bucket string
	Region string
}

func (s *S3Storage) Upload(key string, r io.Reader) error {
	return fmt.Errorf("S3 not implemented yet — set STORAGE_BACKEND=gcs or local")
}
func (s *S3Storage) Download(key string) (io.ReadCloser, error) {
	return nil, fmt.Errorf("S3 not implemented yet")
}
func (s *S3Storage) Delete(key string) error { return fmt.Errorf("S3 not implemented yet") }
func (s *S3Storage) Exists(key string) (bool, error) { return false, fmt.Errorf("S3 not implemented yet") }
func (s *S3Storage) URL(key string) string {
	return fmt.Sprintf("https://%s.s3.%s.amazonaws.com/%s", s.Bucket, s.Region, key)
}
func (s *S3Storage) SignedURL(key string, expiry time.Duration) (string, error) {
	return "", fmt.Errorf("S3 signed URL not implemented yet")
}
func (s *S3Storage) Size(key string) (int64, error) { return 0, fmt.Errorf("S3 not implemented yet") }
func (s *S3Storage) List(prefix string) ([]string, error) { return nil, fmt.Errorf("S3 not implemented yet") }

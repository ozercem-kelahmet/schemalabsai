package services

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// StorageBackend - cloud agnostic storage interface
// Local → GCS → S3 geçişi sadece env değişikliği
type StorageBackend interface {
	Upload(key string, r io.Reader) error
	Download(key string) (io.ReadCloser, error)
	Delete(key string) error
	Exists(key string) (bool, error)
	URL(key string) string
	Size(key string) (int64, error)
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
		DefaultStorage = &GCSStorage{Bucket: bucket}
	case "s3":
		bucket := os.Getenv("S3_BUCKET")
		region := os.Getenv("S3_REGION")
		DefaultStorage = &S3Storage{Bucket: bucket, Region: region}
	default:
		DefaultStorage = &LocalStorage{BasePath: "./uploads"}
	}
	fmt.Printf("[STORAGE] Backend: %s\n", backend)
}

// LocalStorage - şu an kullanılan
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

func (l *LocalStorage) Size(key string) (int64, error) {
	info, err := os.Stat(filepath.Join(l.BasePath, key))
	if err != nil {
		return 0, err
	}
	return info.Size(), nil
}

// GCSStorage - GCP'ye geçince aktif olur
type GCSStorage struct {
	Bucket string
}

func (g *GCSStorage) Upload(key string, r io.Reader) error {
	// TODO: GCS client implement
	return fmt.Errorf("GCS not implemented yet")
}
func (g *GCSStorage) Download(key string) (io.ReadCloser, error) {
	return nil, fmt.Errorf("GCS not implemented yet")
}
func (g *GCSStorage) Delete(key string) error { return fmt.Errorf("GCS not implemented yet") }
func (g *GCSStorage) Exists(key string) (bool, error) { return false, nil }
func (g *GCSStorage) URL(key string) string {
	return fmt.Sprintf("gs://%s/%s", g.Bucket, key)
}
func (g *GCSStorage) Size(key string) (int64, error) { return 0, nil }

// S3Storage - AWS'ye geçince aktif olur
type S3Storage struct {
	Bucket string
	Region string
}

func (s *S3Storage) Upload(key string, r io.Reader) error {
	return fmt.Errorf("S3 not implemented yet")
}
func (s *S3Storage) Download(key string) (io.ReadCloser, error) {
	return nil, fmt.Errorf("S3 not implemented yet")
}
func (s *S3Storage) Delete(key string) error { return fmt.Errorf("S3 not implemented yet") }
func (s *S3Storage) Exists(key string) (bool, error) { return false, nil }
func (s *S3Storage) URL(key string) string {
	return fmt.Sprintf("https://%s.s3.%s.amazonaws.com/%s", s.Bucket, s.Region, key)
}
func (s *S3Storage) Size(key string) (int64, error) { return 0, nil }

// StorageKey - dosya key'i oluştur
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

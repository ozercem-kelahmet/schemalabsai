package main

import (
	"fmt"
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/exec"
	"schemalabsai/handlers"
	"strings"
	"time"

	"github.com/joho/godotenv"
)

func enableCORS(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "http://localhost:8080")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		w.Header().Set("Access-Control-Allow-Credentials", "true")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		next(w, r)
	}
}

func killPort(port string) {
	exec.Command("sh", "-c", "lsof -ti:"+port+" | xargs kill -9").Run()
}

func startFlaskServer(pythonPath string) {
	cmd := exec.Command(pythonPath, "server.py")
	cmd.Dir = "./model"
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Start()
}

func startNextJsServer() {
	exec.Command("pkill", "-f", "next-server").Run()
	exec.Command("pkill", "-f", "npm run dev").Run()
	time.Sleep(time.Millisecond * 500)
	exec.Command("rm", "-rf", "./frontend/node_modules/.cache").Run()
	time.Sleep(time.Millisecond * 500)

	var cmd *exec.Cmd
	if os.Getenv("APP_ENV") == "production" {
		log.Println("Building frontend...")
		buildCmd := exec.Command("npm", "run", "build")
		buildCmd.Dir = "./frontend"
		buildCmd.Run()
		cmd = exec.Command("npm", "start")
	} else {
		exec.Command("rm", "-rf", "./frontend/.next").Run()
		cmd = exec.Command("npm", "run", "dev")
	}
	cmd.Dir = "./frontend"
	cmd.Env = append(os.Environ(), "BROWSER=none", "NODE_OPTIONS=--max-http-header-size=16777216")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	log.Println("Starting Next.js...")
	cmd.Start()
}

func isPortInUse(port string) bool {
	conn, err := net.DialTimeout("tcp", "localhost:"+port, time.Millisecond*500)
	if err != nil {
		return false
	}
	conn.Close()
	return true
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

var BuildVersion = fmt.Sprintf("%d", time.Now().Unix())

func main() {
	start := time.Now()
	godotenv.Load()
	log.Printf("ENV loaded in %v", time.Since(start))

	// Initialize auth (DB + Redis)
	start = time.Now()
	if err := handlers.InitAuth(); err != nil {
		log.Fatal("Failed to initialize auth:", err)
	}
	log.Printf("Auth initialized in %v", time.Since(start))

	start = time.Now()
	handlers.InitGoogleOAuth()
	log.Printf("Google OAuth initialized in %v", time.Since(start))
	flaskPort := getEnv("FLASK_PORT", "6000")
	frontendPort := getEnv("FRONTEND_PORT", "3000")
	apiPort := getEnv("API_PORT", "8080")
	pythonPath := getEnv("PYTHON_PATH", "/opt/homebrew/bin/python3.11")

	// Kill API and Flask ports to ensure fresh start
	killPort(flaskPort)
	killPort(apiPort)
	log.Println("SCHEMALABS AI - Starting services...")
	log.Printf("Flask: %s, Frontend: %s, API: %s", flaskPort, frontendPort, apiPort)

	// Only start services if not already running
	if !isPortInUse(flaskPort) {
		log.Println("Starting Flask server...")
		go startFlaskServer(pythonPath)
	} else {
		log.Println("Flask already running on port", flaskPort)
	}

	if !isPortInUse(frontendPort) {
		go startNextJsServer()
	} else {
		log.Println("Next.js already running on port", frontendPort)
	}

	nextUrl, _ := url.Parse("http://localhost:" + frontendPort)
	nextProxy := httputil.NewSingleHostReverseProxy(nextUrl)

	// Auth routes (no auth required)
	http.HandleFunc("/api/auth/signup", enableCORS(handlers.SignupHandler))
	http.HandleFunc("/api/auth/login", enableCORS(handlers.LoginHandler))
	http.HandleFunc("/api/auth/logout", enableCORS(handlers.LogoutHandler))
	http.HandleFunc("/api/version", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Cache-Control", "no-store")
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"version":"%s"}`, BuildVersion)
	})
	http.HandleFunc("/api/auth/me", enableCORS(handlers.MeHandler))
	http.HandleFunc("/api/auth/update-profile", enableCORS(handlers.AuthMiddleware(handlers.UpdateProfileHandler)))
	http.HandleFunc("/api/auth/delete-account", enableCORS(handlers.AuthMiddleware(handlers.DeleteAccountHandler)))
	http.HandleFunc("/api/auth/change-password-request", enableCORS(handlers.AuthMiddleware(handlers.ChangePasswordRequestHandler)))
	http.HandleFunc("/api/auth/change-password-verify", enableCORS(handlers.AuthMiddleware(handlers.ChangePasswordVerifyHandler)))
	http.HandleFunc("/api/auth/logout-all", enableCORS(handlers.AuthMiddleware(handlers.LogoutAllDevicesHandler)))
	http.HandleFunc("/api/auth/upload-avatar", enableCORS(handlers.AuthMiddleware(handlers.UploadAvatarHandler)))
	http.HandleFunc("/api/auth/sessions", enableCORS(handlers.AuthMiddleware(handlers.GetSessionsHandler)))
	http.Handle("/uploads/", http.StripPrefix("/uploads/", http.FileServer(http.Dir("./uploads"))))

	// Protected API routes
	http.HandleFunc("/api/upload", enableCORS(handlers.AuthMiddleware(handlers.UploadHandler)))
	http.HandleFunc("POST /api/train", enableCORS(handlers.AuthMiddleware(handlers.TrainHandler)))
	http.HandleFunc("/api/train/multi", enableCORS(handlers.AuthMiddleware(handlers.MultiTrainHandler)))
	http.HandleFunc("/api/train/async", enableCORS(handlers.AuthMiddleware(handlers.AsyncTrainHandler)))
	http.HandleFunc("/api/train/status", enableCORS(handlers.TrainingStatusHandler))
	http.HandleFunc("/api/train/analyze", enableCORS(handlers.AuthMiddleware(handlers.AnalyzeFilesHandler)))
	http.HandleFunc("/api/train/progress", enableCORS(handlers.TrainingProgressHandler))
	http.HandleFunc("/api/config/limits", enableCORS(handlers.GetUploadLimitsHandler))
	http.HandleFunc("/api/files", enableCORS(handlers.AuthMiddleware(handlers.GetUploadedFilesHandler)))
	http.HandleFunc("/api/files/delete", enableCORS(handlers.AuthMiddleware(handlers.DeleteFileHandler)))
	http.HandleFunc("/api/files/update", enableCORS(handlers.AuthMiddleware(handlers.UpdateFileHandler)))
	http.HandleFunc("/api/generate", enableCORS(handlers.AuthMiddleware(handlers.GenerateDatasetHandler)))
	http.HandleFunc("/api/download/", enableCORS(handlers.AuthMiddleware(handlers.DownloadFileHandler)))
	http.HandleFunc("/api/folders", enableCORS(handlers.AuthMiddleware(handlers.ListFoldersHandler)))
	http.HandleFunc("/api/folders/create", enableCORS(handlers.AuthMiddleware(handlers.CreateFolderHandler)))
	http.HandleFunc("/api/folders/update", enableCORS(handlers.AuthMiddleware(handlers.UpdateFolderHandler)))
	http.HandleFunc("/api/folders/delete", enableCORS(handlers.AuthMiddleware(handlers.DeleteFolderHandler)))
	http.HandleFunc("/api/files/move", enableCORS(handlers.AuthMiddleware(handlers.MoveFileToFolderHandler)))
	http.HandleFunc("/api/search", enableCORS(handlers.AuthMiddleware(handlers.SearchHandler)))
	http.HandleFunc("/api/chat", enableCORS(handlers.AuthMiddleware(handlers.ChatHandler)))
	http.HandleFunc("/api/queries", enableCORS(handlers.AuthMiddleware(handlers.ListQueriesHandler)))
	http.HandleFunc("/api/queries/create", enableCORS(handlers.AuthMiddleware(handlers.CreateQueryHandler)))
	http.HandleFunc("/api/queries/update", enableCORS(handlers.AuthMiddleware(handlers.UpdateQueryHandler)))
	http.HandleFunc("/api/queries/delete", enableCORS(handlers.AuthMiddleware(handlers.DeleteQueryHandler)))
	http.HandleFunc("/api/messages", enableCORS(handlers.AuthMiddleware(handlers.GetMessagesHandler)))
	http.HandleFunc("/api/chat/clear", enableCORS(handlers.AuthMiddleware(handlers.ClearChatHistoryHandler)))
	http.HandleFunc("/api/predict", enableCORS(handlers.AuthMiddleware(handlers.PredictHandler)))
	http.HandleFunc("/api/predict/sector", enableCORS(handlers.AuthMiddleware(handlers.PredictSectorHandler)))
	http.HandleFunc("/api/health", enableCORS(handlers.HealthHandler))
	http.HandleFunc("/api/model/info", enableCORS(handlers.ModelInfoHandler))
	http.HandleFunc("/api/models/list", enableCORS(handlers.ModelsListHandler))
	http.HandleFunc("/api/models/switch", enableCORS(handlers.ModelsSwitchHandler))
	http.HandleFunc("/api/models/finetuned", enableCORS(handlers.AuthMiddleware(handlers.ListFineTunedModelsHandler)))
	http.HandleFunc("/api/models/finetuned/delete", enableCORS(handlers.AuthMiddleware(handlers.DeleteFineTunedModelHandler)))
	http.HandleFunc("/api/models/finetuned/update", enableCORS(handlers.AuthMiddleware(handlers.UpdateFineTunedModelHandler)))
	http.HandleFunc("/api/models/sync", enableCORS(handlers.AuthMiddleware(handlers.UpdateModelSyncHandler)))
	http.HandleFunc("/api/scheduler/status", enableCORS(handlers.AuthMiddleware(handlers.GetSchedulerStatusHandler)))
	http.HandleFunc("/api/models/finetuned/download", enableCORS(handlers.AuthMiddleware(handlers.DownloadModelHandler)))
	http.HandleFunc("/api/models/finetuned/", enableCORS(handlers.AuthMiddleware(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodDelete:
			handlers.DeleteFineTunedModelHandler(w, r)
		case http.MethodPatch, http.MethodPut:
			handlers.UpdateFineTunedModelHandler(w, r)
		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	})))
	http.HandleFunc("/api/sectors", enableCORS(handlers.SectorsHandler))
	http.HandleFunc("/api/connections", enableCORS(handlers.AuthMiddleware(handlers.ListConnectionsHandler)))
	http.HandleFunc("/api/connections/create", enableCORS(handlers.AuthMiddleware(handlers.CreateConnectionHandler)))
	http.HandleFunc("/api/connections/delete", enableCORS(handlers.AuthMiddleware(handlers.DeleteConnectionHandler)))
	http.HandleFunc("/api/connections/test", enableCORS(handlers.AuthMiddleware(handlers.TestConnectionHandler)))
	http.HandleFunc("/api/connections/update", enableCORS(handlers.AuthMiddleware(handlers.UpdateConnectionHandler)))
	http.HandleFunc("/api/connections/tables", enableCORS(handlers.AuthMiddleware(handlers.ListTablesHandler)))
	http.HandleFunc("/api/connections/export", enableCORS(handlers.AuthMiddleware(handlers.ExportTableHandler)))
	http.HandleFunc("/api/keys", enableCORS(handlers.AuthMiddleware(handlers.ListAPIKeysHandler)))
	http.HandleFunc("/api/keys/create", enableCORS(handlers.AuthMiddleware(handlers.CreateAPIKeyHandler)))
	http.HandleFunc("/api/keys/delete", enableCORS(handlers.AuthMiddleware(handlers.DeleteAPIKeyHandler)))
	http.HandleFunc("/api/endpoints", enableCORS(handlers.AuthMiddleware(handlers.ListEndpointsHandler)))
	http.HandleFunc("/api/endpoints/create", enableCORS(handlers.AuthMiddleware(handlers.CreateEndpointHandler)))
	http.HandleFunc("/api/endpoints/delete", enableCORS(handlers.AuthMiddleware(handlers.DeleteEndpointHandler)))
	http.HandleFunc("/v1/query/", enableCORS(handlers.QueryEndpointHandler))
	// Public API endpoints (API Key auth)
	http.HandleFunc("/v1/predict", enableCORS(handlers.APIKeyAuthMiddleware("query")(handlers.PredictHandler)))
	http.HandleFunc("/v1/chat", enableCORS(handlers.APIKeyAuthMiddleware("query")(handlers.ChatHandler)))
	http.HandleFunc("/v1/files", enableCORS(handlers.APIKeyAuthMiddleware("read")(handlers.GetUploadedFilesHandler)))
	http.HandleFunc("/v1/analyze", enableCORS(handlers.APIKeyAuthMiddleware("query")(handlers.AnalyzeHandler)))
	http.HandleFunc("/api/google/auth", enableCORS(handlers.GoogleAuthHandler))
	http.HandleFunc("/api/google/callback", handlers.GoogleCallbackHandler)
	http.HandleFunc("/api/google/login", handlers.GoogleLoginHandler)
	http.HandleFunc("/api/google/login/callback", handlers.GoogleLoginCallbackHandler)
	http.HandleFunc("/api/google/files", enableCORS(handlers.AuthMiddleware(handlers.GoogleDriveListHandler)))

	// Email verification and password reset
	http.HandleFunc("/api/auth/send-verification", enableCORS(handlers.SendVerificationCodeHandler))
	http.HandleFunc("/api/auth/verify-signup", enableCORS(handlers.VerifyAndSignupHandler))
	http.HandleFunc("/api/auth/request-reset", enableCORS(handlers.RequestPasswordResetHandler))
	http.HandleFunc("/api/auth/verify-reset-code", enableCORS(handlers.VerifyResetCodeHandler))
	http.HandleFunc("/api/auth/reset-password", enableCORS(handlers.ResetPasswordHandler))

	// Admin routes
	http.HandleFunc("/api/admin/users", enableCORS(handlers.AuthMiddleware(handlers.AdminUsersHandler)))
	http.HandleFunc("/api/admin/users/", enableCORS(handlers.AuthMiddleware(handlers.AdminUsersHandler)))
	http.HandleFunc("/api/admin/models", enableCORS(handlers.AuthMiddleware(handlers.AdminModelsHandler)))
	http.HandleFunc("/api/admin/models/", enableCORS(handlers.AuthMiddleware(handlers.AdminModelsHandler)))
	http.HandleFunc("/api/admin/keys", enableCORS(handlers.AuthMiddleware(handlers.AdminKeysHandler)))
	http.HandleFunc("/api/admin/keys/", enableCORS(handlers.AuthMiddleware(handlers.AdminKeysHandler)))

	// Quota & Billing
	http.HandleFunc("/api/quota", enableCORS(handlers.AuthMiddleware(handlers.QuotaHandler)))
	http.HandleFunc("/api/usage/logs", enableCORS(handlers.AuthMiddleware(handlers.GetUsageLogsHandler)))
	http.HandleFunc("/api/admin/files", enableCORS(handlers.AuthMiddleware(handlers.AdminFilesHandler)))
	http.HandleFunc("/api/admin/files/", enableCORS(handlers.AuthMiddleware(handlers.AdminFilesHandler)))
	http.HandleFunc("/api/upload/file/", enableCORS(handlers.AuthMiddleware(handlers.GetFileByIDHandler)))
	http.HandleFunc("/api/admin/queries", enableCORS(handlers.AuthMiddleware(handlers.AdminQueriesHandler)))
	http.HandleFunc("/api/admin/endpoints", enableCORS(handlers.AuthMiddleware(handlers.AdminEndpointsHandler)))
	http.HandleFunc("/api/admin/config", enableCORS(handlers.AuthMiddleware(handlers.AdminConfigHandler)))

	// Organization routes
	http.HandleFunc("/api/organizations", enableCORS(handlers.AuthMiddleware(handlers.OrganizationsHandler)))
	http.HandleFunc("/api/organizations/invite/", enableCORS(handlers.AuthMiddleware(handlers.AcceptInviteHandler)))
	http.HandleFunc("/api/organizations/", enableCORS(handlers.AuthMiddleware(func(w http.ResponseWriter, r *http.Request) {
		path := r.URL.Path
		if strings.Contains(path, "/members/") {
			handlers.OrganizationMemberHandler(w, r)
		} else if strings.Contains(path, "/members") {
			handlers.OrganizationMembersHandler(w, r)
		} else {
			handlers.OrganizationHandler(w, r)
		}
	})))

	// Serve uploaded files
	// Frontend routes with auth check
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		fmt.Println("DEBUG CATCH-ALL: ", r.URL.Path)
		path := r.URL.Path

		// Public routes - no auth needed
		publicPaths := []string{"/login", "/register", "/forgot-password", "/_next", "/favicon", "/icon", "/images", "/api/auth/session", "/api/auth/_log", "/api/auth/login", "/api/auth/signup", "/api/auth/send-verification", "/api/auth/verify-signup", "/api/auth/request-reset", "/api/auth/verify-reset-code", "/api/auth/reset-password", "/api/google/login"}
		for _, p := range publicPaths {
			if strings.HasPrefix(path, p) {
				nextProxy.ServeHTTP(w, r)
				return
			}
		}

		// Check session for all other routes
		cookie, err := r.Cookie("session")
		if err != nil {
			http.Redirect(w, r, "/login", http.StatusFound)
			return
		}

		session, err := handlers.GetSession(cookie.Value)
		if err != nil || session == nil {
			http.Redirect(w, r, "/login", http.StatusFound)
			return
		}

		nextProxy.ServeHTTP(w, r)
	})

	// Start scheduler for scheduled/real-time sync
	handlers.GlobalScheduler.Start()
	log.Println("SCHEMALABS AI running on http://localhost:" + apiPort)
	server := &http.Server{Addr: ":" + apiPort, Handler: nil, MaxHeaderBytes: 1 << 20}
	log.Fatal(server.ListenAndServe())
}

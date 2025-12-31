package main

import (
	"io"
	"log"
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
	time.Sleep(1 * time.Second)
}

func startFlaskServer(pythonPath string) {
	cmd := exec.Command(pythonPath, "server.py")
	cmd.Dir = "./model"
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Start()
	time.Sleep(3 * time.Second)
}

func startNextJsServer() {
	cmd := exec.Command("npm", "run", "dev")
	cmd.Dir = "./frontend"
	cmd.Env = append(os.Environ(), "BROWSER=none")
	cmd.Start()
	time.Sleep(5 * time.Second)
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func main() {
	godotenv.Load()

	// Initialize auth (DB + Redis)
	if err := handlers.InitAuth(); err != nil {
		log.Fatal("Failed to initialize auth:", err)
	}
	log.Println("Auth initialized (PostgreSQL + Redis)")

	handlers.InitGoogleOAuth()
	flaskPort := getEnv("FLASK_PORT", "6000")
	frontendPort := getEnv("FRONTEND_PORT", "3000")
	apiPort := getEnv("API_PORT", "8080")
	pythonPath := getEnv("PYTHON_PATH", "/opt/homebrew/bin/python3.11")

	log.Println("SCHEMALABS AI - Starting services...")
	log.Printf("Flask: %s, Frontend: %s, API: %s", flaskPort, frontendPort, apiPort)

	killPort(frontendPort)
	killPort(flaskPort)
	killPort(apiPort)

	go startFlaskServer(pythonPath)
	go startNextJsServer()

	time.Sleep(6 * time.Second)

	nextUrl, _ := url.Parse("http://localhost:" + frontendPort)
	nextProxy := httputil.NewSingleHostReverseProxy(nextUrl)

	// Auth routes (no auth required)
	http.HandleFunc("/api/auth/signup", enableCORS(handlers.SignupHandler))
	http.HandleFunc("/api/auth/login", enableCORS(handlers.LoginHandler))
	http.HandleFunc("/api/auth/logout", enableCORS(handlers.LogoutHandler))
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
	http.HandleFunc("/api/train", enableCORS(handlers.AuthMiddleware(handlers.TrainHandler)))
	http.HandleFunc("/api/train/multi", enableCORS(handlers.AuthMiddleware(handlers.MultiTrainHandler)))
	http.HandleFunc("/api/train/analyze", enableCORS(handlers.AuthMiddleware(handlers.AnalyzeFilesHandler)))
	http.HandleFunc("/api/train/progress", enableCORS(func(w http.ResponseWriter, r *http.Request) {
		queryID := r.URL.Query().Get("query_id")
flaskURL := "http://localhost:6000/training/progress"
if queryID != "" {
flaskURL = flaskURL + "?query_id=" + queryID
}
resp, err := http.Get(flaskURL)
		if err != nil {
			http.Error(w, "Flask error", 500)
			return
		}
		defer resp.Body.Close()
		w.Header().Set("Content-Type", "application/json")
		io.Copy(w, resp.Body)
	}))
	http.HandleFunc("/api/files", enableCORS(handlers.AuthMiddleware(handlers.GetUploadedFilesHandler)))
	http.HandleFunc("/api/files/delete", enableCORS(handlers.AuthMiddleware(handlers.DeleteFileHandler)))
	http.HandleFunc("/api/folders", enableCORS(handlers.AuthMiddleware(handlers.ListFoldersHandler)))
	http.HandleFunc("/api/folders/create", enableCORS(handlers.AuthMiddleware(handlers.CreateFolderHandler)))
	http.HandleFunc("/api/folders/update", enableCORS(handlers.AuthMiddleware(handlers.UpdateFolderHandler)))
	http.HandleFunc("/api/folders/delete", enableCORS(handlers.AuthMiddleware(handlers.DeleteFolderHandler)))
	http.HandleFunc("/api/files/move", enableCORS(handlers.AuthMiddleware(handlers.MoveFileToFolderHandler)))
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
	http.HandleFunc("/api/admin/files", enableCORS(handlers.AuthMiddleware(handlers.AdminFilesHandler)))
	http.HandleFunc("/api/admin/files/", enableCORS(handlers.AuthMiddleware(handlers.AdminFilesHandler)))
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
		path := r.URL.Path

		// Public routes - no auth needed
		publicPaths := []string{"/login", "/_next", "/favicon", "/icon", "/api/"}
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

	log.Println("SCHEMALABS AI running on http://localhost:" + apiPort)
	log.Fatal(http.ListenAndServe(":"+apiPort, nil))
}

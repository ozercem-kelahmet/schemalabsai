package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/smtp"
	"os"
	"strings"
	"time"

	"github.com/google/uuid"
)

type Organization struct {
	ID         string    `gorm:"primaryKey" json:"id"`
	Name       string    `json:"name"`
	Slug       string    `gorm:"unique" json:"slug"`
	OwnerID    string    `json:"owner_id"`
	Plan       string    `json:"plan"`
	MaxMembers int       `json:"max_members"`
	CreatedAt  time.Time `json:"created_at"`
	UpdatedAt  time.Time `json:"updated_at"`
}

func (Organization) TableName() string { return "organizations" }

type OrganizationMember struct {
	ID             string     `gorm:"primaryKey" json:"id"`
	OrganizationID string     `json:"organization_id"`
	UserID         *string    `json:"user_id"`
	Email          string     `json:"email"`
	Role           string     `json:"role"`
	Status         string     `json:"status"`
	InvitedBy      *string    `json:"invited_by"`
	InvitedAt      time.Time  `json:"invited_at"`
	JoinedAt       *time.Time `json:"joined_at"`
}

func (OrganizationMember) TableName() string { return "organization_members" }

type OrganizationInvite struct {
	ID             string    `gorm:"primaryKey" json:"id"`
	OrganizationID string    `json:"organization_id"`
	Email          string    `json:"email"`
	Role           string    `json:"role"`
	Token          string    `gorm:"unique" json:"token"`
	InvitedBy      *string   `json:"invited_by"`
	ExpiresAt      time.Time `json:"expires_at"`
	CreatedAt      time.Time `json:"created_at"`
}

func (OrganizationInvite) TableName() string { return "organization_invites" }

type OrgResponse struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Slug        string `json:"slug"`
	Plan        string `json:"plan"`
	MaxMembers  int    `json:"max_members"`
	MemberCount int    `json:"member_count"`
	Role        string `json:"role"`
	IsOwner     bool   `json:"is_owner"`
	CreatedAt   string `json:"created_at"`
}

type MemberResponse struct {
	ID        string  `json:"id"`
	Email     string  `json:"email"`
	Name      string  `json:"name"`
	Role      string  `json:"role"`
	Status    string  `json:"status"`
	UserID    *string `json:"user_id"`
	JoinedAt  string  `json:"joined_at"`
	InvitedAt string  `json:"invited_at"`
}

// Get user's max teams based on plan
func getMaxTeams(plan string) int {
	switch plan {
	case "pro":
		return 5
	case "enterprise":
		return 100
	default:
		return 1
	}
}

// OrganizationsHandler handles /api/organizations
func OrganizationsHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	switch r.Method {
	case "GET":
		// Get user's organizations
		var orgs []Organization
		DB.Raw(`
			SELECT o.* FROM organizations o
			LEFT JOIN organization_members om ON o.id = om.organization_id
			WHERE o.owner_id = ? OR (om.user_id = ? AND om.status = 'active')
			GROUP BY o.id
		`, userID, userID).Scan(&orgs)

		var response []OrgResponse
		for _, org := range orgs {
			var memberCount int64
			DB.Model(&OrganizationMember{}).Where("organization_id = ? AND status = 'active'", org.ID).Count(&memberCount)

			var role string
			if org.OwnerID == userID {
				role = "owner"
			} else {
				var member OrganizationMember
				DB.Where("organization_id = ? AND user_id = ?", org.ID, userID).First(&member)
				role = member.Role
			}

			response = append(response, OrgResponse{
				ID:          org.ID,
				Name:        org.Name,
				Slug:        org.Slug,
				Plan:        org.Plan,
				MaxMembers:  org.MaxMembers,
				MemberCount: int(memberCount) + 1, // +1 for owner
				Role:        role,
				IsOwner:     org.OwnerID == userID,
				CreatedAt:   org.CreatedAt.Format("2006-01-02"),
			})
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)

	case "POST":
		// Create new organization
		var user User
		DB.Where("id = ?", userID).First(&user)

		// Check team limit
		var orgCount int64
		DB.Model(&Organization{}).Where("owner_id = ?", userID).Count(&orgCount)
		maxTeams := getMaxTeams(user.Plan)
		if int(orgCount) >= maxTeams {
			http.Error(w, fmt.Sprintf("Team limit reached. Your plan allows %d team(s).", maxTeams), http.StatusForbidden)
			return
		}

		var input struct {
			Name string `json:"name"`
		}
		if err := json.NewDecoder(r.Body).Decode(&input); err != nil {
			http.Error(w, "Invalid request", http.StatusBadRequest)
			return
		}

		if input.Name == "" {
			http.Error(w, "Name is required", http.StatusBadRequest)
			return
		}

		// Generate slug
		slug := strings.ToLower(strings.ReplaceAll(input.Name, " ", "-"))
		slug = fmt.Sprintf("%s-%s", slug, uuid.New().String()[:8])

		org := Organization{
			ID:         uuid.New().String(),
			Name:       input.Name,
			Slug:       slug,
			OwnerID:    userID,
			Plan:       user.Plan,
			MaxMembers: 5,
			CreatedAt:  time.Now(),
			UpdatedAt:  time.Now(),
		}

		if err := DB.Create(&org).Error; err != nil {
			http.Error(w, "Failed to create organization", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(org)

	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

// OrganizationHandler handles /api/organizations/{id}
func OrganizationHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	// Extract org ID from URL
	path := strings.TrimPrefix(r.URL.Path, "/api/organizations/")
	parts := strings.Split(path, "/")
	orgID := parts[0]

	// Check access
	var org Organization
	if err := DB.Where("id = ?", orgID).First(&org).Error; err != nil {
		http.Error(w, "Organization not found", http.StatusNotFound)
		return
	}

	isOwner := org.OwnerID == userID
	var isMember bool
	var member OrganizationMember
	if !isOwner {
		if err := DB.Where("organization_id = ? AND user_id = ? AND status = 'active'", orgID, userID).First(&member).Error; err == nil {
			isMember = true
		}
	}

	if !isOwner && !isMember {
		http.Error(w, "Access denied", http.StatusForbidden)
		return
	}

	switch r.Method {
	case "GET":
		var memberCount int64
		DB.Model(&OrganizationMember{}).Where("organization_id = ? AND status = 'active'", org.ID).Count(&memberCount)

		role := "owner"
		if !isOwner {
			role = member.Role
		}

		response := OrgResponse{
			ID:          org.ID,
			Name:        org.Name,
			Slug:        org.Slug,
			Plan:        org.Plan,
			MaxMembers:  org.MaxMembers,
			MemberCount: int(memberCount) + 1,
			Role:        role,
			IsOwner:     isOwner,
			CreatedAt:   org.CreatedAt.Format("2006-01-02"),
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)

	case "PUT":
		if !isOwner {
			http.Error(w, "Only owner can update organization", http.StatusForbidden)
			return
		}

		var input struct {
			Name string `json:"name"`
		}
		if err := json.NewDecoder(r.Body).Decode(&input); err != nil {
			http.Error(w, "Invalid request", http.StatusBadRequest)
			return
		}

		org.Name = input.Name
		org.UpdatedAt = time.Now()
		DB.Save(&org)

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(org)

	case "DELETE":
		if !isOwner {
			http.Error(w, "Only owner can delete organization", http.StatusForbidden)
			return
		}

		DB.Where("organization_id = ?", orgID).Delete(&OrganizationMember{})
		DB.Where("organization_id = ?", orgID).Delete(&OrganizationInvite{})
		DB.Delete(&org)

		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]string{"message": "Organization deleted"})

	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

// OrganizationMembersHandler handles /api/organizations/{id}/members
func OrganizationMembersHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	path := strings.TrimPrefix(r.URL.Path, "/api/organizations/")
	parts := strings.Split(path, "/")
	orgID := parts[0]

	var org Organization
	if err := DB.Where("id = ?", orgID).First(&org).Error; err != nil {
		http.Error(w, "Organization not found", http.StatusNotFound)
		return
	}

	isOwner := org.OwnerID == userID
	isAdmin := false
	if !isOwner {
		var member OrganizationMember
		if err := DB.Where("organization_id = ? AND user_id = ? AND status = 'active'", orgID, userID).First(&member).Error; err == nil {
			isAdmin = member.Role == "admin"
		}
	}

	switch r.Method {
	case "GET":
		var members []OrganizationMember
		DB.Where("organization_id = ?", orgID).Find(&members)

		var response []MemberResponse
		
		// Add owner first
		var owner User
		DB.Where("id = ?", org.OwnerID).First(&owner)
		response = append(response, MemberResponse{
			ID:        org.OwnerID,
			Email:     owner.Email,
			Name:      owner.Name,
			Role:      "owner",
			Status:    "active",
			UserID:    &org.OwnerID,
			JoinedAt:  org.CreatedAt.Format("2006-01-02"),
			InvitedAt: org.CreatedAt.Format("2006-01-02"),
		})

		// Add members
		for _, m := range members {
			var userName string
			if m.UserID != nil {
				var user User
				DB.Where("id = ?", m.UserID).First(&user)
				userName = user.Name
			}

			joinedAt := ""
			if m.JoinedAt != nil {
				joinedAt = m.JoinedAt.Format("2006-01-02")
			}

			response = append(response, MemberResponse{
				ID:        m.ID,
				Email:     m.Email,
				Name:      userName,
				Role:      m.Role,
				Status:    m.Status,
				UserID:    m.UserID,
				JoinedAt:  joinedAt,
				InvitedAt: m.InvitedAt.Format("2006-01-02"),
			})
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)

	case "POST":
		// Invite member
		if !isOwner && !isAdmin {
			http.Error(w, "Permission denied", http.StatusForbidden)
			return
		}

		var input struct {
			Email string `json:"email"`
			Role  string `json:"role"`
		}
		if err := json.NewDecoder(r.Body).Decode(&input); err != nil {
			http.Error(w, "Invalid request", http.StatusBadRequest)
			return
		}

		if input.Email == "" {
			http.Error(w, "Email is required", http.StatusBadRequest)
			return
		}

		if input.Role == "" {
			input.Role = "member"
		}

		// Check member limit
		var memberCount int64
		DB.Model(&OrganizationMember{}).Where("organization_id = ?", orgID).Count(&memberCount)
		if int(memberCount) >= org.MaxMembers {
			http.Error(w, "Member limit reached", http.StatusForbidden)
			return
		}

		// Check if already member or invited
		var existing OrganizationMember
		if err := DB.Where("organization_id = ? AND email = ?", orgID, input.Email).First(&existing).Error; err == nil {
			http.Error(w, "User already invited or member", http.StatusConflict)
			return
		}

		// Check if user exists
		var existingUser User
		userExists := DB.Where("email = ?", input.Email).First(&existingUser).Error == nil

		// Create invite token
		token := uuid.New().String()

		// Create invite
		invite := OrganizationInvite{
			ID:             uuid.New().String(),
			OrganizationID: orgID,
			Email:          input.Email,
			Role:           input.Role,
			Token:          token,
			InvitedBy:      &userID,
			ExpiresAt:      time.Now().Add(7 * 24 * time.Hour),
			CreatedAt:      time.Now(),
		}
		DB.Create(&invite)

		// Create pending member
		member := OrganizationMember{
			ID:             uuid.New().String(),
			OrganizationID: orgID,
			Email:          input.Email,
			Role:           input.Role,
			Status:         "pending",
			InvitedBy:      &userID,
			InvitedAt:      time.Now(),
		}
		if userExists {
			member.UserID = &existingUser.ID
		}
		DB.Create(&member)

		// Send invite email
		go sendInviteEmail(input.Email, org.Name, token, userExists)

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"message": "Invitation sent", "status": "pending"})

	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

// OrganizationMemberHandler handles /api/organizations/{id}/members/{memberId}
func OrganizationMemberHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	path := strings.TrimPrefix(r.URL.Path, "/api/organizations/")
	parts := strings.Split(path, "/")
	if len(parts) < 3 {
		http.Error(w, "Invalid path", http.StatusBadRequest)
		return
	}
	orgID := parts[0]
	memberID := parts[2]

	var org Organization
	if err := DB.Where("id = ?", orgID).First(&org).Error; err != nil {
		http.Error(w, "Organization not found", http.StatusNotFound)
		return
	}

	isOwner := org.OwnerID == userID

	switch r.Method {
	case "PUT":
		// Update member role
		if !isOwner {
			http.Error(w, "Only owner can change roles", http.StatusForbidden)
			return
		}

		var input struct {
			Role string `json:"role"`
		}
		if err := json.NewDecoder(r.Body).Decode(&input); err != nil {
			http.Error(w, "Invalid request", http.StatusBadRequest)
			return
		}

		DB.Model(&OrganizationMember{}).Where("id = ? AND organization_id = ?", memberID, orgID).Update("role", input.Role)
		
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"message": "Role updated"})

	case "DELETE":
		// Remove member
		if !isOwner {
			http.Error(w, "Only owner can remove members", http.StatusForbidden)
			return
		}

		var member OrganizationMember
		if err := DB.Where("id = ? AND organization_id = ?", memberID, orgID).First(&member).Error; err != nil {
			http.Error(w, "Member not found", http.StatusNotFound)
			return
		}

		DB.Delete(&member)
		DB.Where("organization_id = ? AND email = ?", orgID, member.Email).Delete(&OrganizationInvite{})

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"message": "Member removed"})

	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

// AcceptInviteHandler handles /api/organizations/invite/{token}
func AcceptInviteHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	token := strings.TrimPrefix(r.URL.Path, "/api/organizations/invite/")

	var invite OrganizationInvite
	if err := DB.Where("token = ?", token).First(&invite).Error; err != nil {
		http.Error(w, "Invalid or expired invite", http.StatusNotFound)
		return
	}

	if time.Now().After(invite.ExpiresAt) {
		http.Error(w, "Invite expired", http.StatusGone)
		return
	}

	// Get user email
	var user User
	DB.Where("id = ?", userID).First(&user)

	if user.Email != invite.Email {
		http.Error(w, "This invite is for a different email address", http.StatusForbidden)
		return
	}

	// Update member status
	now := time.Now()
	DB.Model(&OrganizationMember{}).Where("organization_id = ? AND email = ?", invite.OrganizationID, invite.Email).Updates(map[string]interface{}{
		"user_id":   userID,
		"status":    "active",
		"joined_at": now,
	})

	// Delete invite
	DB.Delete(&invite)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"message": "Joined organization successfully", "organization_id": invite.OrganizationID})
}

// CheckInviteHandler handles GET /api/organizations/invite/{token}
func CheckInviteHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "GET" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	token := strings.TrimPrefix(r.URL.Path, "/api/organizations/invite/")

	var invite OrganizationInvite
	if err := DB.Where("token = ?", token).First(&invite).Error; err != nil {
		http.Error(w, "Invalid invite", http.StatusNotFound)
		return
	}

	if time.Now().After(invite.ExpiresAt) {
		http.Error(w, "Invite expired", http.StatusGone)
		return
	}

	var org Organization
	DB.Where("id = ?", invite.OrganizationID).First(&org)

	// Check if user exists
	var userExists bool
	var existingUser User
	if err := DB.Where("email = ?", invite.Email).First(&existingUser).Error; err == nil {
		userExists = true
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"organization_name": org.Name,
		"email":             invite.Email,
		"role":              invite.Role,
		"user_exists":       userExists,
	})
}

func sendInviteEmail(email, orgName, token string, userExists bool) {
	smtpEmail := os.Getenv("SMTP_EMAIL")
	smtpPassword := os.Getenv("SMTP_PASSWORD")
	smtpHost := os.Getenv("SMTP_HOST")
	baseURL := os.Getenv("BASE_URL")
	if baseURL == "" {
		baseURL = "https://stage.schemalabs.ai"
	}

	var actionURL string
	var actionText string
	if userExists {
		actionURL = fmt.Sprintf("%s/login?invite=%s", baseURL, token)
		actionText = "Sign in to accept"
	} else {
		actionURL = fmt.Sprintf("%s/signup?invite=%s", baseURL, token)
		actionText = "Create account to join"
	}

	subject := fmt.Sprintf("You've been invited to join %s on Schema Labs", orgName)
	body := fmt.Sprintf(`
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 40px; background: #f5f5f5;">
<div style="max-width: 500px; margin: 0 auto; background: white; border-radius: 12px; padding: 40px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
<h1 style="margin: 0 0 20px; font-size: 24px;">Join %s</h1>
<p style="color: #666; line-height: 1.6;">You've been invited to join <strong>%s</strong> on Schema Labs.</p>
<a href="%s" style="display: inline-block; margin: 20px 0; padding: 12px 24px; background: #000; color: white; text-decoration: none; border-radius: 8px; font-weight: 500;">%s</a>
<p style="color: #999; font-size: 14px; margin-top: 30px;">This invite expires in 7 days.</p>
</div>
</body>
</html>
`, orgName, orgName, actionURL, actionText)

	msg := fmt.Sprintf("From: %s\r\nTo: %s\r\nSubject: %s\r\nMIME-Version: 1.0\r\nContent-Type: text/html; charset=UTF-8\r\n\r\n%s", smtpEmail, email, subject, body)

	auth := smtp.PlainAuth("", smtpEmail, smtpPassword, smtpHost)
	smtp.SendMail(smtpHost+":587", auth, smtpEmail, []string{email}, []byte(msg))
}

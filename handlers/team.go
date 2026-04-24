package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/google/uuid"
)

type TeamData struct {
	ID         string `json:"id"`
	Name       string `json:"name"`
	Plan       string `json:"plan"`
	SeatsUsed  int    `json:"seats_used"`
	SeatsTotal int    `json:"seats_total"`
	OwnerID    string `json:"owner_id"`
}

type TeamMemberResponse struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Email       string `json:"email"`
	Role        string `json:"role"`
	Status      string `json:"status"`
	AvatarColor string `json:"avatar_color"`
	JoinedAt    string `json:"joined_at"`
	LastActive  string `json:"last_active,omitempty"`
}

type TeamInviteResponse struct {
	ID        string `json:"id"`
	Email     string `json:"email"`
	Role      string `json:"role"`
	Status    string `json:"status"`
	InvitedAt string `json:"invited_at"`
	ExpiresAt string `json:"expires_at"`
	InvitedBy string `json:"invited_by"`
}

type TeamFullResponse struct {
	Team    TeamData             `json:"team"`
	Members []TeamMemberResponse `json:"members"`
	Invites []TeamInviteResponse `json:"invites"`
}

var avatarColors = []string{
	"bg-blue-500", "bg-emerald-500", "bg-purple-500", "bg-amber-500",
	"bg-pink-500", "bg-indigo-500", "bg-teal-500", "bg-orange-500",
}

func pickAvatarColor(seed string) string {
	if seed == "" {
		return avatarColors[0]
	}
	sum := 0
	for _, c := range seed {
		sum += int(c)
	}
	return avatarColors[sum%len(avatarColors)]
}

func getSeatsForPlan(plan string) int {
	switch plan {
	case "free":
		return 1
	case "plus", "starter":
		return 1
	case "pro", "professional":
		return 3
	case "alpha_unlimited", "unlimited", "limitless":
		return 9999
	default:
		return 1
	}
}

func getSeatsForEnterpriseTier(tier int) int {
	return GetEnterpriseSeats(tier)
}

func ensurePrimaryOrg(userID string, user User) (Organization, error) {
	var org Organization
	err := DB.Where("owner_id = ?", userID).Order("created_at asc").First(&org).Error
	if err == nil {
		return org, nil
	}

	name := user.Name
	if name == "" {
		name = strings.Split(user.Email, "@")[0]
	}
	name = name + "'s Team"
	slug := strings.ToLower(strings.ReplaceAll(name, " ", "-"))
	slug = fmt.Sprintf("%s-%s", slug, uuid.New().String()[:8])

	q, _ := GetOrCreateQuota(userID)
	effectivePlan := "free"
	if q != nil && q.Plan != "" {
		effectivePlan = q.Plan
	}
	org = Organization{
		ID:         uuid.New().String(),
		Name:       name,
		Slug:       slug,
		OwnerID:    userID,
		Plan:       effectivePlan,
		MaxMembers: func() int {
			if effectivePlan == "enterprise" && q != nil {
				return GetEnterpriseSeats(q.EnterpriseTier)
			}
			return getSeatsForPlan(effectivePlan)
		}(),
		CreatedAt:  time.Now(),
		UpdatedAt:  time.Now(),
	}
	if err := DB.Create(&org).Error; err != nil {
		return org, err
	}
	return org, nil
}

func resolveUserOrg(userID string) (Organization, error) {
	var org Organization
	err := DB.Where("owner_id = ?", userID).Order("created_at asc").First(&org).Error
	if err == nil {
		return org, nil
	}

	var member OrganizationMember
	err = DB.Where("user_id = ? AND status = 'active'", userID).Order("joined_at desc").First(&member).Error
	if err != nil {
		return org, err
	}
	err = DB.Where("id = ?", member.OrganizationID).First(&org).Error
	return org, err
}

func roleIsPrivileged(role string) bool {
	return role == "owner" || role == "admin"
}

func getRoleInOrg(userID string, org Organization) string {
	if org.OwnerID == userID {
		return "owner"
	}
	var member OrganizationMember
	if err := DB.Where("organization_id = ? AND user_id = ? AND status = 'active'", org.ID, userID).First(&member).Error; err == nil {
		return member.Role
	}
	return ""
}

func TeamHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}
	if r.Method != "GET" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var user User
	if err := DB.Where("id = ?", userID).First(&user).Error; err != nil {
		http.Error(w, "User not found", http.StatusNotFound)
		return
	}

	org, err := resolveUserOrg(userID)
	if err != nil {
		org, err = ensurePrimaryOrg(userID, user)
		if err != nil {
			http.Error(w, "Failed to load team", http.StatusInternalServerError)
			return
		}
	}

	seatsTotal := org.MaxMembers
	if seatsTotal <= 0 {
		if org.Plan == "enterprise" {
		q, _ := GetOrCreateQuota(org.OwnerID)
		if q != nil {
			seatsTotal = GetEnterpriseSeats(q.EnterpriseTier)
		} else {
			seatsTotal = getSeatsForPlan(org.Plan)
		}
	} else {
		seatsTotal = getSeatsForPlan(org.Plan)
	}
	}

	resp := TeamFullResponse{
		Team: TeamData{
			ID:         org.ID,
			Name:       org.Name,
			Plan:       org.Plan,
			SeatsTotal: seatsTotal,
			OwnerID:    org.OwnerID,
		},
		Members: []TeamMemberResponse{},
		Invites: []TeamInviteResponse{},
	}

	var owner User
	DB.Where("id = ?", org.OwnerID).First(&owner)
	lastActive := ""
	if owner.LastSeen != nil {
		lastActive = owner.LastSeen.Format(time.RFC3339)
	}
	resp.Members = append(resp.Members, TeamMemberResponse{
		ID:          org.OwnerID,
		Name:        owner.Name,
		Email:       owner.Email,
		Role:        "owner",
		Status:      "active",
		AvatarColor: pickAvatarColor(owner.Email),
		JoinedAt:    org.CreatedAt.Format(time.RFC3339),
		LastActive:  lastActive,
	})

	var members []OrganizationMember
	DB.Where("organization_id = ?", org.ID).Find(&members)
	for _, m := range members {
		if m.Status == "pending" {
			continue
		}
		name := ""
		lastSeen := ""
		if m.UserID != nil {
			var u User
			if err := DB.Where("id = ?", *m.UserID).First(&u).Error; err == nil {
				name = u.Name
				if u.LastSeen != nil {
					lastSeen = u.LastSeen.Format(time.RFC3339)
				}
			}
		}
		joined := ""
		if m.JoinedAt != nil {
			joined = m.JoinedAt.Format(time.RFC3339)
		}
		resp.Members = append(resp.Members, TeamMemberResponse{
			ID:          m.ID,
			Name:        name,
			Email:       m.Email,
			Role:        m.Role,
			Status:      m.Status,
			AvatarColor: pickAvatarColor(m.Email),
			JoinedAt:    joined,
			LastActive:  lastSeen,
		})
	}

	var invites []OrganizationInvite
	DB.Where("organization_id = ?", org.ID).Find(&invites)
	for _, inv := range invites {
		status := "pending"
		if time.Now().After(inv.ExpiresAt) {
			status = "expired"
		}
		invitedByName := ""
		if inv.InvitedBy != nil {
			var u User
			if err := DB.Where("id = ?", *inv.InvitedBy).First(&u).Error; err == nil {
				invitedByName = u.Name
			}
		}
		resp.Invites = append(resp.Invites, TeamInviteResponse{
			ID:        inv.ID,
			Email:     inv.Email,
			Role:      inv.Role,
			Status:    status,
			InvitedAt: inv.CreatedAt.Format(time.RFC3339),
			ExpiresAt: inv.ExpiresAt.Format(time.RFC3339),
			InvitedBy: invitedByName,
		})
	}

	activeMembers := 0
	for _, m := range resp.Members {
		if m.Status != "deactivated" {
			activeMembers++
		}
	}
	pendingInvites := 0
	for _, inv := range resp.Invites {
		if inv.Status == "pending" {
			pendingInvites++
		}
	}
	resp.Team.SeatsUsed = activeMembers + pendingInvites

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func TeamInviteHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
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
	input.Email = strings.ToLower(strings.TrimSpace(input.Email))
	if input.Email == "" {
		http.Error(w, "Email is required", http.StatusBadRequest)
		return
	}
	if len(input.Email) > 254 || !isValidEmail(input.Email) {
		http.Error(w, "Invalid email format", http.StatusBadRequest)
		return
	}
	if input.Role == "" {
		input.Role = "member"
	}
	if input.Role != "admin" && input.Role != "member" {
		http.Error(w, "Invalid role", http.StatusBadRequest)
		return
	}

	org, err := resolveUserOrg(userID)
	if err != nil {
		var user User
		DB.Where("id = ?", userID).First(&user)
		org, err = ensurePrimaryOrg(userID, user)
		if err != nil {
			http.Error(w, "Failed to resolve team", http.StatusInternalServerError)
			return
		}
	}

	role := getRoleInOrg(userID, org)
	if !roleIsPrivileged(role) {
		http.Error(w, "Permission denied", http.StatusForbidden)
		return
	}

	var existingMember OrganizationMember
	if err := DB.Where("organization_id = ? AND email = ? AND status != 'deactivated'", org.ID, input.Email).First(&existingMember).Error; err == nil {
		http.Error(w, "This email is already a team member", http.StatusConflict)
		return
	}
	var existingInvite OrganizationInvite
	if err := DB.Where("organization_id = ? AND email = ?", org.ID, input.Email).First(&existingInvite).Error; err == nil {
		if time.Now().Before(existingInvite.ExpiresAt) {
			http.Error(w, "An invitation is already pending for this email", http.StatusConflict)
			return
		}
		DB.Delete(&existingInvite)
	}

	seatsTotal := org.MaxMembers
	if seatsTotal <= 0 {
		if org.Plan == "enterprise" {
		q, _ := GetOrCreateQuota(org.OwnerID)
		if q != nil {
			seatsTotal = GetEnterpriseSeats(q.EnterpriseTier)
		} else {
			seatsTotal = getSeatsForPlan(org.Plan)
		}
	} else {
		seatsTotal = getSeatsForPlan(org.Plan)
	}
	}
	var activeCount int64
	DB.Model(&OrganizationMember{}).Where("organization_id = ? AND status = 'active'", org.ID).Count(&activeCount)
	var pendingCount int64
	DB.Model(&OrganizationInvite{}).Where("organization_id = ? AND expires_at > ?", org.ID, time.Now()).Count(&pendingCount)
	if int(activeCount)+int(pendingCount)+1 >= seatsTotal {
		http.Error(w, fmt.Sprintf("You've reached your seat limit (%d). Upgrade your plan for more seats.", seatsTotal), http.StatusForbidden)
		return
	}

	token := uuid.New().String()
	invite := OrganizationInvite{
		ID:             uuid.New().String(),
		OrganizationID: org.ID,
		Email:          input.Email,
		Role:           input.Role,
		Token:          token,
		InvitedBy:      &userID,
		ExpiresAt:      time.Now().Add(7 * 24 * time.Hour),
		CreatedAt:      time.Now(),
	}
	if err := DB.Create(&invite).Error; err != nil {
		http.Error(w, "Failed to create invite", http.StatusInternalServerError)
		return
	}

	var existingUser User
	userExists := DB.Where("email = ?", input.Email).First(&existingUser).Error == nil
	pendingMember := OrganizationMember{
		ID:             uuid.New().String(),
		OrganizationID: org.ID,
		Email:          input.Email,
		Role:           input.Role,
		Status:         "pending",
		InvitedBy:      &userID,
		InvitedAt:      time.Now(),
	}
	if userExists {
		pendingMember.UserID = &existingUser.ID
	}
	DB.Create(&pendingMember)

	go sendTeamInviteEmail(input.Email, org.Name, token, userExists)

	var inviter User
	DB.Where("id = ?", userID).First(&inviter)
	resp := map[string]interface{}{
		"invite": TeamInviteResponse{
			ID:        invite.ID,
			Email:     invite.Email,
			Role:      invite.Role,
			Status:    "pending",
			InvitedAt: invite.CreatedAt.Format(time.RFC3339),
			ExpiresAt: invite.ExpiresAt.Format(time.RFC3339),
			InvitedBy: inviter.Name,
		},
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func TeamInviteCancelHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var input struct {
		InviteID string `json:"invite_id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&input); err != nil || input.InviteID == "" {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	org, err := resolveUserOrg(userID)
	if err != nil {
		http.Error(w, "Team not found", http.StatusNotFound)
		return
	}
	role := getRoleInOrg(userID, org)
	if !roleIsPrivileged(role) {
		http.Error(w, "Permission denied", http.StatusForbidden)
		return
	}

	var invite OrganizationInvite
	if err := DB.Where("id = ? AND organization_id = ?", input.InviteID, org.ID).First(&invite).Error; err != nil {
		http.Error(w, "Invite not found", http.StatusNotFound)
		return
	}

	DB.Where("organization_id = ? AND email = ? AND status = 'pending'", org.ID, invite.Email).Delete(&OrganizationMember{})
	DB.Delete(&invite)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"message": "Invite cancelled"})
}

func TeamInviteResendHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var input struct {
		InviteID string `json:"invite_id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&input); err != nil || input.InviteID == "" {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	org, err := resolveUserOrg(userID)
	if err != nil {
		http.Error(w, "Team not found", http.StatusNotFound)
		return
	}
	role := getRoleInOrg(userID, org)
	if !roleIsPrivileged(role) {
		http.Error(w, "Permission denied", http.StatusForbidden)
		return
	}

	var invite OrganizationInvite
	if err := DB.Where("id = ? AND organization_id = ?", input.InviteID, org.ID).First(&invite).Error; err != nil {
		http.Error(w, "Invite not found", http.StatusNotFound)
		return
	}

	invite.ExpiresAt = time.Now().Add(7 * 24 * time.Hour)
	invite.Token = uuid.New().String()
	DB.Save(&invite)

	var existingUser User
	userExists := DB.Where("email = ?", invite.Email).First(&existingUser).Error == nil
	go sendTeamInviteEmail(invite.Email, "Schema Labs", invite.Token, userExists)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"message":    "Invite resent",
		"expires_at": invite.ExpiresAt.Format(time.RFC3339),
	})
}

func TeamRemoveHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var input struct {
		MemberID string `json:"member_id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&input); err != nil || input.MemberID == "" {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	org, err := resolveUserOrg(userID)
	if err != nil {
		http.Error(w, "Team not found", http.StatusNotFound)
		return
	}
	if org.OwnerID != userID {
		http.Error(w, "Only owner can remove members", http.StatusForbidden)
		return
	}
	if input.MemberID == org.OwnerID {
		http.Error(w, "Cannot remove owner", http.StatusBadRequest)
		return
	}

	var member OrganizationMember
	if err := DB.Where("id = ? AND organization_id = ?", input.MemberID, org.ID).First(&member).Error; err != nil {
		http.Error(w, "Member not found", http.StatusNotFound)
		return
	}

	DB.Delete(&member)
	DB.Where("organization_id = ? AND email = ?", org.ID, member.Email).Delete(&OrganizationInvite{})

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"message": "Member removed"})
}

func TeamDeactivateHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var input struct {
		MemberID string `json:"member_id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&input); err != nil || input.MemberID == "" {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	org, err := resolveUserOrg(userID)
	if err != nil {
		http.Error(w, "Team not found", http.StatusNotFound)
		return
	}
	if org.OwnerID != userID {
		http.Error(w, "Only owner can deactivate members", http.StatusForbidden)
		return
	}
	if input.MemberID == org.OwnerID {
		http.Error(w, "Cannot deactivate owner", http.StatusBadRequest)
		return
	}

	res := DB.Model(&OrganizationMember{}).Where("id = ? AND organization_id = ?", input.MemberID, org.ID).Update("status", "deactivated")
	if res.RowsAffected == 0 {
		http.Error(w, "Member not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"message": "Member deactivated"})
}

func TeamReactivateHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var input struct {
		MemberID string `json:"member_id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&input); err != nil || input.MemberID == "" {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	org, err := resolveUserOrg(userID)
	if err != nil {
		http.Error(w, "Team not found", http.StatusNotFound)
		return
	}
	if org.OwnerID != userID {
		http.Error(w, "Only owner can reactivate members", http.StatusForbidden)
		return
	}

	res := DB.Model(&OrganizationMember{}).Where("id = ? AND organization_id = ?", input.MemberID, org.ID).Update("status", "active")
	if res.RowsAffected == 0 {
		http.Error(w, "Member not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"message": "Member reactivated"})
}

func TeamRoleHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var input struct {
		MemberID string `json:"member_id"`
		Role     string `json:"role"`
	}
	if err := json.NewDecoder(r.Body).Decode(&input); err != nil || input.MemberID == "" {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}
	if input.Role != "admin" && input.Role != "member" {
		http.Error(w, "Invalid role", http.StatusBadRequest)
		return
	}

	org, err := resolveUserOrg(userID)
	if err != nil {
		http.Error(w, "Team not found", http.StatusNotFound)
		return
	}
	if org.OwnerID != userID {
		http.Error(w, "Only owner can change roles", http.StatusForbidden)
		return
	}
	if input.MemberID == org.OwnerID {
		http.Error(w, "Cannot change owner role", http.StatusBadRequest)
		return
	}

	res := DB.Model(&OrganizationMember{}).Where("id = ? AND organization_id = ?", input.MemberID, org.ID).Update("role", input.Role)
	if res.RowsAffected == 0 {
		http.Error(w, "Member not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"message": "Role updated"})
}

func sendTeamInviteEmail(email, orgName, token string, userExists bool) {
	smtpEmail := os.Getenv("SMTP_EMAIL")
	smtpPassword := os.Getenv("SMTP_PASSWORD")
	smtpHost := os.Getenv("SMTP_HOST")
	if smtpEmail == "" || smtpPassword == "" || smtpHost == "" {
		return
	}
	sendInviteEmail(email, orgName, token, userExists)
}

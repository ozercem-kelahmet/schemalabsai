package handlers

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"encoding/json"
	"net/http"
	"net/smtp"
	"strings"
	"os"
	"time"
)

type EmailService struct {
	host     string
	port     string
	email    string
	password string
}

func NewEmailService() *EmailService {
	return &EmailService{
		host:     os.Getenv("SMTP_HOST"),
		port:     os.Getenv("SMTP_PORT"),
		email:    os.Getenv("SMTP_EMAIL"),
		password: os.Getenv("SMTP_PASSWORD"),
	}
}

func generateMessageID() string {
	bytes := make([]byte, 16)
	rand.Read(bytes)
	return fmt.Sprintf("<%s.%d@schemalabs.ai>", hex.EncodeToString(bytes), time.Now().UnixNano())
}

func (e *EmailService) SendEmail(to, subject, htmlBody string) error {
	auth := smtp.PlainAuth("", e.email, e.password, e.host)

	boundary := fmt.Sprintf("boundary%d", time.Now().UnixNano())
	msgID := generateMessageID()

	// Plain text version
	plainText := "Your SchemaLabs code is in this email. Please view with an HTML-capable email client."

	msg := fmt.Sprintf("From: SchemaLabs <hello@schemalabs.ai>\r\n"+
		"To: %s\r\n"+
		"Subject: %s\r\n"+
		"Message-ID: %s\r\n"+
		"Date: %s\r\n"+
		"MIME-Version: 1.0\r\n"+
		"Content-Type: multipart/alternative; boundary=\"%s\"\r\n"+
		"\r\n"+
		"--%s\r\n"+
		"Content-Type: text/plain; charset=UTF-8\r\n"+
		"Content-Transfer-Encoding: 7bit\r\n"+
		"\r\n"+
		"%s\r\n"+
		"\r\n"+
		"--%s\r\n"+
		"Content-Type: text/html; charset=UTF-8\r\n"+
		"Content-Transfer-Encoding: 7bit\r\n"+
		"\r\n"+
		"%s\r\n"+
		"--%s--\r\n",
		to, subject, msgID, time.Now().Format(time.RFC1123Z), boundary, boundary, plainText, boundary, htmlBody, boundary)

	addr := fmt.Sprintf("%s:%s", e.host, e.port)
	return smtp.SendMail(addr, auth, e.email, []string{to}, []byte(msg))
}

func (e *EmailService) SendVerificationCode(to, code string) error {
	subject := "Your SchemaLabs verification code"
	body := fmt.Sprintf(`<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body>
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
	<h2 style="color: #333;">Welcome to SchemaLabs!</h2>
	<p>Your verification code is:</p>
	<div style="background: #f5f5f5; padding: 20px; text-align: center; margin: 20px 0; border-radius: 8px;">
		<span style="font-size: 32px; font-weight: bold; letter-spacing: 8px; color: #333;">%s</span>
	</div>
	<p style="color: #666;">This code will expire in 10 minutes.</p>
	<p style="color: #666;">If you did not request this code, please ignore this email.</p>
</div>
</body>
</html>`, code)
	return e.SendEmail(to, subject, body)
}

func (e *EmailService) SendPasswordReset(to, resetLink string) error {
	subject := "Reset your SchemaLabs password"
	body := fmt.Sprintf(`<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body>
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
	<h2 style="color: #333;">Password Reset Request</h2>
	<p>Click the button below to reset your password:</p>
	<div style="text-align: center; margin: 30px 0;">
		<a href="%s" style="background: #000; color: #fff; padding: 12px 30px; text-decoration: none; border-radius: 6px; display: inline-block;">Reset Password</a>
	</div>
	<p style="color: #666;">This link will expire in 1 hour.</p>
	<p style="color: #666;">If you did not request this, please ignore this email.</p>
</div>
</body>
</html>`, resetLink)
	return e.SendEmail(to, subject, body)
}

func (e *EmailService) SendTrainingComplete(to, modelName string, accuracy float64) error {
	subject := "Your SchemaLabs model is ready"
	body := fmt.Sprintf(`<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body>
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
	<h2 style="color: #333;">Training Complete!</h2>
	<p>Your model <strong>%s</strong> has finished training.</p>
	<div style="background: #f0fdf4; padding: 20px; margin: 20px 0; border-radius: 8px; border-left: 4px solid #22c55e;">
		<p style="margin: 0; color: #166534;"><strong>Final Accuracy: %.1f%%</strong></p>
	</div>
	<p>You can now use your model in the playground or via API.</p>
	<div style="text-align: center; margin: 30px 0;">
		<a href="https://console.schemalabs.ai/playground" style="background: #000; color: #fff; padding: 12px 30px; text-decoration: none; border-radius: 6px; display: inline-block;">Go to Playground</a>
	</div>
</div>
</body>
</html>`, modelName, accuracy)
	return e.SendEmail(to, subject, body)
}

func (e *EmailService) SendPasswordResetCode(to, code string) error {
	subject := "Your SchemaLabs password reset code"
	body := fmt.Sprintf(`<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body>
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
	<h2 style="color: #333;">Password Reset Request</h2>
	<p>Your password reset code is:</p>
	<div style="background: #f5f5f5; padding: 20px; text-align: center; margin: 20px 0; border-radius: 8px;">
		<span style="font-size: 32px; font-weight: bold; letter-spacing: 8px; color: #333;">%s</span>
	</div>
	<p style="color: #666;">This code will expire in 10 minutes.</p>
	<p style="color: #666;">If you did not request this, please ignore this email.</p>
</div>
</body>
</html>`, code)
	return e.SendEmail(to, subject, body)
}


func getIPLocation(ip string) string {
	cleanIP := strings.TrimSpace(strings.Split(ip, ",")[0])
	if strings.Contains(cleanIP, ":") && !strings.Contains(cleanIP, "[") {
		cleanIP, _, _ = strings.Cut(cleanIP, ":")
	}
	resp, err := http.Get("http://ip-api.com/json/" + cleanIP + "?fields=country,regionName,city,isp")
	if err != nil {
		return cleanIP
	}
	defer resp.Body.Close()
	var result struct {
		Country    string `json:"country"`
		RegionName string `json:"regionName"`
		City       string `json:"city"`
		ISP        string `json:"isp"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return cleanIP
	}
	return fmt.Sprintf("%s, %s, %s (%s)", result.City, result.RegionName, result.Country, result.ISP)
}


func SendNewUserNotification(name, email, ip string) {
	notifyEmails := os.Getenv("NOTIFY_EMAILS")
	if notifyEmails == "" {
		return
	}
	location := getIPLocation(ip)
	svc := NewEmailService()
	subject := "[SchemaLabs] New User - " + name
	body := "<div style='font-family:sans-serif;max-width:600px;margin:0 auto;padding:32px;background:#fff'>"
	body += "<div style='text-align:center;margin-bottom:24px'>"
	body += "<div style='font-family:-apple-system,BlinkMacSystemFont,sans-serif;font-size:26px;font-weight:700;letter-spacing:-0.5px;margin-bottom:4px'><span style='color:#000000'>Schema</span><span style='color:#555555'>Labs</span></div>"
	body += "<h2 style='color:#111111;margin:12px 0 4px'>New User Registered</h2>"
	body += "<p style='color:#555555;margin:0;font-size:14px'>SchemaLabs AI Platform</p>"
	body += "</div>"
	body += "<table style='width:100%;border-collapse:collapse;border:1px solid #f0f0f0;border-radius:8px'>"
	body += fmt.Sprintf("<tr style='border-bottom:1px solid #f0f0f0'><td style='padding:12px 16px;color:#666;width:120px;background:#fafafa'>Name</td><td style='padding:12px 16px;font-weight:600'>%s</td></tr>", name)
	body += fmt.Sprintf("<tr style='border-bottom:1px solid #f0f0f0'><td style='padding:12px 16px;color:#666;background:#fafafa'>Email</td><td style='padding:12px 16px;font-weight:600'>%s</td></tr>", email)
	body += fmt.Sprintf("<tr style='border-bottom:1px solid #f0f0f0'><td style='padding:12px 16px;color:#666;background:#fafafa'>Location</td><td style='padding:12px 16px;font-weight:600'>%s</td></tr>", location)
	body += fmt.Sprintf("<tr style='border-bottom:1px solid #f0f0f0'><td style='padding:12px 16px;color:#666;background:#fafafa'>IP</td><td style='padding:12px 16px;font-weight:600'>%s</td></tr>", ip)
	body += fmt.Sprintf("<tr><td style='padding:12px 16px;color:#666;background:#fafafa'>Time</td><td style='padding:12px 16px;font-weight:600'>%s UTC</td></tr>", time.Now().UTC().Format("2006-01-02 15:04:05"))
	body += "</table>"
	body += "<p style='margin-top:24px;color:#bbb;font-size:12px;text-align:center'>SchemaLabs AI Admin Notification</p>"
	body += "</div>"
	for _, r := range strings.Split(notifyEmails, ",") {
		svc.SendEmail(strings.TrimSpace(r), subject, body)
	}
}

package handlers

import (
	"fmt"
	"net/smtp"
	"os"
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

func (e *EmailService) SendEmail(to, subject, body string) error {
	auth := smtp.PlainAuth("", e.email, e.password, e.host)

	msg := fmt.Sprintf("From: SchemaLabs <%s>\r\n"+
		"To: %s\r\n"+
		"Subject: %s\r\n"+
		"MIME-Version: 1.0\r\n"+
		"Content-Type: text/html; charset=UTF-8\r\n"+
		"\r\n"+
		"%s", e.email, to, subject, body)

	addr := fmt.Sprintf("%s:%s", e.host, e.port)
	return smtp.SendMail(addr, auth, e.email, []string{to}, []byte(msg))
}

func (e *EmailService) SendVerificationCode(to, code string) error {
	subject := "SchemaLabs - Email Verification Code"
	body := fmt.Sprintf(`
		<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
			<h2 style="color: #333;">Welcome to SchemaLabs!</h2>
			<p>Your verification code is:</p>
			<div style="background: #f5f5f5; padding: 20px; text-align: center; margin: 20px 0; border-radius: 8px;">
				<span style="font-size: 32px; font-weight: bold; letter-spacing: 8px; color: #333;">%s</span>
			</div>
			<p style="color: #666;">This code will expire in 10 minutes.</p>
			<p style="color: #666;">If you didn't request this code, please ignore this email.</p>
		</div>
	`, code)
	return e.SendEmail(to, subject, body)
}

func (e *EmailService) SendPasswordReset(to, resetLink string) error {
	subject := "SchemaLabs - Password Reset"
	body := fmt.Sprintf(`
		<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
			<h2 style="color: #333;">Password Reset Request</h2>
			<p>Click the button below to reset your password:</p>
			<div style="text-align: center; margin: 30px 0;">
				<a href="%s" style="background: #000; color: #fff; padding: 12px 30px; text-decoration: none; border-radius: 6px; display: inline-block;">Reset Password</a>
			</div>
			<p style="color: #666;">This link will expire in 1 hour.</p>
			<p style="color: #666;">If you didn't request this, please ignore this email.</p>
		</div>
	`, resetLink)
	return e.SendEmail(to, subject, body)
}

func (e *EmailService) SendTrainingComplete(to, modelName string, accuracy float64) error {
	subject := "SchemaLabs - Model Training Complete!"
	body := fmt.Sprintf(`
		<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
			<h2 style="color: #333;">🎉 Training Complete!</h2>
			<p>Your model <strong>%s</strong> has finished training.</p>
			<div style="background: #f0fdf4; padding: 20px; margin: 20px 0; border-radius: 8px; border-left: 4px solid #22c55e;">
				<p style="margin: 0; color: #166534;"><strong>Final Accuracy: %.1f%%</strong></p>
			</div>
			<p>You can now use your model in the playground or via API.</p>
			<div style="text-align: center; margin: 30px 0;">
				<a href="https://schemalabs.ai/playground" style="background: #000; color: #fff; padding: 12px 30px; text-decoration: none; border-radius: 6px; display: inline-block;">Go to Playground</a>
			</div>
		</div>
	`, modelName, accuracy)
	return e.SendEmail(to, subject, body)
}

func (e *EmailService) SendPasswordResetCode(to, code string) error {
	subject := "SchemaLabs - Password Reset Code"
	body := fmt.Sprintf(`
		<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
			<h2 style="color: #333;">Password Reset Request</h2>
			<p>Your password reset code is:</p>
			<div style="background: #f5f5f5; padding: 20px; text-align: center; margin: 20px 0; border-radius: 8px;">
				<span style="font-size: 32px; font-weight: bold; letter-spacing: 8px; color: #333;">%s</span>
			</div>
			<p style="color: #666;">This code will expire in 10 minutes.</p>
			<p style="color: #666;">If you didn't request this, please ignore this email.</p>
		</div>
	`, code)
	return e.SendEmail(to, subject, body)
}

package services

import (
	"fmt"
	"net/smtp"
	"os"
)

func SendTrainingCompletedEmail(userEmail, queryName string, accuracy float64) error {
	smtpHost := os.Getenv("SMTP_HOST")
	smtpPort := os.Getenv("SMTP_PORT")
	smtpEmail := os.Getenv("SMTP_EMAIL")
	smtpPassword := os.Getenv("SMTP_PASSWORD")

	if smtpHost == "" || smtpEmail == "" || smtpPassword == "" {
		return fmt.Errorf("SMTP configuration missing")
	}

	subject := fmt.Sprintf("Training Completed: %s", queryName)
	body := fmt.Sprintf(`
Your fine-tuning job has completed successfully!

Model: %s
Accuracy: %.1f%%

You can now use your trained model in SchemaLabs AI.

Best regards,
SchemaLabs AI Team
`, queryName, accuracy)

	msg := fmt.Sprintf("From: %s\r\nTo: %s\r\nSubject: %s\r\nContent-Type: text/plain; charset=UTF-8\r\n\r\n%s",
		smtpEmail, userEmail, subject, body)

	auth := smtp.PlainAuth("", smtpEmail, smtpPassword, smtpHost)
	addr := fmt.Sprintf("%s:%s", smtpHost, smtpPort)

	return smtp.SendMail(addr, auth, smtpEmail, []string{userEmail}, []byte(msg))
}

func SendTrainingFailedEmail(userEmail, queryName, errorMsg string) error {
	smtpHost := os.Getenv("SMTP_HOST")
	smtpPort := os.Getenv("SMTP_PORT")
	smtpEmail := os.Getenv("SMTP_EMAIL")
	smtpPassword := os.Getenv("SMTP_PASSWORD")

	if smtpHost == "" || smtpEmail == "" || smtpPassword == "" {
		return fmt.Errorf("SMTP configuration missing")
	}

	subject := fmt.Sprintf("Training Failed: %s", queryName)
	body := fmt.Sprintf(`
Unfortunately, your fine-tuning job has failed.

Model: %s
Error: %s

Please try again or contact support if the issue persists.

Best regards,
SchemaLabs AI Team
`, queryName, errorMsg)

	msg := fmt.Sprintf("From: %s\r\nTo: %s\r\nSubject: %s\r\nContent-Type: text/plain; charset=UTF-8\r\n\r\n%s",
		smtpEmail, userEmail, subject, body)

	auth := smtp.PlainAuth("", smtpEmail, smtpPassword, smtpHost)
	addr := fmt.Sprintf("%s:%s", smtpHost, smtpPort)

	return smtp.SendMail(addr, auth, smtpEmail, []string{userEmail}, []byte(msg))
}

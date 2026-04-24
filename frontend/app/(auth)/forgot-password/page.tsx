"use client"

import React from "react"
import { useState, useEffect } from "react"
import Link from "next/link"
import Image from "next/image"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Loader2, ArrowLeft } from "lucide-react"

type Step = "email" | "verify" | "reset" | "success"

export default function ForgotPasswordPage() {
  const [step, setStep] = useState<Step>("email")
  const [email, setEmail] = useState("")
  const [verificationCode, setVerificationCode] = useState("")
  const [newPassword, setNewPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")
  const [isDark, setIsDark] = useState(true)

  useEffect(() => {
    const savedTheme = localStorage.getItem("schemalabs-theme")
    setIsDark(savedTheme !== "light")
  }, [])

  const handleSendCode = async (e: React.FormEvent) => {
    e.preventDefault()
    setError("")

    if (!email.trim()) {
      setError("Email is required")
      return
    }
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      setError("Please enter a valid email address")
      return
    }

    setIsLoading(true)
    try {
      const res = await fetch("/api/auth/request-reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email }),
      })

      const data = await res.json()
      if (!res.ok) {
        setError(data.error || "Failed to send reset code")
        setIsLoading(false)
        return
      }

      setStep("verify")
    } catch {
      setError("Something went wrong")
    }
    setIsLoading(false)
  }

  const handleVerifyCode = async (e: React.FormEvent) => {
    e.preventDefault()
    setError("")

    if (verificationCode.length !== 6) {
      setError("Please enter a 6-digit code")
      return
    }

    setIsLoading(true)
    try {
      const res = await fetch("/api/auth/verify-reset-code", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, code: verificationCode }),
      })

      const data = await res.json()
      if (!res.ok) {
        setError(data.error || "Invalid verification code")
        setIsLoading(false)
        return
      }

      setStep("reset")
    } catch {
      setError("Something went wrong")
    }
    setIsLoading(false)
  }

  const handleResetPassword = async (e: React.FormEvent) => {
    e.preventDefault()
    setError("")

    if (newPassword.length < 6) {
      setError("Password must be at least 6 characters")
      return
    }
    if (newPassword !== confirmPassword) {
      setError("Passwords do not match")
      return
    }

    setIsLoading(true)
    try {
      const res = await fetch("/api/auth/reset-password", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, code: verificationCode, password: newPassword }),
      })

      const data = await res.json()
      if (!res.ok) {
        setError(data.error || "Failed to reset password")
        setIsLoading(false)
        return
      }

      setStep("success")
    } catch {
      setError("Something went wrong")
    }
    setIsLoading(false)
  }

  return (
    <div className="relative min-h-screen w-full overflow-hidden bg-background">
      <div className="absolute inset-0 bg-[linear-gradient(to_right,hsl(var(--border))_1px,transparent_1px),linear-gradient(to_bottom,hsl(var(--border))_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_80%_50%_at_50%_0%,#000_70%,transparent_110%)]" />
      <div className="absolute inset-0 bg-gradient-to-b from-[#0052CC]/5 via-transparent to-transparent" />
      
      <div className="relative flex min-h-screen flex-col items-center justify-center px-4">
        <Link href="/" className="mb-8">
          <Image unoptimized
            src={isDark ? "/images/schemalabs-light.png" : "/images/schemalabs-dark.png"}
            alt="SchemaLabs"
            width={180}
            height={40}
            className="h-10 w-auto"
            priority
          />
        </Link>

        <div className="w-full max-w-[400px] rounded-2xl border border-border bg-card/80 p-8 shadow-xl backdrop-blur-sm">
          {step === "email" && (
            <>
              <Link href="/login" className="mb-4 flex items-center text-sm text-muted-foreground hover:text-foreground">
                <ArrowLeft className="h-4 w-4 mr-1" />
                Back to login
              </Link>

              <div className="mb-6 text-center">
                <h1 className="text-2xl font-semibold text-foreground">Forgot password?</h1>
                <p className="mt-2 text-sm text-muted-foreground">
                  Enter your email and we'll send you a reset code
                </p>
              </div>

              {error && (
                <div className="mb-4 rounded-lg bg-destructive/10 p-3 text-sm text-destructive">
                  {error}
                </div>
              )}

              <form onSubmit={handleSendCode} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="email" className="text-foreground">Email</Label>
                  <Input
                    id="email"
                    type="email"
                    placeholder="you@company.com"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="h-11 border-border bg-background text-foreground placeholder:text-muted-foreground"
                    required
                  />
                </div>

                <Button
                  type="submit"
                  className="h-11 w-full bg-[#0052CC] text-white hover:bg-[#003D99]"
                  disabled={isLoading}
                >
                  {isLoading ? (
                    <div className="flex items-center gap-2">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Sending...
                    </div>
                  ) : (
                    "Send reset code"
                  )}
                </Button>
              </form>
            </>
          )}

          {step === "verify" && (
            <>
              <button
                type="button"
                onClick={() => setStep("email")}
                className="mb-4 flex items-center text-sm text-muted-foreground hover:text-foreground"
              >
                <ArrowLeft className="h-4 w-4 mr-1" />
                Back
              </button>

              <div className="mb-6 text-center">
                <h1 className="text-2xl font-semibold text-foreground">Check your email</h1>
                <p className="mt-2 text-sm text-muted-foreground">
                  Enter the 6-digit code sent to <span className="font-medium text-foreground">{email}</span>
                </p>
              </div>

              {error && (
                <div className="mb-4 rounded-lg bg-destructive/10 p-3 text-sm text-destructive">
                  {error}
                </div>
              )}

              <form onSubmit={handleVerifyCode} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="code" className="text-foreground">Verification Code</Label>
                  <Input
                    id="code"
                    type="text"
                    placeholder="000000"
                    className="h-11 text-center text-2xl tracking-[0.5em] font-mono border-border bg-background text-foreground"
                    value={verificationCode}
                    onChange={(e) => setVerificationCode(e.target.value.replace(/\D/g, "").slice(0, 6))}
                    maxLength={6}
                    required
                  />
                </div>

                <Button
                  type="submit"
                  className="h-11 w-full bg-[#0052CC] text-white hover:bg-[#003D99]"
                  disabled={isLoading || verificationCode.length !== 6}
                >
                  {isLoading ? (
                    <div className="flex items-center gap-2">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Verifying...
                    </div>
                  ) : (
                    "Verify code"
                  )}
                </Button>

                <p className="text-center text-sm text-muted-foreground">
                  Didn't receive the code?{" "}
                  <button
                    type="button"
                    onClick={handleSendCode}
                    className="font-medium text-[#0052CC] hover:text-[#2684FF] dark:text-[#2684FF] dark:hover:text-[#4C9AFF]"
                    disabled={isLoading}
                  >
                    Resend
                  </button>
                </p>
              </form>
            </>
          )}

          {step === "reset" && (
            <>
              <div className="mb-6 text-center">
                <h1 className="text-2xl font-semibold text-foreground">Reset password</h1>
                <p className="mt-2 text-sm text-muted-foreground">
                  Enter your new password
                </p>
              </div>

              {error && (
                <div className="mb-4 rounded-lg bg-destructive/10 p-3 text-sm text-destructive">
                  {error}
                </div>
              )}

              <form onSubmit={handleResetPassword} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="newPassword" className="text-foreground">New Password</Label>
                  <Input
                    id="newPassword"
                    type="password"
                    placeholder="Min 6 characters"
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                    className="h-11 border-border bg-background text-foreground placeholder:text-muted-foreground"
                    required
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="confirmPassword" className="text-foreground">Confirm Password</Label>
                  <Input
                    id="confirmPassword"
                    type="password"
                    placeholder="Confirm your password"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    className="h-11 border-border bg-background text-foreground placeholder:text-muted-foreground"
                    required
                  />
                </div>

                <Button
                  type="submit"
                  className="h-11 w-full bg-[#0052CC] text-white hover:bg-[#003D99]"
                  disabled={isLoading}
                >
                  {isLoading ? (
                    <div className="flex items-center gap-2">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Resetting...
                    </div>
                  ) : (
                    "Reset password"
                  )}
                </Button>
              </form>
            </>
          )}

          {step === "success" && (
            <>
              <div className="mb-6 text-center">
                <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-green-500/10">
                  <svg className="h-6 w-6 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                </div>
                <h1 className="text-2xl font-semibold text-foreground">Password reset!</h1>
                <p className="mt-2 text-sm text-muted-foreground">
                  Your password has been successfully reset
                </p>
              </div>

              <Link href="/login">
                <Button className="h-11 w-full bg-[#0052CC] text-white hover:bg-[#003D99]">
                  Back to login
                </Button>
              </Link>
            </>
          )}
        </div>

        <div className="mt-6 flex items-center gap-2 rounded-full border border-border bg-card/50 px-4 py-2 text-xs text-muted-foreground backdrop-blur-sm">
          <div className="h-2 w-2 rounded-full bg-[#0052CC] animate-pulse" />
          Beta Version
        </div>
      </div>
    </div>
  )
}

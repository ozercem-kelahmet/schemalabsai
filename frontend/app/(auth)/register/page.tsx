"use client"

import React from "react"

import { useState, useEffect } from "react"
import Link from "next/link"
import Image from "next/image"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Eye, EyeOff, ArrowLeft, Check } from "lucide-react"

export default function RegisterPage() {
  const router = useRouter()
  const [step, setStep] = useState<"details" | "verify">("details")
  const [name, setName] = useState("")
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [verificationCode, setVerificationCode] = useState(["", "", "", "", "", ""])
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")
  const [isDark, setIsDark] = useState(true)

  useEffect(() => {
    const savedTheme = localStorage.getItem("schemalabs-theme")
    setIsDark(savedTheme !== "light")
  }, [])

  const passwordRequirements = [
    { label: "At least 8 characters", met: password.length >= 8 },
    { label: "Contains a number", met: /\d/.test(password) },
    { label: "Contains uppercase", met: /[A-Z]/.test(password) },
  ]

  const isPasswordValid = passwordRequirements.every((req) => req.met)
  const doPasswordsMatch = password === confirmPassword && confirmPassword.length > 0

  const handleSubmitDetails = async (e: React.FormEvent) => {
    e.preventDefault()
    setError("")

    if (!isPasswordValid) {
      setError("Please meet all password requirements")
      return
    }

    if (!doPasswordsMatch) {
      setError("Passwords do not match")
      return
    }


    setIsLoading(true)
    try {
      const res = await fetch("/api/auth/send-verification", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email }),
      })
      const data = await res.json()
      if (!res.ok) {
        setError(data.error || "Failed to send verification code")
        setIsLoading(false)
        return
      }
      setStep("verify")
    } catch {
      setError("Something went wrong")
    }
    setIsLoading(false)  }

  const handleVerificationInput = (index: number, value: string) => {
    if (value.length > 1) return
    
    const newCode = [...verificationCode]
    newCode[index] = value
    setVerificationCode(newCode)

    // Auto-focus next input
    if (value && index < 5) {
      const nextInput = document.getElementById(`code-${index + 1}`)
      nextInput?.focus()
    }
  }

  const handleKeyDown = (index: number, e: React.KeyboardEvent) => {
    if (e.key === "Backspace" && !verificationCode[index] && index > 0) {
      const prevInput = document.getElementById(`code-${index - 1}`)
      prevInput?.focus()
    }
  }

  const handleVerifyCode = async () => {
    const code = verificationCode.join("")
    if (code.length !== 6) {
      setError("Please enter the complete verification code")
      return
    }

    setIsLoading(true)
    try {
      const res = await fetch("/api/auth/verify-signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, code, name, password }),
      })
      const data = await res.json()
      if (!res.ok) {
        setError(data.error || "Verification failed")
        setIsLoading(false)
        return
      }
      if (data.token) {
        document.cookie = `session=${data.token}; path=/; max-age=${7 * 24 * 60 * 60}`
      }
      window.location.href = "/"
    } catch {
      setError("Something went wrong")
    }
    setIsLoading(false)
  }

  const handleGoogleSignUp = () => {
    setIsLoading(true)
    window.location.href = "/api/google/login"
  }
  const handleResendCode = async () => {
    setIsLoading(true)
    try {
      await fetch("/api/auth/send-verification", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email }),
      })
      setVerificationCode(["", "", "", "", "", ""])
    } catch {
      setError("Failed to resend code")
    }
    setIsLoading(false)
  }
  return (
    <div className="relative min-h-screen w-full overflow-hidden bg-background">
      {/* Grid Background */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,hsl(var(--border))_1px,transparent_1px),linear-gradient(to_bottom,hsl(var(--border))_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_80%_50%_at_50%_0%,#000_70%,transparent_110%)]" />
      
      {/* Gradient Overlay */}
      <div className="absolute inset-0 bg-gradient-to-b from-[#0052CC]/5 via-transparent to-transparent" />
      
      {/* Content */}
      <div className="relative flex min-h-screen flex-col items-center justify-center px-4 py-8">
        {/* Logo */}
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

        {/* Card */}
        <div className="w-full max-w-[400px] rounded-2xl border border-border bg-card/80 p-8 shadow-xl backdrop-blur-sm">
          {step === "details" ? (
            <>
              <div className="mb-6 text-center">
                <h1 className="text-2xl font-semibold text-foreground">Create your account</h1>
                <p className="mt-2 text-sm text-muted-foreground">
                  Get started with SchemaLabs for free
                </p>
              </div>

              {error && (
                <div className="mb-4 rounded-lg bg-destructive/10 p-3 text-sm text-destructive">
                  {error}
                </div>
              )}

              <form onSubmit={handleSubmitDetails} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="name" className="text-foreground">Full Name</Label>
                  <Input
                    id="name"
                    type="text"
                    placeholder="John Doe"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    className="h-11 border-border bg-background text-foreground placeholder:text-muted-foreground"
                    required
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="email" className="text-foreground">Work Email</Label>
                  <Input
                    id="email"
                    type="email"
                    placeholder="you@company.com"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="h-11 border-border bg-background text-foreground placeholder:text-muted-foreground"
                    required
                  />
                  <p className="text-xs text-muted-foreground">Use your company email for full access</p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="password" className="text-foreground">Password</Label>
                  <div className="relative">
                    <Input
                      id="password"
                      type={showPassword ? "text" : "password"}
                      placeholder="Create a password"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      className="h-11 border-border bg-background pr-10 text-foreground placeholder:text-muted-foreground"
                      required
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                    >
                      {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </button>
                  </div>
                  {password && (
                    <div className="space-y-1 pt-1">
                      {passwordRequirements.map((req, i) => (
                        <div key={i} className="flex items-center gap-2 text-xs">
                          <div className={`h-1.5 w-1.5 rounded-full ${req.met ? "bg-green-500" : "bg-muted-foreground"}`} />
                          <span className={req.met ? "text-green-500" : "text-muted-foreground"}>{req.label}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                <div className="space-y-2">
                  <Label htmlFor="confirmPassword" className="text-foreground">Confirm Password</Label>
                  <div className="relative">
                    <Input
                      id="confirmPassword"
                      type={showConfirmPassword ? "text" : "password"}
                      placeholder="Confirm your password"
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      className="h-11 border-border bg-background pr-10 text-foreground placeholder:text-muted-foreground"
                      required
                    />
                    <button
                      type="button"
                      onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                    >
                      {showConfirmPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </button>
                  </div>
                  {confirmPassword && (
                    <div className="flex items-center gap-2 pt-1 text-xs">
                      {doPasswordsMatch ? (
                        <>
                          <Check className="h-3 w-3 text-green-500" />
                          <span className="text-green-500">Passwords match</span>
                        </>
                      ) : (
                        <>
                          <div className="h-1.5 w-1.5 rounded-full bg-destructive" />
                          <span className="text-destructive">Passwords do not match</span>
                        </>
                      )}
                    </div>
                  )}
                </div>

                <Button
                  type="submit"
                  className="h-11 w-full bg-[#0052CC] text-white hover:bg-[#003D99]"
                  disabled={isLoading}
                >
                  {isLoading ? (
                    <div className="flex items-center gap-2">
                      <div className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                      Creating account...
                    </div>
                  ) : (
                    "Create account"
                  )}
                </Button>
              </form>

              <div className="relative my-6">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t border-border" />
                </div>
                <div className="relative flex justify-center text-xs">
                  <span className="bg-card px-2 text-muted-foreground">OR CONTINUE WITH</span>
                </div>
              </div>

              <Button
                type="button"
                variant="outline"
                className="h-11 w-full gap-3 border-border bg-transparent text-foreground hover:bg-muted"
                onClick={handleGoogleSignUp}
                disabled={isLoading}
              >
                <svg className="h-5 w-5" viewBox="0 0 24 24">
                  <path
                    fill="currentColor"
                    d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                  />
                  <path
                    fill="currentColor"
                    d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                  />
                  <path
                    fill="currentColor"
                    d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                  />
                  <path
                    fill="currentColor"
                    d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                  />
                </svg>
                Continue with Google
              </Button>

              <p className="mt-6 text-center text-sm text-muted-foreground">
                Already have an account?{" "}
                <Link 
                  href="/login" 
                  className="font-medium text-[#0052CC] hover:text-[#2684FF] dark:text-[#2684FF] dark:hover:text-[#4C9AFF]"
                >
                  Sign in
                </Link>
              </p>
            </>
          ) : (
            <>
              <button
                onClick={() => setStep("details")}
                className="mb-4 flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground"
              >
                <ArrowLeft className="h-4 w-4" />
                Back
              </button>

              <div className="mb-6 text-center">
                <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-[#0052CC]/10">
                  <svg className="h-6 w-6 text-[#0052CC] dark:text-[#2684FF]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                  </svg>
                </div>
                <h1 className="text-2xl font-semibold text-foreground">Check your email</h1>
                <p className="mt-2 text-sm text-muted-foreground">
                  We've sent a verification code to
                </p>
                <p className="mt-1 text-sm font-medium text-foreground">{email}</p>
              </div>

              {error && (
                <div className="mb-4 rounded-lg bg-destructive/10 p-3 text-sm text-destructive">
                  {error}
                </div>
              )}

              <div className="space-y-4">
                <div className="space-y-2">
                  <Label className="text-foreground">Verification Code</Label>
                  <div className="flex gap-2 justify-center">
                    {verificationCode.map((digit, index) => (
                      <Input
                        key={index}
                        id={`code-${index}`}
                        type="text"
                        inputMode="numeric"
                        maxLength={1}
                        value={digit}
                        onChange={(e) => handleVerificationInput(index, e.target.value)}
                        onKeyDown={(e) => handleKeyDown(index, e)}
                        className="h-12 w-12 text-center text-lg font-semibold border-border bg-background text-foreground"
                      />
                    ))}
                  </div>
                </div>

                <Button
                  onClick={handleVerifyCode}
                  className="h-11 w-full bg-[#0052CC] text-white hover:bg-[#003D99]"
                  disabled={isLoading || verificationCode.join("").length !== 6}
                >
                  {isLoading ? (
                    <div className="flex items-center gap-2">
                      <div className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                      Verifying...
                    </div>
                  ) : (
                    "Verify email"
                  )}
                </Button>

                <p className="text-center text-sm text-muted-foreground">
                  Didn't receive the code?{" "}
                  <button
                    onClick={handleResendCode}
                    className="font-medium text-[#0052CC] hover:text-[#2684FF] dark:text-[#2684FF] dark:hover:text-[#4C9AFF]"
                    disabled={isLoading}
                  >
                    Resend
                  </button>
                </p>
              </div>
            </>
          )}
        </div>

        {/* Alpha Badge */}
        <div className="mt-6 flex items-center gap-2 rounded-full border border-border bg-card/50 px-4 py-2 text-xs text-muted-foreground backdrop-blur-sm">
          <div className="h-2 w-2 rounded-full bg-[#0052CC] animate-pulse" />
          Alpha Version
        </div>

        {/* Terms */}
        <p className="mt-4 max-w-sm text-center text-xs text-muted-foreground">
          By creating an account, you agree to our{" "}
          <Link href="https://www.schemalabs.ai/terms" target="_blank" className="underline hover:text-foreground">Terms of Service</Link>
          {" "}and{" "}
          <Link href="https://www.schemalabs.ai/privacy" target="_blank" className="underline hover:text-foreground">Privacy Policy</Link>
        </p>
      </div>
    </div>
  )
}

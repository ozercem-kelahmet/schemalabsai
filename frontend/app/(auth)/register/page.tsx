"use client"

import React from "react"
import { useState, useEffect } from "react"
import Link from "next/link"
import Image from "next/image"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Eye, EyeOff, Loader2, ArrowLeft } from "lucide-react"

type Step = "signup" | "verify"

export default function RegisterPage() {
  const [step, setStep] = useState<Step>("signup")
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [name, setName] = useState("")
  const [verificationCode, setVerificationCode] = useState("")
  const [showPassword, setShowPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")
  const [success, setSuccess] = useState("")
  const [isDark, setIsDark] = useState(true)

  useEffect(() => {
    const savedTheme = localStorage.getItem("schemalabs-theme")
    setIsDark(savedTheme !== "light")
  }, [])

  const handleSendVerificationCode = async () => {
    setIsLoading(true)
    setError("")
    
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

      setSuccess("Verification code sent to your email")
      setStep("verify")
    } catch {
      setError("Something went wrong")
    }
    setIsLoading(false)
  }

  const handleVerifyAndSignup = async () => {
    if (password !== confirmPassword) {
      setError("Passwords do not match")
      return
    }

    setIsLoading(true)
    setError("")
    
    try {
      const res = await fetch("/api/auth/verify-signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, code: verificationCode, name, password }),
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

  const handleSignupStart = async (e: React.FormEvent) => {
    e.preventDefault()
    setError("")

    if (!name.trim()) {
      setError("Name is required")
      return
    }
    if (!email.trim()) {
      setError("Email is required")
      return
    }
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      setError("Please enter a valid email address")
      return
    }
    if (!password) {
      setError("Password is required")
      return
    }
    if (password.length < 6) {
      setError("Password must be at least 6 characters")
      return
    }
    if (password !== confirmPassword) {
      setError("Passwords do not match")
      return
    }

    await handleSendVerificationCode()
  }

  const handleGoogleSignIn = () => {
    setIsLoading(true)
    window.location.href = "/api/google/login"
  }

  return (
    <div className="relative min-h-screen w-full overflow-hidden bg-background">
      <div className="absolute inset-0 bg-[linear-gradient(to_right,hsl(var(--border))_1px,transparent_1px),linear-gradient(to_bottom,hsl(var(--border))_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_80%_50%_at_50%_0%,#000_70%,transparent_110%)]" />
      <div className="absolute inset-0 bg-gradient-to-b from-[#0052CC]/5 via-transparent to-transparent" />
      
      <div className="relative flex min-h-screen flex-col items-center justify-center px-4">
        <Link href="/" className="mb-8">
          <Image
            src={isDark ? "/images/schemalabs-light.png" : "/images/schemalabs-dark.png"}
            alt="SchemaLabs"
            width={180}
            height={40}
            className="h-10 w-auto"
            priority
          />
        </Link>

        <div className="w-full max-w-[400px] rounded-2xl border border-border bg-card/80 p-8 shadow-xl backdrop-blur-sm">
          {step === "signup" && (
            <>
              <div className="mb-6 text-center">
                <h1 className="text-2xl font-semibold text-foreground">Create account</h1>
                <p className="mt-2 text-sm text-muted-foreground">
                  Get started with SchemaLabs
                </p>
              </div>

              {error && (
                <div className="mb-4 rounded-lg bg-destructive/10 p-3 text-sm text-destructive">
                  {error}
                </div>
              )}

              <form onSubmit={handleSignupStart} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="name" className="text-foreground">Name</Label>
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

                <div className="space-y-2">
                  <Label htmlFor="password" className="text-foreground">Password</Label>
                  <div className="relative">
                    <Input
                      id="password"
                      type={showPassword ? "text" : "password"}
                      placeholder="Min 6 characters"
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
                onClick={handleGoogleSignIn}
                disabled={isLoading}
              >
                <svg className="h-5 w-5" viewBox="0 0 24 24">
                  <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
                  <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
                  <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" />
                  <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
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
          )}

          {step === "verify" && (
            <>
              <button
                type="button"
                onClick={() => setStep("signup")}
                className="mb-4 flex items-center text-sm text-muted-foreground hover:text-foreground"
              >
                <ArrowLeft className="h-4 w-4 mr-1" />
                Back
              </button>

              <div className="mb-6 text-center">
                <h1 className="text-2xl font-semibold text-foreground">Verify email</h1>
                <p className="mt-2 text-sm text-muted-foreground">
                  Enter the 6-digit code sent to <span className="font-medium text-foreground">{email}</span>
                </p>
              </div>

              {error && (
                <div className="mb-4 rounded-lg bg-destructive/10 p-3 text-sm text-destructive">
                  {error}
                </div>
              )}

              {success && (
                <div className="mb-4 rounded-lg bg-green-500/10 p-3 text-sm text-green-600">
                  {success}
                </div>
              )}

              <div className="space-y-4">
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
                  className="h-11 w-full bg-[#0052CC] text-white hover:bg-[#003D99]"
                  disabled={isLoading || verificationCode.length !== 6}
                  onClick={handleVerifyAndSignup}
                >
                  {isLoading ? (
                    <div className="flex items-center gap-2">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Verifying...
                    </div>
                  ) : (
                    "Verify & Create Account"
                  )}
                </Button>

                <p className="text-center text-sm text-muted-foreground">
                  Didn't receive the code?{" "}
                  <button
                    type="button"
                    onClick={handleSendVerificationCode}
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

        <div className="mt-6 flex items-center gap-2 rounded-full border border-border bg-card/50 px-4 py-2 text-xs text-muted-foreground backdrop-blur-sm">
          <div className="h-2 w-2 rounded-full bg-[#0052CC] animate-pulse" />
          Alpha Version
        </div>
      </div>
    </div>
  )
}

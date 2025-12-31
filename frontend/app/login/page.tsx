"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Mail, Lock, User, ArrowRight, Loader2, Database, BarChart3, Zap, ArrowLeft } from "lucide-react"

type Step = "login" | "signup" | "verify" | "forgot" | "forgot-verify" | "reset-password"

export default function LoginPage() {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")
  const [success, setSuccess] = useState("")
  const [step, setStep] = useState<Step>("login")
  
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [name, setName] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [verificationCode, setVerificationCode] = useState("")
  const [newPassword, setNewPassword] = useState("")
  const [confirmNewPassword, setConfirmNewPassword] = useState("")

  const handleGoogleSignIn = () => {
    window.location.href = "http://localhost:8080/api/google/login"
  }

  const handleSendVerificationCode = async () => {
    setIsLoading(true)
    setError("")
    
    try {
      const res = await fetch("http://localhost:8080/api/auth/send-verification", {
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
      const res = await fetch("http://localhost:8080/api/auth/verify-signup", {
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
      window.location.href = "/data-sources"
    } catch {
      setError("Something went wrong")
    }
    setIsLoading(false)
  }

  const handleRequestPasswordReset = async () => {
    setError("")

    // Validation
    if (!email.trim()) {
      setError("Email is required")
      return
    }
    if (email.includes(" ")) {
      setError("Email cannot contain spaces")
      return
    }
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      setError("Please enter a valid email address")
      return
    }

    setIsLoading(true)
    
    try {
      const res = await fetch("http://localhost:8080/api/auth/request-reset", {
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

      setSuccess("Reset code sent to your email")
      setStep("forgot-verify")
    } catch {
      setError("Something went wrong")
    }
    setIsLoading(false)
  }

  const handleVerifyResetCode = async () => {
    setIsLoading(true)
    setError("")
    
    try {
      const res = await fetch("http://localhost:8080/api/auth/verify-reset-code", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, code: verificationCode }),
      })

      const data = await res.json()
      
      if (!res.ok) {
        setError(data.error || "Invalid code")
        setIsLoading(false)
        return
      }

      setStep("reset-password")
    } catch {
      setError("Something went wrong")
    }
    setIsLoading(false)
  }

  const handleResetPassword = async () => {
    if (newPassword !== confirmNewPassword) {
      setError("Passwords do not match")
      return
    }

    if (newPassword.length < 6) {
      setError("Password must be at least 6 characters")
      return
    }

    setIsLoading(true)
    setError("")
    
    try {
      const res = await fetch("http://localhost:8080/api/auth/reset-password", {
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

      setSuccess("Password reset successfully! Please sign in.")
      setStep("login")
      setPassword("")
      setVerificationCode("")
      setNewPassword("")
      setConfirmNewPassword("")
    } catch {
      setError("Something went wrong")
    }
    setIsLoading(false)
  }

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    setError("")

    // Validation
    if (!email.trim()) {
      setError("Email is required")
      return
    }
    if (!password) {
      setError("Password is required")
      return
    }
    if (email.includes(" ")) {
      setError("Email cannot contain spaces")
      return
    }
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      setError("Please enter a valid email address")
      return
    }

    setIsLoading(true)

    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ email, password }),
      })

      if (!res.ok) {
        const data = await res.json()
        setError(data.error || "Invalid credentials")
        setIsLoading(false)
        return
      }

      window.location.href = "/data-sources"
    } catch {
      setError("Something went wrong")
      setIsLoading(false)
    }
  }

  const handleSignupStart = async (e: React.FormEvent) => {
    e.preventDefault()
    setError("")

    // Validation
    if (!name.trim()) {
      setError("Name is required")
      return
    }
    if (!email.trim()) {
      setError("Email is required")
      return
    }
    if (email.includes(" ")) {
      setError("Email cannot contain spaces")
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

  const resetForm = () => {
    setError("")
    setSuccess("")
    setVerificationCode("")
  }

  return (
    <div className="min-h-screen flex bg-background">
      {/* Left Side - Branding */}
      <div className="hidden lg:flex lg:w-1/2 bg-foreground p-12 flex-col justify-between">
        <div>
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-background rounded-lg flex items-center justify-center">
              <Database className="w-6 h-6 text-foreground" />
            </div>
            <span className="text-2xl font-bold text-background">Schema Labs</span>
          </div>
        </div>
        
        <div className="space-y-8">
          <h1 className="text-4xl font-bold text-background leading-tight">
            Transform your data into insights with AI
          </h1>
          <p className="text-background/70 text-lg">
            Upload any dataset and let our fine-tuned models analyze, predict, and explain your data in natural language.
          </p>
          
          <div className="space-y-4">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-background/10 rounded-lg flex items-center justify-center">
                <BarChart3 className="w-6 h-6 text-background" />
              </div>
              <div>
                <h3 className="text-background font-semibold">Instant Analysis</h3>
                <p className="text-background/60 text-sm">Get insights from any CSV in seconds</p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-background/10 rounded-lg flex items-center justify-center">
                <Zap className="w-6 h-6 text-background" />
              </div>
              <div>
                <h3 className="text-background font-semibold">200M+ Rows Trained</h3>
                <p className="text-background/60 text-sm">Foundation model trained on 200M+ rows of tabular data</p>
              </div>
            </div>
          </div>
        </div>

        <p className="text-background/40 text-sm">
          © 2024 Schema Labs. All rights reserved.
        </p>
      </div>

      {/* Right Side - Auth Form */}
      <div className="w-full lg:w-1/2 flex items-center justify-center p-8">
        <div className="w-full max-w-md space-y-8">
          {/* Mobile Logo */}
          <div className="lg:hidden flex items-center justify-center gap-3 mb-8">
            <div className="w-10 h-10 bg-foreground rounded-lg flex items-center justify-center">
              <Database className="w-6 h-6 text-background" />
            </div>
            <span className="text-2xl font-bold">Schema Labs</span>
          </div>

          {/* Login Form */}
          {step === "login" && (
            <>
              <div className="text-center lg:text-left">
                <h2 className="text-3xl font-bold">Welcome back</h2>
                <p className="text-muted-foreground mt-2">Sign in to your account</p>
              </div>

              {error && (
                <div className="bg-destructive/10 text-destructive text-sm p-3 rounded-lg">
                  {error}
                </div>
              )}

              {success && (
                <div className="bg-green-500/10 text-green-600 text-sm p-3 rounded-lg">
                  {success}
                </div>
              )}

              <form onSubmit={handleLogin} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="email">Email</Label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      id="email"
                      type="email"
                      placeholder="you@example.com"
                      className="pl-10"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      required
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label htmlFor="password">Password</Label>
                    <button
                      type="button"
                      onClick={() => { resetForm(); setStep("forgot") }}
                      className="text-sm text-primary hover:underline"
                    >
                      Forgot password?
                    </button>
                  </div>
                  <div className="relative">
                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      id="password"
                      type="password"
                      placeholder="••••••••"
                      className="pl-10"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      required
                    />
                  </div>
                </div>

                <Button type="submit" className="w-full" disabled={isLoading}>
                  {isLoading ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <>
                      Sign In
                      <ArrowRight className="ml-2 h-4 w-4" />
                    </>
                  )}
                </Button>
              </form>

              <div className="relative">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t"></div>
                </div>
                <div className="relative flex justify-center text-xs uppercase">
                  <span className="bg-background px-2 text-muted-foreground">Or continue with</span>
                </div>
              </div>

              <Button 
                variant="outline" 
                className="w-full" 
                onClick={handleGoogleSignIn}
              >
                <svg className="mr-2 h-4 w-4" viewBox="0 0 24 24">
                  <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4" />
                  <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853" />
                  <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05" />
                  <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335" />
                </svg>
                Continue with Google
              </Button>

              <p className="text-center text-sm text-muted-foreground">
                Don&apos;t have an account?{" "}
                <button
                  type="button"
                  onClick={() => { resetForm(); setStep("signup") }}
                  className="text-primary hover:underline font-medium"
                >
                  Sign up
                </button>
              </p>
            </>
          )}

          {/* Signup Form */}
          {step === "signup" && (
            <>
              <div className="text-center lg:text-left">
                <h2 className="text-3xl font-bold">Create account</h2>
                <p className="text-muted-foreground mt-2">Enter your details to get started</p>
              </div>

              {error && (
                <div className="bg-destructive/10 text-destructive text-sm p-3 rounded-lg">
                  {error}
                </div>
              )}

              <form onSubmit={handleSignupStart} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="signup-name">Name</Label>
                  <div className="relative">
                    <User className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      id="signup-name"
                      type="text"
                      placeholder="John Doe"
                      className="pl-10"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      required
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="signup-email">Email</Label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      id="signup-email"
                      type="email"
                      placeholder="you@example.com"
                      className="pl-10"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      required
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="signup-password">Password</Label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      id="signup-password"
                      type="password"
                      placeholder="••••••••"
                      className="pl-10"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      required
                      minLength={6}
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="signup-confirm">Confirm Password</Label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      id="signup-confirm"
                      type="password"
                      placeholder="••••••••"
                      className="pl-10"
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      required
                    />
                  </div>
                </div>

                <Button type="submit" className="w-full" disabled={isLoading}>
                  {isLoading ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <>
                      Send Verification Code
                      <ArrowRight className="ml-2 h-4 w-4" />
                    </>
                  )}
                </Button>
              </form>

              <div className="relative">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t"></div>
                </div>
                <div className="relative flex justify-center text-xs uppercase">
                  <span className="bg-background px-2 text-muted-foreground">Or continue with</span>
                </div>
              </div>

              <Button variant="outline" className="w-full" onClick={handleGoogleSignIn}>
                <svg className="mr-2 h-4 w-4" viewBox="0 0 24 24">
                  <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4" />
                  <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853" />
                  <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05" />
                  <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335" />
                </svg>
                Continue with Google
              </Button>

              <p className="text-center text-sm text-muted-foreground">
                Already have an account?{" "}
                <button
                  type="button"
                  onClick={() => { resetForm(); setStep("login") }}
                  className="text-primary hover:underline font-medium"
                >
                  Sign in
                </button>
              </p>
            </>
          )}

          {/* Verify Signup Code */}
          {step === "verify" && (
            <>
              <button
                type="button"
                onClick={() => setStep("signup")}
                className="flex items-center text-sm text-muted-foreground hover:text-foreground"
              >
                <ArrowLeft className="h-4 w-4 mr-1" />
                Back
              </button>

              <div className="text-center lg:text-left">
                <h2 className="text-3xl font-bold">Verify email</h2>
                <p className="text-muted-foreground mt-2">
                  Enter the 6-digit code sent to <span className="font-medium text-foreground">{email}</span>
                </p>
              </div>

              {error && (
                <div className="bg-destructive/10 text-destructive text-sm p-3 rounded-lg">
                  {error}
                </div>
              )}

              {success && (
                <div className="bg-green-500/10 text-green-600 text-sm p-3 rounded-lg">
                  {success}
                </div>
              )}

              <div className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="code">Verification Code</Label>
                  <Input
                    id="code"
                    type="text"
                    placeholder="000000"
                    className="text-center text-2xl tracking-[0.5em] font-mono"
                    value={verificationCode}
                    onChange={(e) => setVerificationCode(e.target.value.replace(/\D/g, "").slice(0, 6))}
                    maxLength={6}
                    required
                  />
                </div>

                <Button 
                  className="w-full" 
                  disabled={isLoading || verificationCode.length !== 6}
                  onClick={handleVerifyAndSignup}
                >
                  {isLoading ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <>
                      Verify & Create Account
                      <ArrowRight className="ml-2 h-4 w-4" />
                    </>
                  )}
                </Button>

                <p className="text-center text-sm text-muted-foreground">
                  Didn&apos;t receive the code?{" "}
                  <button
                    type="button"
                    onClick={handleSendVerificationCode}
                    className="text-primary hover:underline font-medium"
                    disabled={isLoading}
                  >
                    Resend
                  </button>
                </p>
              </div>
            </>
          )}

          {/* Forgot Password - Enter Email */}
          {step === "forgot" && (
            <>
              <button
                type="button"
                onClick={() => { resetForm(); setStep("login") }}
                className="flex items-center text-sm text-muted-foreground hover:text-foreground"
              >
                <ArrowLeft className="h-4 w-4 mr-1" />
                Back to login
              </button>

              <div className="text-center lg:text-left">
                <h2 className="text-3xl font-bold">Reset password</h2>
                <p className="text-muted-foreground mt-2">
                  Enter your email to receive a reset code
                </p>
              </div>

              {error && (
                <div className="bg-destructive/10 text-destructive text-sm p-3 rounded-lg">
                  {error}
                </div>
              )}

              <div className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="reset-email">Email</Label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      id="reset-email"
                      type="email"
                      placeholder="you@example.com"
                      className="pl-10"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      required
                    />
                  </div>
                </div>

                <Button 
                  className="w-full" 
                  disabled={isLoading || !email}
                  onClick={handleRequestPasswordReset}
                >
                  {isLoading ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <>
                      Send Reset Code
                      <ArrowRight className="ml-2 h-4 w-4" />
                    </>
                  )}
                </Button>
              </div>
            </>
          )}

          {/* Forgot Password - Verify Code */}
          {step === "forgot-verify" && (
            <>
              <button
                type="button"
                onClick={() => setStep("forgot")}
                className="flex items-center text-sm text-muted-foreground hover:text-foreground"
              >
                <ArrowLeft className="h-4 w-4 mr-1" />
                Back
              </button>

              <div className="text-center lg:text-left">
                <h2 className="text-3xl font-bold">Enter reset code</h2>
                <p className="text-muted-foreground mt-2">
                  Enter the 6-digit code sent to <span className="font-medium text-foreground">{email}</span>
                </p>
              </div>

              {error && (
                <div className="bg-destructive/10 text-destructive text-sm p-3 rounded-lg">
                  {error}
                </div>
              )}

              {success && (
                <div className="bg-green-500/10 text-green-600 text-sm p-3 rounded-lg">
                  {success}
                </div>
              )}

              <div className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="reset-code">Reset Code</Label>
                  <Input
                    id="reset-code"
                    type="text"
                    placeholder="000000"
                    className="text-center text-2xl tracking-[0.5em] font-mono"
                    value={verificationCode}
                    onChange={(e) => setVerificationCode(e.target.value.replace(/\D/g, "").slice(0, 6))}
                    maxLength={6}
                    required
                  />
                </div>

                <Button 
                  className="w-full" 
                  disabled={isLoading || verificationCode.length !== 6}
                  onClick={handleVerifyResetCode}
                >
                  {isLoading ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <>
                      Verify Code
                      <ArrowRight className="ml-2 h-4 w-4" />
                    </>
                  )}
                </Button>

                <p className="text-center text-sm text-muted-foreground">
                  Didn&apos;t receive the code?{" "}
                  <button
                    type="button"
                    onClick={handleRequestPasswordReset}
                    className="text-primary hover:underline font-medium"
                    disabled={isLoading}
                  >
                    Resend
                  </button>
                </p>
              </div>
            </>
          )}

          {/* Reset Password - New Password */}
          {step === "reset-password" && (
            <>
              <div className="text-center lg:text-left">
                <h2 className="text-3xl font-bold">Set new password</h2>
                <p className="text-muted-foreground mt-2">
                  Enter your new password below
                </p>
              </div>

              {error && (
                <div className="bg-destructive/10 text-destructive text-sm p-3 rounded-lg">
                  {error}
                </div>
              )}

              <div className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="new-password">New Password</Label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      id="new-password"
                      type="password"
                      placeholder="••••••••"
                      className="pl-10"
                      value={newPassword}
                      onChange={(e) => setNewPassword(e.target.value)}
                      required
                      minLength={6}
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="confirm-new-password">Confirm New Password</Label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      id="confirm-new-password"
                      type="password"
                      placeholder="••••••••"
                      className="pl-10"
                      value={confirmNewPassword}
                      onChange={(e) => setConfirmNewPassword(e.target.value)}
                      required
                    />
                  </div>
                </div>

                <Button 
                  className="w-full" 
                  disabled={isLoading || !newPassword || !confirmNewPassword}
                  onClick={handleResetPassword}
                >
                  {isLoading ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <>
                      Reset Password
                      <ArrowRight className="ml-2 h-4 w-4" />
                    </>
                  )}
                </Button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

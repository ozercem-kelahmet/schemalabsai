"use client"

import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Sidebar } from "@/components/sidebar"
import { Textarea } from "@/components/ui/textarea"
import { useTheme } from "next-themes"
import { 
  User, Camera, Check, Loader2, LogOut, Copy, Key, Eye, EyeOff,
  Monitor, Smartphone, MoreHorizontal, ChevronLeft, ChevronRight, Trash2, AlertTriangle, X
} from "lucide-react"

interface UserProfile {
  id: string
  name: string
  email: string
  avatar_url?: string
  created_at: string
}

interface Session {
  id: string
  device: string
  device_type: string
  location: string
  created_at: string
  updated_at: string
  is_current: boolean
}

interface Toast {
  id: number
  message: string
  type: "success" | "error"
}

export default function AccountPage() {
  const { theme, setTheme, resolvedTheme } = useTheme()
  const [mounted, setMounted] = useState(false)
  const [user, setUser] = useState<UserProfile | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [isSaving, setIsSaving] = useState(false)
  const [toasts, setToasts] = useState<Toast[]>([])
  
  // Profile
  const [editedName, setEditedName] = useState("")
  const [nickname, setNickname] = useState("")
  const [workDescription, setWorkDescription] = useState("Engineering")
  const [personalPreferences, setPersonalPreferences] = useState("")
  const [avatarPreview, setAvatarPreview] = useState<string | null>(null)
  const [isUploadingAvatar, setIsUploadingAvatar] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  
  // Notifications
  const [notifyTrainingComplete, setNotifyTrainingComplete] = useState(true)
  const [notifyApiUsage, setNotifyApiUsage] = useState(false)
  const [notifyNewFeatures, setNotifyNewFeatures] = useState(true)
  const [notifyEmail, setNotifyEmail] = useState(false)
  
  // Appearance
  const [currentFont, setCurrentFont] = useState("default")
  
  // Account
  const [copiedId, setCopiedId] = useState(false)
  
  // Delete Account
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  const [deleteConfirmText, setDeleteConfirmText] = useState("")
  const [isDeleting, setIsDeleting] = useState(false)
  
  // Password
  const [showPasswordChange, setShowPasswordChange] = useState(false)
  const [passwordStep, setPasswordStep] = useState<"request" | "verify">("request")
  const [verificationCode, setVerificationCode] = useState("")
  const [newPassword, setNewPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [showNewPassword, setShowNewPassword] = useState(false)
  const [passwordLoading, setPasswordLoading] = useState(false)
  const [passwordError, setPasswordError] = useState("")
  const [passwordSuccess, setPasswordSuccess] = useState("")

  // Sessions
  const [sessions, setSessions] = useState<Session[]>([])
  const [sessionPage, setSessionPage] = useState(1)
  const [totalSessions, setTotalSessions] = useState(0)
  const sessionsPerPage = 10
  const totalSessionPages = Math.ceil(totalSessions / sessionsPerPage)

  const showToast = (message: string, type: "success" | "error" = "success") => {
    const id = Date.now()
    setToasts(prev => [...prev, { id, message, type }])
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id))
    }, 4000)
  }

  const removeToast = (id: number) => {
    setToasts(prev => prev.filter(t => t.id !== id))
  }

  useEffect(() => {
    setMounted(true)
    fetchUserProfile()
    fetchSessions()
    
    const savedFont = localStorage.getItem("schemalabs-font") || "default"
    setCurrentFont(savedFont)
    applyFont(savedFont)
  }, [])

  useEffect(() => {
    fetchSessions()
  }, [sessionPage])

  const applyFont = (fontValue: string) => {
    document.documentElement.setAttribute("data-font", fontValue)
    const styles: Record<string, { fontFamily: string; letterSpacing: string }> = {
      default: { fontFamily: "ui-serif, Georgia, serif", letterSpacing: "normal" },
      sans: { fontFamily: "ui-sans-serif, system-ui, sans-serif", letterSpacing: "normal" },
      system: { fontFamily: "system-ui, -apple-system, sans-serif", letterSpacing: "normal" },
      dyslexic: { fontFamily: "OpenDyslexic, Comic Sans MS, sans-serif", letterSpacing: "0.05em" }
    }
    const style = styles[fontValue] || styles.default
    document.body.style.fontFamily = style.fontFamily
    document.body.style.letterSpacing = style.letterSpacing
  }

  const handleFontChange = (fontValue: string) => {
    setCurrentFont(fontValue)
    localStorage.setItem("schemalabs-font", fontValue)
    applyFont(fontValue)
    showToast("Font updated")
  }

  const handleThemeChange = (newTheme: string) => {
    setTheme(newTheme)
    showToast("Theme updated")
  }

  const fetchUserProfile = async () => {
    try {
      const res = await fetch("/api/auth/me", { credentials: "include" })
      if (res.ok) {
        const data = await res.json()
        const userData = data.user || data
        if (userData) {
          setUser(userData)
          setEditedName(userData.name || "")
          setNickname(userData.name?.split(" ")[0] || "")
          if (userData.avatar_url) {
            setAvatarPreview("http://localhost:8080" + userData.avatar_url)
          }
        }
      }
    } catch (error) {
      console.error("Failed to fetch user:", error)
    }
    setIsLoading(false)
  }

  const fetchSessions = async () => {
    try {
      const res = await fetch("/api/auth/sessions?page=" + sessionPage + "&limit=" + sessionsPerPage, { credentials: "include" })
      if (res.ok) {
        const data = await res.json()
        setSessions(data.sessions || [])
        setTotalSessions(data.total || 0)
      }
    } catch (error) {
      console.error("Failed to fetch sessions:", error)
    }
  }

  const handleSaveProfile = async () => {
    if (!user) return
    setIsSaving(true)
    try {
      const res = await fetch("/api/auth/update-profile", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ name: editedName })
      })
      if (res.ok) {
        setUser(prev => prev ? { ...prev, name: editedName } : null)
        showToast("Profile updated successfully")
      } else {
        showToast("Failed to update profile", "error")
      }
    } catch (error) {
      showToast("Failed to update profile", "error")
    }
    setIsSaving(false)
  }

  const handleAvatarChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (e) => setAvatarPreview(e.target?.result as string)
    reader.readAsDataURL(file)
    setIsUploadingAvatar(true)
    try {
      const formData = new FormData()
      formData.append("avatar", file)
      const res = await fetch("/api/auth/upload-avatar", { method: "POST", credentials: "include", body: formData })
      if (res.ok) {
        const data = await res.json()
        setAvatarPreview("http://localhost:8080" + data.avatar_url)
        showToast("Avatar updated successfully")
      } else {
        showToast("Failed to upload avatar", "error")
      }
    } catch (error) {
      showToast("Failed to upload avatar", "error")
    }
    setIsUploadingAvatar(false)
  }

  const handleDeleteAccount = async () => {
    if (deleteConfirmText !== "DELETE") return
    setIsDeleting(true)
    try {
      const res = await fetch("/api/auth/delete-account", { method: "DELETE", credentials: "include" })
      if (res.ok) {
        showToast("Account deleted successfully")
        setTimeout(() => {
          window.location.href = "/login"
        }, 1500)
      } else {
        showToast("Failed to delete account", "error")
        setIsDeleting(false)
      }
    } catch (error) {
      showToast("Failed to delete account", "error")
      setIsDeleting(false)
    }
  }

  const handleRequestPasswordChange = async () => {
    setPasswordLoading(true)
    setPasswordError("")
    try {
      const res = await fetch("/api/auth/change-password-request", { method: "POST", credentials: "include" })
      if (res.ok) {
        setPasswordStep("verify")
        setPasswordSuccess("Verification code sent to your email")
      } else {
        setPasswordError("Failed to send verification code")
      }
    } catch (error) {
      setPasswordError("Failed to send verification code")
    }
    setPasswordLoading(false)
  }

  const handleVerifyAndChangePassword = async () => {
    if (newPassword !== confirmPassword) { setPasswordError("Passwords do not match"); return }
    if (newPassword.length < 6) { setPasswordError("Password must be at least 6 characters"); return }
    setPasswordLoading(true)
    setPasswordError("")
    try {
      const res = await fetch("/api/auth/change-password-verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ code: verificationCode, new_password: newPassword })
      })
      if (res.ok) {
        showToast("Password changed successfully")
        setShowPasswordChange(false)
        resetPasswordForm()
      } else {
        setPasswordError("Invalid verification code")
      }
    } catch (error) {
      setPasswordError("Failed to change password")
    }
    setPasswordLoading(false)
  }

  const resetPasswordForm = () => {
    setPasswordStep("request")
    setVerificationCode("")
    setNewPassword("")
    setConfirmPassword("")
    setPasswordError("")
    setPasswordSuccess("")
  }

  const handleLogoutAllDevices = async () => {
    try {
      const res = await fetch("/api/auth/logout-all", { method: "POST", credentials: "include" })
      if (res.ok) {
        fetchSessions()
        showToast("Logged out from all other devices")
      } else {
        showToast("Failed to logout", "error")
      }
    } catch (error) {
      showToast("Failed to logout", "error")
    }
  }

  const copyUserId = () => {
    if (user?.id) {
      navigator.clipboard.writeText(user.id)
      setCopiedId(true)
      showToast("User ID copied to clipboard")
      setTimeout(() => setCopiedId(false), 2000)
    }
  }

  const formatDate = (dateStr: string) => {
    if (!dateStr) return "N/A"
    return new Date(dateStr).toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })
  }

  const Toggle = ({ enabled, onChange }: { enabled: boolean; onChange: (v: boolean) => void }) => (
    <button
      onClick={() => onChange(!enabled)}
      className={`relative w-[42px] h-[24px] rounded-full transition-colors duration-200 ${
        enabled ? "bg-[#4B9EFF]" : "bg-[#D1D5DB]"
      }`}
    >
      <div
        className={`absolute top-[2px] w-[20px] h-[20px] rounded-full bg-white shadow-sm transition-transform duration-200 ${
          enabled ? "left-[20px]" : "left-[2px]"
        }`}
      />
    </button>
  )

  if (isLoading || !mounted) {
    return <Sidebar><div className="flex-1 flex items-center justify-center"><Loader2 className="w-6 h-6 animate-spin" /></div></Sidebar>
  }

  return (
    <Sidebar>
      <div className="flex-1 overflow-auto bg-background">
        <div className="max-w-[680px] mx-auto py-8 px-6">
          
          {/* Toast Notifications */}
          <div className="fixed top-4 right-4 z-50 space-y-2">
            {toasts.map((toast) => (
              <div
                key={toast.id}
                className={`flex items-center gap-3 px-4 py-3 rounded-lg shadow-lg animate-in slide-in-from-right ${
                  toast.type === "success" ? "bg-green-600 text-white" : "bg-red-600 text-white"
                }`}
              >
                {toast.type === "success" ? <Check className="w-4 h-4" /> : <AlertTriangle className="w-4 h-4" />}
                <span className="text-sm font-medium">{toast.message}</span>
                <button onClick={() => removeToast(toast.id)} className="ml-2 hover:opacity-70">
                  <X className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
          
          {/* Profile Section */}
          <section className="mb-12">
            <h2 className="text-[15px] font-semibold mb-6">Profile</h2>
            
            <div className="flex items-center gap-4 mb-6">
              <div className="relative">
                <div className="w-[72px] h-[72px] rounded-full bg-[#E8D5B7] flex items-center justify-center overflow-hidden">
                  {isUploadingAvatar ? (
                    <Loader2 className="w-6 h-6 animate-spin text-[#8B7355]" />
                  ) : avatarPreview ? (
                    <img src={avatarPreview} alt="" className="w-full h-full object-cover" />
                  ) : (
                    <span className="text-2xl font-medium text-[#8B7355]">{user?.name?.charAt(0) || "U"}</span>
                  )}
                </div>
                <button 
                  onClick={() => fileInputRef.current?.click()} 
                  className="absolute -bottom-0.5 -right-0.5 w-6 h-6 bg-white border border-gray-200 rounded-full flex items-center justify-center shadow-sm hover:bg-gray-50"
                >
                  <Camera className="w-3 h-3 text-gray-600" />
                </button>
                <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={handleAvatarChange} />
              </div>
              
              <div className="flex-1 grid grid-cols-2 gap-4">
                <div>
                  <label className="text-[13px] text-gray-500 mb-1.5 block">Full name</label>
                  <Input value={editedName} onChange={(e) => setEditedName(e.target.value)} className="h-10 text-[14px]" />
                </div>
                <div>
                  <label className="text-[13px] text-gray-500 mb-1.5 block">What should we call you?</label>
                  <Input value={nickname} onChange={(e) => setNickname(e.target.value)} className="h-10 text-[14px]" />
                </div>
              </div>
            </div>
            
            <div className="mb-6">
              <label className="text-[13px] text-gray-500 mb-1.5 block">What best describes your work?</label>
              <select 
                value={workDescription} 
                onChange={(e) => setWorkDescription(e.target.value)} 
                className="w-full h-10 px-3 rounded-md border border-gray-200 bg-white text-[14px] focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option>Engineering</option>
                <option>Data Science</option>
                <option>Machine Learning</option>
                <option>Product</option>
                <option>Research</option>
                <option>Business</option>
                <option>Other</option>
              </select>
            </div>
            
            <div className="mb-4">
              <label className="text-[13px] text-gray-500 mb-1 block">What personal preferences should we consider in responses?</label>
              <p className="text-[12px] text-gray-400 mb-2">Your preferences will apply to all conversations.</p>
              <Textarea 
                value={personalPreferences} 
                onChange={(e) => setPersonalPreferences(e.target.value)} 
                placeholder="e.g. ask clarifying questions before giving detailed answers"
                className="min-h-[80px] text-[14px] resize-none"
              />
            </div>

            <Button onClick={handleSaveProfile} disabled={isSaving} size="sm" className="bg-black text-white hover:bg-gray-800">
              {isSaving ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Check className="w-4 h-4 mr-2" />}
              Save Changes
            </Button>
          </section>

          <hr className="border-gray-200 mb-12" />

          {/* Notifications Section */}
          <section className="mb-12">
            <h2 className="text-[15px] font-semibold mb-6">Notifications</h2>
            
            <div className="space-y-5">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-[14px] font-medium">Training completions</p>
                  <p className="text-[13px] text-gray-500">Get notified when model fine-tuning is complete.</p>
                </div>
                <Toggle enabled={notifyTrainingComplete} onChange={setNotifyTrainingComplete} />
              </div>
              
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-[14px] font-medium">API usage alerts</p>
                  <p className="text-[13px] text-gray-500">Get notified when API usage reaches limits.</p>
                </div>
                <Toggle enabled={notifyApiUsage} onChange={setNotifyApiUsage} />
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <p className="text-[14px] font-medium">New features</p>
                  <p className="text-[13px] text-gray-500">Get notified about new SchemaLabs features.</p>
                </div>
                <Toggle enabled={notifyNewFeatures} onChange={setNotifyNewFeatures} />
              </div>
              
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-[14px] font-medium">Email notifications</p>
                  <p className="text-[13px] text-gray-500">Receive notifications via email.</p>
                </div>
                <Toggle enabled={notifyEmail} onChange={setNotifyEmail} />
              </div>
            </div>
          </section>

          <hr className="border-gray-200 mb-12" />

          {/* Appearance Section */}
          <section className="mb-12">
            <h2 className="text-[15px] font-semibold mb-6">Appearance</h2>
            
            <div className="mb-8">
              <p className="text-[14px] font-medium mb-3">Color mode</p>
              <div className="flex gap-3">
                {/* Light */}
                <button
                  onClick={() => handleThemeChange("light")}
                  className={`flex flex-col items-center rounded-lg overflow-hidden border-2 transition-colors ${
                    theme === "light" ? "border-black" : "border-gray-200 hover:border-gray-300"
                  }`}
                >
                  <div className="w-[88px] h-[60px] bg-gray-50 p-2 flex flex-col">
                    <div className="flex gap-1 mb-1">
                      <div className="w-1.5 h-1.5 rounded-full bg-gray-300" />
                      <div className="w-1.5 h-1.5 rounded-full bg-gray-300" />
                      <div className="w-1.5 h-1.5 rounded-full bg-gray-300" />
                    </div>
                    <div className="flex-1 flex gap-1">
                      <div className="w-4 bg-gray-200 rounded-sm" />
                      <div className="flex-1 bg-white rounded-sm border border-gray-200" />
                    </div>
                  </div>
                  <div className={`w-full py-2 text-center text-[12px] ${theme === "light" ? "bg-black text-white" : "bg-white text-gray-700"}`}>
                    Light
                  </div>
                </button>
                
                {/* Auto */}
                <button
                  onClick={() => handleThemeChange("system")}
                  className={`flex flex-col items-center rounded-lg overflow-hidden border-2 transition-colors ${
                    theme === "system" ? "border-black" : "border-gray-200 hover:border-gray-300"
                  }`}
                >
                  <div className="w-[88px] h-[60px] flex overflow-hidden">
                    <div className="w-1/2 bg-gray-50 p-1 flex flex-col">
                      <div className="flex gap-0.5 mb-0.5">
                        <div className="w-1 h-1 rounded-full bg-gray-300" />
                        <div className="w-1 h-1 rounded-full bg-gray-300" />
                      </div>
                      <div className="flex-1 flex gap-0.5">
                        <div className="w-2 bg-gray-200 rounded-sm" />
                        <div className="flex-1 bg-white rounded-sm" />
                      </div>
                    </div>
                    <div className="w-1/2 bg-gray-800 p-1 flex flex-col">
                      <div className="flex gap-0.5 mb-0.5">
                        <div className="w-1 h-1 rounded-full bg-gray-600" />
                        <div className="w-1 h-1 rounded-full bg-gray-600" />
                      </div>
                      <div className="flex-1 flex gap-0.5">
                        <div className="w-2 bg-gray-700 rounded-sm" />
                        <div className="flex-1 bg-gray-900 rounded-sm" />
                      </div>
                    </div>
                  </div>
                  <div className={`w-full py-2 text-center text-[12px] ${theme === "system" ? "bg-black text-white" : "bg-white text-gray-700"}`}>
                    Auto
                  </div>
                </button>
                
                {/* Dark */}
                <button
                  onClick={() => handleThemeChange("dark")}
                  className={`flex flex-col items-center rounded-lg overflow-hidden border-2 transition-colors ${
                    theme === "dark" ? "border-black" : "border-gray-200 hover:border-gray-300"
                  }`}
                >
                  <div className="w-[88px] h-[60px] bg-gray-800 p-2 flex flex-col">
                    <div className="flex gap-1 mb-1">
                      <div className="w-1.5 h-1.5 rounded-full bg-gray-600" />
                      <div className="w-1.5 h-1.5 rounded-full bg-gray-600" />
                      <div className="w-1.5 h-1.5 rounded-full bg-gray-600" />
                    </div>
                    <div className="flex-1 flex gap-1">
                      <div className="w-4 bg-gray-700 rounded-sm" />
                      <div className="flex-1 bg-gray-900 rounded-sm" />
                    </div>
                  </div>
                  <div className={`w-full py-2 text-center text-[12px] ${theme === "dark" ? "bg-black text-white" : "bg-white text-gray-700"}`}>
                    Dark
                  </div>
                </button>
              </div>
            </div>
            
            <div>
              <p className="text-[14px] font-medium mb-3">Chat font</p>
              <div className="flex gap-3">
                {[
                  { value: "default", label: "Default", fontClass: "font-serif" },
                  { value: "sans", label: "Sans", fontClass: "font-sans" },
                  { value: "system", label: "System", fontClass: "" },
                  { value: "dyslexic", label: "Dyslexic friendly", fontClass: "tracking-wide" }
                ].map(({ value, label, fontClass }) => (
                  <button
                    key={value}
                    onClick={() => handleFontChange(value)}
                    className={`flex flex-col items-center rounded-lg overflow-hidden border-2 transition-colors ${
                      currentFont === value ? "border-black" : "border-gray-200 hover:border-gray-300"
                    }`}
                  >
                    <div className={`w-[88px] h-[48px] bg-gray-50 flex items-center justify-center ${fontClass}`}>
                      <span className="text-2xl">Aa</span>
                    </div>
                    <div className={`w-full py-2 text-center text-[12px] ${currentFont === value ? "bg-black text-white" : "bg-white text-gray-700"}`}>
                      {label}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </section>

          <hr className="border-gray-200 mb-12" />

          {/* Account Section */}
          <section className="mb-12">
            <h2 className="text-[15px] font-semibold mb-6">Account</h2>
            
            <div className="space-y-4">
              <div className="flex items-center justify-between py-2">
                <span className="text-[14px]">Log out of all devices</span>
                <Button variant="outline" size="sm" onClick={handleLogoutAllDevices} className="h-9">
                  Log out
                </Button>
              </div>
              
              <div className="flex items-center justify-between py-2">
                <span className="text-[14px]">Change password</span>
                <Button variant="outline" size="sm" onClick={() => setShowPasswordChange(true)} className="h-9">
                  Change password
                </Button>
              </div>
              
              <div className="flex items-center justify-between py-2">
                <span className="text-[14px]">Delete account</span>
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={() => setShowDeleteConfirm(true)} 
                  className="h-9 text-red-600 border-red-200 hover:bg-red-50 hover:border-red-300"
                >
                  <Trash2 className="w-4 h-4 mr-2" />
                  Delete account
                </Button>
              </div>
              
              <div className="flex items-center justify-between py-2">
                <span className="text-[14px]">User ID</span>
                <div className="flex items-center gap-2">
                  <span className="text-[13px] font-mono text-gray-500">{user?.id}</span>
                  <button onClick={copyUserId} className="text-gray-400 hover:text-gray-600">
                    {copiedId ? <Check className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
                  </button>
                </div>
              </div>
            </div>
          </section>

          <hr className="border-gray-200 mb-12" />

          {/* Active Sessions Section */}
          <section className="mb-12">
            <h2 className="text-[15px] font-semibold mb-6">Active sessions</h2>
            
            {sessions.length > 0 ? (
              <>
                <div className="overflow-x-auto">
                  <table className="w-full text-[14px]">
                    <thead>
                      <tr className="border-b border-gray-200">
                        <th className="text-left py-3 font-medium text-gray-500">Device</th>
                        <th className="text-left py-3 font-medium text-gray-500">Location</th>
                        <th className="text-left py-3 font-medium text-gray-500">Created</th>
                        <th className="text-left py-3 font-medium text-gray-500">Updated</th>
                        <th className="w-10"></th>
                      </tr>
                    </thead>
                    <tbody>
                      {sessions.map((s) => (
                        <tr key={s.id} className="border-b border-gray-100">
                          <td className="py-3">
                            <div className="flex items-center gap-2">
                              {s.device_type === "mobile" ? <Smartphone className="w-4 h-4 text-gray-400" /> : <Monitor className="w-4 h-4 text-gray-400" />}
                              <span>{s.device}</span>
                              {s.is_current && <span className="px-1.5 py-0.5 text-[10px] font-medium bg-green-100 text-green-700 rounded">Current</span>}
                            </div>
                          </td>
                          <td className="py-3 text-gray-500">{s.location}</td>
                          <td className="py-3 text-gray-500">{formatDate(s.created_at)}</td>
                          <td className="py-3 text-gray-500">{formatDate(s.updated_at)}</td>
                          <td className="py-3">
                            <button className="p-1 hover:bg-gray-100 rounded">
                              <MoreHorizontal className="w-4 h-4 text-gray-400" />
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                
                {totalSessionPages > 1 && (
                  <div className="flex items-center justify-between mt-4 pt-4 border-t border-gray-200">
                    <p className="text-[13px] text-gray-500">
                      Showing {(sessionPage - 1) * sessionsPerPage + 1}-{Math.min(sessionPage * sessionsPerPage, totalSessions)} of {totalSessions} sessions
                    </p>
                    <div className="flex items-center gap-2">
                      <button disabled={sessionPage === 1} onClick={() => setSessionPage(p => p - 1)} className="p-1 hover:bg-gray-100 rounded disabled:opacity-50">
                        <ChevronLeft className="w-5 h-5" />
                      </button>
                      <span className="text-[13px]">Page {sessionPage} of {totalSessionPages}</span>
                      <button disabled={sessionPage === totalSessionPages} onClick={() => setSessionPage(p => p + 1)} className="p-1 hover:bg-gray-100 rounded disabled:opacity-50">
                        <ChevronRight className="w-5 h-5" />
                      </button>
                    </div>
                  </div>
                )}
              </>
            ) : (
              <p className="text-[14px] text-gray-500">No active sessions found.</p>
            )}
          </section>

          {/* Delete Account Modal */}
          {showDeleteConfirm && (
            <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
              <div className="bg-white rounded-xl p-6 max-w-md w-full mx-4 shadow-2xl">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 rounded-full bg-red-100 flex items-center justify-center">
                    <AlertTriangle className="w-5 h-5 text-red-600" />
                  </div>
                  <h3 className="text-lg font-semibold">Delete Account</h3>
                </div>
                
                <p className="text-[14px] text-gray-600 mb-4">
                  This action is <strong>permanent</strong> and cannot be undone. All your data, models, queries, and settings will be permanently deleted.
                </p>
                
                <div className="bg-red-50 border border-red-200 rounded-lg p-3 mb-4">
                  <p className="text-[13px] text-red-700">
                    To confirm, type <strong>DELETE</strong> below:
                  </p>
                </div>
                
                <Input 
                  value={deleteConfirmText} 
                  onChange={(e) => setDeleteConfirmText(e.target.value)} 
                  placeholder="Type DELETE to confirm"
                  className="mb-4 text-center font-mono"
                />
                
                <div className="flex gap-3 justify-end">
                  <Button 
                    variant="outline" 
                    onClick={() => { setShowDeleteConfirm(false); setDeleteConfirmText("") }}
                    disabled={isDeleting}
                  >
                    Cancel
                  </Button>
                  <Button 
                    onClick={handleDeleteAccount} 
                    disabled={deleteConfirmText !== "DELETE" || isDeleting}
                    className="bg-red-600 hover:bg-red-700 text-white"
                  >
                    {isDeleting ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Trash2 className="w-4 h-4 mr-2" />}
                    Delete Account
                  </Button>
                </div>
              </div>
            </div>
          )}

          {/* Password Change Modal */}
          {showPasswordChange && (
            <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
              <div className="bg-white rounded-xl p-6 max-w-md w-full mx-4 shadow-2xl">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <Key className="w-5 h-5" />
                  Change Password
                </h3>
                {passwordError && (
                  <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-[13px] text-red-600">{passwordError}</div>
                )}
                {passwordSuccess && (
                  <div className="mb-4 p-3 bg-green-50 border border-green-200 rounded-lg text-[13px] text-green-600">{passwordSuccess}</div>
                )}
                {passwordStep === "request" ? (
                  <>
                    <p className="text-[14px] text-gray-600 mb-4">
                      We will send a 6-digit verification code to <strong>{user?.email}</strong>
                    </p>
                    <div className="flex gap-2 justify-end">
                      <Button variant="outline" onClick={() => { setShowPasswordChange(false); resetPasswordForm() }}>Cancel</Button>
                      <Button onClick={handleRequestPasswordChange} disabled={passwordLoading} className="bg-black hover:bg-gray-800">
                        {passwordLoading && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                        Send Code
                      </Button>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="space-y-4 mb-4">
                      <div>
                        <label className="text-[13px] font-medium mb-1.5 block">Verification Code</label>
                        <Input value={verificationCode} onChange={(e) => setVerificationCode(e.target.value)} maxLength={6} className="text-center text-lg tracking-widest" />
                      </div>
                      <div>
                        <label className="text-[13px] font-medium mb-1.5 block">New Password</label>
                        <div className="relative">
                          <Input type={showNewPassword ? "text" : "password"} value={newPassword} onChange={(e) => setNewPassword(e.target.value)} />
                          <button onClick={() => setShowNewPassword(!showNewPassword)} className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400">
                            {showNewPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                          </button>
                        </div>
                      </div>
                      <div>
                        <label className="text-[13px] font-medium mb-1.5 block">Confirm Password</label>
                        <Input type="password" value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} />
                      </div>
                    </div>
                    <div className="flex gap-2 justify-end">
                      <Button variant="outline" onClick={() => { setShowPasswordChange(false); resetPasswordForm() }}>Cancel</Button>
                      <Button onClick={handleVerifyAndChangePassword} disabled={passwordLoading} className="bg-black hover:bg-gray-800">
                        {passwordLoading && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                        Change Password
                      </Button>
                    </div>
                  </>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </Sidebar>
  )
}

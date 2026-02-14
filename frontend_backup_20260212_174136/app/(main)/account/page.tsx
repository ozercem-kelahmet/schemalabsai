"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import {
  User,
  Mail,
  Bell,
  Shield,
  Copy,
  Check,
  Laptop,
  Smartphone,
  Monitor,
  AlertTriangle,
  LogOut,
  Key,
  Trash2,
  Loader2,
} from "lucide-react"
import { cn } from "@/lib/utils"

const avatarColors = [
  { name: "Blue", from: "#2684FF", to: "#0052CC" },
  { name: "Purple", from: "#9F7AEA", to: "#6B46C1" },
  { name: "Green", from: "#48BB78", to: "#2F855A" },
  { name: "Orange", from: "#ED8936", to: "#C05621" },
  { name: "Red", from: "#FC8181", to: "#C53030" },
  { name: "Teal", from: "#38B2AC", to: "#2C7A7B" },
  { name: "Pink", from: "#F687B3", to: "#B83280" },
  { name: "Gray", from: "#A0AEC0", to: "#4A5568" },
]

interface UserData {
  id: string
  name: string
  email: string
  created_at: string
}

interface SessionData {
  id: string
  device: string
  device_type: string
  location: string
  created_at: string
  updated_at: string
  is_current: boolean
}

export default function AccountPage() {
  const router = useRouter()
  const [user, setUser] = useState<UserData | null>(null)
  const [sessions, setSessions] = useState<SessionData[]>([])
  const [loading, setLoading] = useState(true)
  
  const [firstName, setFirstName] = useState("")
  const [lastName, setLastName] = useState("")
  const [email, setEmail] = useState("")
  const [selectedColor, setSelectedColor] = useState(avatarColors[0])
  const [isEditingName, setIsEditingName] = useState(false)
  const [tempFirstName, setTempFirstName] = useState("")
  const [tempLastName, setTempLastName] = useState("")
  const [saving, setSaving] = useState(false)
  
  // Email change state
  const [changeEmailModalOpen, setChangeEmailModalOpen] = useState(false)
  const [newEmail, setNewEmail] = useState("")
  const [verificationCode, setVerificationCode] = useState("")
  const [emailStep, setEmailStep] = useState<"enter" | "verify">("enter")
  const [sendingCode, setSendingCode] = useState(false)

  // Password change state
  const [changePasswordModalOpen, setChangePasswordModalOpen] = useState(false)
  const [passwordStep, setPasswordStep] = useState<"request" | "verify">("request")
  const [passwordCode, setPasswordCode] = useState("")
  const [newPassword, setNewPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [changingPassword, setChangingPassword] = useState(false)

  // Notifications state
  const [notifications, setNotifications] = useState({
    trainingCompletions: true,
    apiUsageAlerts: true,
    newFeatures: true,
    emailNotifications: true,
  })

  // Delete account state
  const [deleteAccountModalOpen, setDeleteAccountModalOpen] = useState(false)
  const [deleteConfirmText, setDeleteConfirmText] = useState("")
  const [deleting, setDeleting] = useState(false)
  
  const [copiedUserId, setCopiedUserId] = useState(false)
  const [loggingOutAll, setLoggingOutAll] = useState(false)

  useEffect(() => {
    fetchUser()
    fetchSessions()
  }, [])

  const fetchUser = async () => {
    try {
      const res = await fetch("/api/auth/me", { credentials: "include" })
      if (res.ok) {
        const data = await res.json()
        setUser(data)
        const nameParts = (data.name || "").split(" ")
        setFirstName(nameParts[0] || "")
        setLastName(nameParts.slice(1).join(" ") || "")
        setTempFirstName(nameParts[0] || "")
        setTempLastName(nameParts.slice(1).join(" ") || "")
        setEmail(data.email || "")
        const colorIndex = data.id ? data.id.charCodeAt(0) % avatarColors.length : 0
        setSelectedColor(avatarColors[colorIndex])
      }
    } catch (e) {
      console.error("Failed to fetch user:", e)
    } finally {
      setLoading(false)
    }
  }

  const fetchSessions = async () => {
    try {
      const res = await fetch("/api/auth/sessions", { credentials: "include" })
      if (res.ok) {
        const data = await res.json()
        setSessions(data.sessions || [])
      }
    } catch (e) {
      console.error("Failed to fetch sessions:", e)
    }
  }

  const handleSaveName = async () => {
    setSaving(true)
    try {
      const fullName = `${tempFirstName} ${tempLastName}`.trim()
      const res = await fetch("/api/auth/update-profile", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ name: fullName })
      })
      if (res.ok) {
        setFirstName(tempFirstName)
        setLastName(tempLastName)
        setIsEditingName(false)
      }
    } catch (e) {
      console.error("Failed to update name:", e)
    } finally {
      setSaving(false)
    }
  }

  const handleCancelNameEdit = () => {
    setTempFirstName(firstName)
    setTempLastName(lastName)
    setIsEditingName(false)
  }

  // Password change handlers
  const handleRequestPasswordCode = async () => {
    setChangingPassword(true)
    try {
      const res = await fetch("/api/auth/change-password-request", {
        method: "POST",
        credentials: "include"
      })
      if (res.ok) {
        setPasswordStep("verify")
      } else {
        const data = await res.json()
        alert(data.error || "Failed to send code")
      }
    } catch (e) {
      console.error("Failed to request password change:", e)
    } finally {
      setChangingPassword(false)
    }
  }

  const handleVerifyAndChangePassword = async () => {
    if (newPassword !== confirmPassword) {
      alert("Passwords do not match")
      return
    }
    if (newPassword.length < 6) {
      alert("Password must be at least 6 characters")
      return
    }
    
    setChangingPassword(true)
    try {
      const res = await fetch("/api/auth/change-password-verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ code: passwordCode, new_password: newPassword })
      })
      if (res.ok) {
        setChangePasswordModalOpen(false)
        setPasswordStep("request")
        setPasswordCode("")
        setNewPassword("")
        setConfirmPassword("")
        alert("Password changed successfully!")
      } else {
        const data = await res.json()
        alert(data.error || "Failed to change password")
      }
    } catch (e) {
      console.error("Failed to change password:", e)
    } finally {
      setChangingPassword(false)
    }
  }

  const handleLogoutAll = async () => {
    setLoggingOutAll(true)
    try {
      const res = await fetch("/api/auth/logout-all", {
        method: "POST",
        credentials: "include"
      })
      if (res.ok) {
        fetchSessions()
        alert("Logged out from all other devices")
      }
    } catch (e) {
      console.error("Failed to logout all:", e)
    } finally {
      setLoggingOutAll(false)
    }
  }

  const handleDeleteAccount = async () => {
    if (deleteConfirmText !== "DELETE") return
    setDeleting(true)
    try {
      const res = await fetch("/api/auth/delete-account", {
        method: "POST",
        credentials: "include"
      })
      if (res.ok) {
        router.push("/login")
      }
    } catch (e) {
      console.error("Failed to delete account:", e)
    } finally {
      setDeleting(false)
    }
  }

  const copyUserId = () => {
    if (user?.id) {
      navigator.clipboard.writeText(user.id)
      setCopiedUserId(true)
      setTimeout(() => setCopiedUserId(false), 2000)
    }
  }

  const getDeviceIcon = (type: string) => {
    switch (type) {
      case "laptop": return <Laptop className="h-5 w-5" />
      case "mobile": return <Smartphone className="h-5 w-5" />
      default: return <Monitor className="h-5 w-5" />
    }
  }

  const formatDate = (date: Date | string) => {
    return new Date(date).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    })
  }

  const getInitials = () => {
    return `${firstName.charAt(0)}${lastName.charAt(0)}`.toUpperCase() || "U"
  }

  if (loading) {
    return <div className="flex items-center justify-center h-64 text-muted-foreground">Loading...</div>
  }

  return (
    <div className="space-y-6 max-w-4xl">
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[#0052CC]/10 dark:bg-[#0052CC]/20">
          <User className="h-5 w-5 text-[#0052CC] dark:text-[#2684FF]" />
        </div>
        <div>
          <h1 className="text-xl font-semibold text-foreground">Account</h1>
          <p className="text-sm text-muted-foreground">Manage your profile and preferences</p>
        </div>
      </div>

      {/* Profile Section */}
      <Card className="border-border bg-card">
        <CardHeader>
          <CardTitle className="text-foreground flex items-center gap-2">
            <User className="h-5 w-5" />
            Profile
          </CardTitle>
          <CardDescription>Your personal information and avatar</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="flex items-start gap-6">
            <div className="space-y-3">
              <div
                className="flex h-20 w-20 items-center justify-center rounded-full text-2xl font-semibold text-white"
                style={{ background: `linear-gradient(135deg, ${selectedColor.from}, ${selectedColor.to})` }}
              >
                {getInitials()}
              </div>
              <p className="text-xs text-muted-foreground text-center">Avatar Color</p>
            </div>
            <div className="flex flex-wrap gap-2 pt-2">
              {avatarColors.map((color) => (
                <button
                  key={color.name}
                  onClick={() => setSelectedColor(color)}
                  className={cn(
                    "h-8 w-8 rounded-full transition-all",
                    selectedColor.name === color.name && "ring-2 ring-offset-2 ring-offset-background ring-[#0052CC]"
                  )}
                  style={{ background: `linear-gradient(135deg, ${color.from}, ${color.to})` }}
                  title={color.name}
                />
              ))}
            </div>
          </div>

          <div className="space-y-3">
            <Label className="text-foreground">Name</Label>
            {isEditingName ? (
              <div className="flex items-center gap-3">
                <Input value={tempFirstName} onChange={(e) => setTempFirstName(e.target.value)} placeholder="First name" className="max-w-[200px] border-border bg-background" />
                <Input value={tempLastName} onChange={(e) => setTempLastName(e.target.value)} placeholder="Last name" className="max-w-[200px] border-border bg-background" />
                <Button size="sm" onClick={handleSaveName} disabled={saving} className="bg-[#0052CC] text-white hover:bg-[#003D99]">
                  {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : "Save"}
                </Button>
                <Button size="sm" variant="outline" onClick={handleCancelNameEdit} className="bg-transparent">Cancel</Button>
              </div>
            ) : (
              <div className="flex items-center gap-3">
                <p className="text-foreground">{firstName} {lastName || "(No name set)"}</p>
                <Button size="sm" variant="outline" onClick={() => setIsEditingName(true)} className="bg-transparent">Edit</Button>
              </div>
            )}
          </div>

          <div className="space-y-3">
            <Label className="text-foreground">Email</Label>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 text-foreground">
                <Mail className="h-4 w-4 text-muted-foreground" />
                {email}
              </div>
            </div>
          </div>

          <div className="space-y-3">
            <Label className="text-foreground">Member Since</Label>
            <p className="text-muted-foreground text-sm">{user?.created_at ? formatDate(user.created_at) : "Unknown"}</p>
          </div>
        </CardContent>
      </Card>

      {/* Notifications Section */}
      <Card className="border-border bg-card">
        <CardHeader>
          <CardTitle className="text-foreground flex items-center gap-2">
            <Bell className="h-5 w-5" />
            Notifications
          </CardTitle>
          <CardDescription>Configure how you receive notifications</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between py-2">
            <div className="space-y-0.5">
              <Label className="text-foreground">Training completions</Label>
              <p className="text-sm text-muted-foreground">Get notified when model fine-tuning is complete</p>
            </div>
            <Switch checked={notifications.trainingCompletions} onCheckedChange={(checked) => setNotifications(prev => ({ ...prev, trainingCompletions: checked }))} />
          </div>
          <div className="flex items-center justify-between py-2 border-t border-border">
            <div className="space-y-0.5">
              <Label className="text-foreground">API usage alerts</Label>
              <p className="text-sm text-muted-foreground">Get notified when API usage reaches limits</p>
            </div>
            <Switch checked={notifications.apiUsageAlerts} onCheckedChange={(checked) => setNotifications(prev => ({ ...prev, apiUsageAlerts: checked }))} />
          </div>
          <div className="flex items-center justify-between py-2 border-t border-border">
            <div className="space-y-0.5">
              <Label className="text-foreground">New features</Label>
              <p className="text-sm text-muted-foreground">Get notified about new SchemaLabs features</p>
            </div>
            <Switch checked={notifications.newFeatures} onCheckedChange={(checked) => setNotifications(prev => ({ ...prev, newFeatures: checked }))} />
          </div>
          <div className="flex items-center justify-between py-2 border-t border-border">
            <div className="space-y-0.5">
              <Label className="text-foreground">Email notifications</Label>
              <p className="text-sm text-muted-foreground">Receive notifications via email</p>
            </div>
            <Switch checked={notifications.emailNotifications} onCheckedChange={(checked) => setNotifications(prev => ({ ...prev, emailNotifications: checked }))} />
          </div>
        </CardContent>
      </Card>

      {/* Active Sessions Section */}
      <Card className="border-border bg-card">
        <CardHeader>
          <CardTitle className="text-foreground flex items-center gap-2">
            <Monitor className="h-5 w-5" />
            Active Sessions
          </CardTitle>
          <CardDescription>Devices where you are currently logged in</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {sessions.length === 0 ? (
              <div className="flex items-center justify-between p-3 rounded-lg border border-[#0052CC]/30 bg-[#0052CC]/5">
                <div className="flex items-center gap-3">
                  <div className="text-muted-foreground">{getDeviceIcon("desktop")}</div>
                  <div>
                    <div className="flex items-center gap-2">
                      <p className="text-sm font-medium text-foreground">Current Device</p>
                      <span className="rounded bg-[#0052CC]/20 px-1.5 py-0.5 text-[10px] font-medium text-[#0052CC] dark:text-[#2684FF]">Current</span>
                    </div>
                    <p className="text-xs text-muted-foreground">Active now</p>
                  </div>
                </div>
              </div>
            ) : (
              sessions.map((session) => (
                <div
                  key={session.id}
                  className={cn(
                    "flex items-center justify-between p-3 rounded-lg border",
                    session.is_current ? "border-[#0052CC]/30 bg-[#0052CC]/5" : "border-border bg-muted/30"
                  )}
                >
                  <div className="flex items-center gap-3">
                    <div className="text-muted-foreground">{getDeviceIcon(session.device_type)}</div>
                    <div>
                      <div className="flex items-center gap-2">
                        <p className="text-sm font-medium text-foreground">{session.device}</p>
                        {session.is_current && (
                          <span className="rounded bg-[#0052CC]/20 px-1.5 py-0.5 text-[10px] font-medium text-[#0052CC] dark:text-[#2684FF]">Current</span>
                        )}
                      </div>
                      <p className="text-xs text-muted-foreground">{session.location}</p>
                    </div>
                  </div>
                  <div className="text-right text-xs text-muted-foreground">
                    <p>Created: {formatDate(session.created_at)}</p>
                    <p>Updated: {formatDate(session.updated_at)}</p>
                  </div>
                </div>
              ))
            )}
          </div>
        </CardContent>
      </Card>

      {/* Advanced Section */}
      <Card className="border-border bg-card">
        <CardHeader>
          <CardTitle className="text-foreground flex items-center gap-2">
            <Shield className="h-5 w-5" />
            Advanced
          </CardTitle>
          <CardDescription>Security and account management</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between py-2">
            <div className="space-y-0.5">
              <Label className="text-foreground">User ID</Label>
              <p className="text-xs text-muted-foreground font-mono">{user?.id || "Unknown"}</p>
            </div>
            <Button size="sm" variant="outline" onClick={copyUserId} className="gap-2 bg-transparent">
              {copiedUserId ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
              {copiedUserId ? "Copied" : "Copy"}
            </Button>
          </div>

          <div className="flex items-center justify-between py-2 border-t border-border">
            <div className="space-y-0.5">
              <Label className="text-foreground">Change password</Label>
              <p className="text-sm text-muted-foreground">Update your account password</p>
            </div>
            <Button size="sm" variant="outline" onClick={() => setChangePasswordModalOpen(true)} className="gap-2 bg-transparent">
              <Key className="h-4 w-4" />
              Change
            </Button>
          </div>

          <div className="flex items-center justify-between py-2 border-t border-border">
            <div className="space-y-0.5">
              <Label className="text-foreground">Log out of all devices</Label>
              <p className="text-sm text-muted-foreground">Sign out from all active sessions</p>
            </div>
            <Button size="sm" variant="outline" onClick={handleLogoutAll} disabled={loggingOutAll} className="gap-2 bg-transparent">
              {loggingOutAll ? <Loader2 className="h-4 w-4 animate-spin" /> : <LogOut className="h-4 w-4" />}
              Log out all
            </Button>
          </div>

          <div className="flex items-center justify-between py-2 border-t border-border">
            <div className="space-y-0.5">
              <Label className="text-red-500">Delete account</Label>
              <p className="text-sm text-muted-foreground">Permanently delete your account and all data</p>
            </div>
            <Button size="sm" variant="outline" onClick={() => setDeleteAccountModalOpen(true)} className="gap-2 border-red-500/30 text-red-500 hover:bg-red-500/10 hover:text-red-500 bg-transparent">
              <Trash2 className="h-4 w-4" />
              Delete
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Change Password Modal */}
      <Dialog open={changePasswordModalOpen} onOpenChange={(open) => { if (!open) { setChangePasswordModalOpen(false); setPasswordStep("request"); setPasswordCode(""); setNewPassword(""); setConfirmPassword("") } else setChangePasswordModalOpen(true) }}>
        <DialogContent className="border-border bg-card sm:max-w-[400px]">
          <DialogHeader>
            <DialogTitle className="text-foreground">Change Password</DialogTitle>
            <DialogDescription className="text-muted-foreground">
              {passwordStep === "request" ? "We'll send a verification code to your email" : "Enter the code and your new password"}
            </DialogDescription>
          </DialogHeader>
          {passwordStep === "request" ? (
            <div className="space-y-4">
              <p className="text-sm text-muted-foreground">A 6-digit verification code will be sent to: <span className="text-foreground font-medium">{email}</span></p>
              <DialogFooter>
                <Button variant="outline" onClick={() => setChangePasswordModalOpen(false)} className="bg-transparent">Cancel</Button>
                <Button onClick={handleRequestPasswordCode} disabled={changingPassword} className="bg-[#0052CC] text-white hover:bg-[#003D99]">
                  {changingPassword ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
                  Send Code
                </Button>
              </DialogFooter>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="password-code" className="text-foreground">Verification Code</Label>
                <Input id="password-code" placeholder="000000" value={passwordCode} onChange={(e) => setPasswordCode(e.target.value.replace(/\D/g, "").slice(0, 6))} className="border-border bg-background text-center text-lg tracking-widest font-mono" maxLength={6} />
              </div>
              <div className="space-y-2">
                <Label htmlFor="new-password" className="text-foreground">New Password</Label>
                <Input id="new-password" type="password" value={newPassword} onChange={(e) => setNewPassword(e.target.value)} className="border-border bg-background" />
                <p className="text-xs text-muted-foreground">Minimum 6 characters</p>
              </div>
              <div className="space-y-2">
                <Label htmlFor="confirm-password" className="text-foreground">Confirm New Password</Label>
                <Input id="confirm-password" type="password" value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} className="border-border bg-background" />
                {confirmPassword && newPassword !== confirmPassword && <p className="text-xs text-red-500">Passwords do not match</p>}
              </div>
              <DialogFooter>
                <Button variant="outline" onClick={() => { setPasswordStep("request"); setPasswordCode(""); setNewPassword(""); setConfirmPassword("") }} className="bg-transparent">Back</Button>
                <Button onClick={handleVerifyAndChangePassword} disabled={changingPassword || passwordCode.length !== 6 || newPassword.length < 6 || newPassword !== confirmPassword} className="bg-[#0052CC] text-white hover:bg-[#003D99]">
                  {changingPassword ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
                  Change Password
                </Button>
              </DialogFooter>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Delete Account Modal */}
      <Dialog open={deleteAccountModalOpen} onOpenChange={setDeleteAccountModalOpen}>
        <DialogContent className="border-border bg-card sm:max-w-[400px]">
          <DialogHeader>
            <DialogTitle className="text-foreground flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-red-500" />
              Delete Account
            </DialogTitle>
            <DialogDescription className="text-muted-foreground">
              This action cannot be undone. This will permanently delete your account, all your models, datasets, and API keys.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-3">
              <p className="text-sm text-red-500">You will lose access to:</p>
              <ul className="mt-2 text-sm text-muted-foreground list-disc list-inside space-y-1">
                <li>All trained models</li>
                <li>Connected data sources</li>
                <li>API keys and endpoints</li>
                <li>Query history and sessions</li>
              </ul>
            </div>
            <div className="space-y-2">
              <Label htmlFor="delete-confirm" className="text-foreground">
                Type <span className="font-mono font-bold text-red-500">DELETE</span> to confirm
              </Label>
              <Input id="delete-confirm" value={deleteConfirmText} onChange={(e) => setDeleteConfirmText(e.target.value)} className="border-border bg-background" placeholder="DELETE" />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => { setDeleteAccountModalOpen(false); setDeleteConfirmText("") }} className="bg-transparent">Cancel</Button>
            <Button onClick={handleDeleteAccount} disabled={deleteConfirmText !== "DELETE" || deleting} className="bg-red-500 text-white hover:bg-red-600">
              {deleting ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
              Delete Account
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
"use client"

import { useState, useEffect } from "react"
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
} from "lucide-react"
import { cn } from "@/lib/utils"

// Avatar color options
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

// Mock active sessions
const mockSessions = [
  {
    id: "sess-1",
    device: "MacBook Pro",
    deviceType: "laptop",
    location: "New York, US",
    createdAt: new Date(Date.now() - 1000 * 60 * 60 * 2),
    updatedAt: new Date(Date.now() - 1000 * 60 * 5),
    current: true,
  },
  {
    id: "sess-2",
    device: "iPhone 15 Pro",
    deviceType: "mobile",
    location: "New York, US",
    createdAt: new Date(Date.now() - 1000 * 60 * 60 * 24 * 3),
    updatedAt: new Date(Date.now() - 1000 * 60 * 60 * 12),
    current: false,
  },
  {
    id: "sess-3",
    device: "Windows Desktop",
    deviceType: "desktop",
    location: "Boston, US",
    createdAt: new Date(Date.now() - 1000 * 60 * 60 * 24 * 7),
    updatedAt: new Date(Date.now() - 1000 * 60 * 60 * 24 * 2),
    current: false,
  },
]

export default function AccountPage() {
  // Profile state
  const [firstName, setFirstName] = useState("Schema")
  const [lastName, setLastName] = useState("User")
  const [email, setEmail] = useState("user@schemalabs.ai")
  const [selectedColor, setSelectedColor] = useState(avatarColors[0])
  const [isEditingName, setIsEditingName] = useState(false)
  const [tempFirstName, setTempFirstName] = useState(firstName)
  const [tempLastName, setTempLastName] = useState(lastName)
  
  // Email change state
  const [changeEmailModalOpen, setChangeEmailModalOpen] = useState(false)
  const [newEmail, setNewEmail] = useState("")
  const [verificationCode, setVerificationCode] = useState("")
  const [emailStep, setEmailStep] = useState<"enter" | "verify">("enter")
  const [codeSent, setCodeSent] = useState(false)

  // Notification state
  const [notifications, setNotifications] = useState({
    trainingCompletions: true,
    apiUsageAlerts: true,
    newFeatures: true,
    emailNotifications: true,
  })

  // Advanced state
  const [deleteAccountModalOpen, setDeleteAccountModalOpen] = useState(false)
  const [deleteConfirmText, setDeleteConfirmText] = useState("")
  const [changePasswordModalOpen, setChangePasswordModalOpen] = useState(false)
  const [currentPassword, setCurrentPassword] = useState("")
  const [newPassword, setNewPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [copiedUserId, setCopiedUserId] = useState(false)

  const userId = "usr_7x9k2m4n8p1q3r5t"

  // Handlers
  const handleSaveName = () => {
    setFirstName(tempFirstName)
    setLastName(tempLastName)
    setIsEditingName(false)
  }

  const handleCancelNameEdit = () => {
    setTempFirstName(firstName)
    setTempLastName(lastName)
    setIsEditingName(false)
  }

  const handleSendVerificationCode = () => {
    setCodeSent(true)
    setEmailStep("verify")
  }

  const handleVerifyEmail = () => {
    if (verificationCode.length === 6) {
      setEmail(newEmail)
      setChangeEmailModalOpen(false)
      setNewEmail("")
      setVerificationCode("")
      setEmailStep("enter")
      setCodeSent(false)
    }
  }

  const handleChangePassword = () => {
    if (newPassword === confirmPassword && newPassword.length >= 8) {
      setChangePasswordModalOpen(false)
      setCurrentPassword("")
      setNewPassword("")
      setConfirmPassword("")
    }
  }

  const handleDeleteAccount = () => {
    if (deleteConfirmText === "DELETE") {
      // Handle account deletion
      console.log("Account deleted")
    }
  }

  const copyUserId = () => {
    navigator.clipboard.writeText(userId)
    setCopiedUserId(true)
    setTimeout(() => setCopiedUserId(false), 2000)
  }

  const getDeviceIcon = (type: string) => {
    switch (type) {
      case "laptop":
        return <Laptop className="h-5 w-5" />
      case "mobile":
        return <Smartphone className="h-5 w-5" />
      default:
        return <Monitor className="h-5 w-5" />
    }
  }

  const formatDate = (date: Date) => {
    return date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    })
  }

  const getInitials = () => {
    return `${firstName.charAt(0)}${lastName.charAt(0)}`.toUpperCase()
  }

  return (
    <div className="space-y-6 max-w-4xl">
      {/* Page Header */}
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
          {/* Avatar */}
          <div className="flex items-start gap-6">
            <div className="space-y-3">
              <div
                className="flex h-20 w-20 items-center justify-center rounded-full text-2xl font-semibold text-white"
                style={{
                  background: `linear-gradient(135deg, ${selectedColor.from}, ${selectedColor.to})`,
                }}
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
                  style={{
                    background: `linear-gradient(135deg, ${color.from}, ${color.to})`,
                  }}
                  title={color.name}
                />
              ))}
            </div>
          </div>

          {/* Name */}
          <div className="space-y-3">
            <Label className="text-foreground">Name</Label>
            {isEditingName ? (
              <div className="flex items-center gap-3">
                <Input
                  value={tempFirstName}
                  onChange={(e) => setTempFirstName(e.target.value)}
                  placeholder="First name"
                  className="max-w-[200px] border-border bg-background"
                />
                <Input
                  value={tempLastName}
                  onChange={(e) => setTempLastName(e.target.value)}
                  placeholder="Last name"
                  className="max-w-[200px] border-border bg-background"
                />
                <Button size="sm" onClick={handleSaveName} className="bg-[#0052CC] text-white hover:bg-[#003D99]">
                  Save
                </Button>
                <Button size="sm" variant="outline" onClick={handleCancelNameEdit} className="bg-transparent">
                  Cancel
                </Button>
              </div>
            ) : (
              <div className="flex items-center gap-3">
                <p className="text-foreground">{firstName} {lastName}</p>
                <Button size="sm" variant="outline" onClick={() => setIsEditingName(true)} className="bg-transparent">
                  Edit
                </Button>
              </div>
            )}
          </div>

          {/* Email */}
          <div className="space-y-3">
            <Label className="text-foreground">Email</Label>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 text-foreground">
                <Mail className="h-4 w-4 text-muted-foreground" />
                {email}
              </div>
              <Button 
                size="sm" 
                variant="outline" 
                onClick={() => setChangeEmailModalOpen(true)} 
                className="bg-transparent"
              >
                Change
              </Button>
            </div>
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
            <Switch
              checked={notifications.trainingCompletions}
              onCheckedChange={(checked) => setNotifications((prev) => ({ ...prev, trainingCompletions: checked }))}
            />
          </div>
          <div className="flex items-center justify-between py-2 border-t border-border">
            <div className="space-y-0.5">
              <Label className="text-foreground">API usage alerts</Label>
              <p className="text-sm text-muted-foreground">Get notified when API usage reaches limits</p>
            </div>
            <Switch
              checked={notifications.apiUsageAlerts}
              onCheckedChange={(checked) => setNotifications((prev) => ({ ...prev, apiUsageAlerts: checked }))}
            />
          </div>
          <div className="flex items-center justify-between py-2 border-t border-border">
            <div className="space-y-0.5">
              <Label className="text-foreground">New features</Label>
              <p className="text-sm text-muted-foreground">Get notified about new SchemaLabs features</p>
            </div>
            <Switch
              checked={notifications.newFeatures}
              onCheckedChange={(checked) => setNotifications((prev) => ({ ...prev, newFeatures: checked }))}
            />
          </div>
          <div className="flex items-center justify-between py-2 border-t border-border">
            <div className="space-y-0.5">
              <Label className="text-foreground">Email notifications</Label>
              <p className="text-sm text-muted-foreground">Receive notifications via email</p>
            </div>
            <Switch
              checked={notifications.emailNotifications}
              onCheckedChange={(checked) => setNotifications((prev) => ({ ...prev, emailNotifications: checked }))}
            />
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
            {mockSessions.map((session) => (
              <div
                key={session.id}
                className={cn(
                  "flex items-center justify-between p-3 rounded-lg border",
                  session.current 
                    ? "border-[#0052CC]/30 bg-[#0052CC]/5" 
                    : "border-border bg-muted/30"
                )}
              >
                <div className="flex items-center gap-3">
                  <div className="text-muted-foreground">
                    {getDeviceIcon(session.deviceType)}
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <p className="text-sm font-medium text-foreground">{session.device}</p>
                      {session.current && (
                        <span className="rounded bg-[#0052CC]/20 px-1.5 py-0.5 text-[10px] font-medium text-[#0052CC] dark:text-[#2684FF]">
                          Current
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-muted-foreground">{session.location}</p>
                  </div>
                </div>
                <div className="text-right text-xs text-muted-foreground">
                  <p>Created: {formatDate(session.createdAt)}</p>
                  <p>Updated: {formatDate(session.updatedAt)}</p>
                </div>
              </div>
            ))}
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
          {/* User ID */}
          <div className="flex items-center justify-between py-2">
            <div className="space-y-0.5">
              <Label className="text-foreground">User ID</Label>
              <p className="text-xs text-muted-foreground font-mono">{userId}</p>
            </div>
            <Button size="sm" variant="outline" onClick={copyUserId} className="gap-2 bg-transparent">
              {copiedUserId ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
              {copiedUserId ? "Copied" : "Copy"}
            </Button>
          </div>

          {/* Change Password */}
          <div className="flex items-center justify-between py-2 border-t border-border">
            <div className="space-y-0.5">
              <Label className="text-foreground">Change password</Label>
              <p className="text-sm text-muted-foreground">Update your account password</p>
            </div>
            <Button 
              size="sm" 
              variant="outline" 
              onClick={() => setChangePasswordModalOpen(true)}
              className="gap-2 bg-transparent"
            >
              <Key className="h-4 w-4" />
              Change
            </Button>
          </div>

          {/* Log out all devices */}
          <div className="flex items-center justify-between py-2 border-t border-border">
            <div className="space-y-0.5">
              <Label className="text-foreground">Log out of all devices</Label>
              <p className="text-sm text-muted-foreground">Sign out from all active sessions</p>
            </div>
            <Button size="sm" variant="outline" className="gap-2 bg-transparent">
              <LogOut className="h-4 w-4" />
              Log out all
            </Button>
          </div>

          {/* Delete Account */}
          <div className="flex items-center justify-between py-2 border-t border-border">
            <div className="space-y-0.5">
              <Label className="text-red-500">Delete account</Label>
              <p className="text-sm text-muted-foreground">Permanently delete your account and all data</p>
            </div>
            <Button 
              size="sm" 
              variant="outline" 
              onClick={() => setDeleteAccountModalOpen(true)}
              className="gap-2 border-red-500/30 text-red-500 hover:bg-red-500/10 hover:text-red-500 bg-transparent"
            >
              <Trash2 className="h-4 w-4" />
              Delete
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Change Email Modal */}
      <Dialog open={changeEmailModalOpen} onOpenChange={setChangeEmailModalOpen}>
        <DialogContent className="border-border bg-card sm:max-w-[400px]">
          <DialogHeader>
            <DialogTitle className="text-foreground">Change Email</DialogTitle>
            <DialogDescription className="text-muted-foreground">
              {emailStep === "enter" 
                ? "Enter your new email address" 
                : "Enter the verification code sent to your new email"}
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            {emailStep === "enter" ? (
              <div className="space-y-2">
                <Label htmlFor="new-email" className="text-foreground">New Email</Label>
                <Input
                  id="new-email"
                  type="email"
                  placeholder="newemail@example.com"
                  value={newEmail}
                  onChange={(e) => setNewEmail(e.target.value)}
                  className="border-border bg-background"
                />
              </div>
            ) : (
              <div className="space-y-2">
                <Label htmlFor="verification-code" className="text-foreground">Verification Code</Label>
                <Input
                  id="verification-code"
                  placeholder="000000"
                  value={verificationCode}
                  onChange={(e) => setVerificationCode(e.target.value.replace(/\D/g, "").slice(0, 6))}
                  className="border-border bg-background text-center text-lg tracking-widest font-mono"
                  maxLength={6}
                />
                <p className="text-xs text-muted-foreground">Code sent to {newEmail}</p>
              </div>
            )}
          </div>
          <DialogFooter>
            <Button 
              variant="outline" 
              onClick={() => {
                setChangeEmailModalOpen(false)
                setEmailStep("enter")
                setNewEmail("")
                setVerificationCode("")
              }} 
              className="bg-transparent"
            >
              Cancel
            </Button>
            {emailStep === "enter" ? (
              <Button
                onClick={handleSendVerificationCode}
                disabled={!newEmail.includes("@")}
                className="bg-[#0052CC] text-white hover:bg-[#003D99]"
              >
                Send Code
              </Button>
            ) : (
              <Button
                onClick={handleVerifyEmail}
                disabled={verificationCode.length !== 6}
                className="bg-[#0052CC] text-white hover:bg-[#003D99]"
              >
                Verify
              </Button>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Change Password Modal */}
      <Dialog open={changePasswordModalOpen} onOpenChange={setChangePasswordModalOpen}>
        <DialogContent className="border-border bg-card sm:max-w-[400px]">
          <DialogHeader>
            <DialogTitle className="text-foreground">Change Password</DialogTitle>
            <DialogDescription className="text-muted-foreground">
              Enter your current password and choose a new one
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="current-password" className="text-foreground">Current Password</Label>
              <Input
                id="current-password"
                type="password"
                value={currentPassword}
                onChange={(e) => setCurrentPassword(e.target.value)}
                className="border-border bg-background"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="new-password" className="text-foreground">New Password</Label>
              <Input
                id="new-password"
                type="password"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                className="border-border bg-background"
              />
              <p className="text-xs text-muted-foreground">Minimum 8 characters</p>
            </div>
            <div className="space-y-2">
              <Label htmlFor="confirm-password" className="text-foreground">Confirm New Password</Label>
              <Input
                id="confirm-password"
                type="password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                className="border-border bg-background"
              />
              {confirmPassword && newPassword !== confirmPassword && (
                <p className="text-xs text-red-500">Passwords do not match</p>
              )}
            </div>
          </div>
          <DialogFooter>
            <Button 
              variant="outline" 
              onClick={() => {
                setChangePasswordModalOpen(false)
                setCurrentPassword("")
                setNewPassword("")
                setConfirmPassword("")
              }} 
              className="bg-transparent"
            >
              Cancel
            </Button>
            <Button
              onClick={handleChangePassword}
              disabled={!currentPassword || newPassword.length < 8 || newPassword !== confirmPassword}
              className="bg-[#0052CC] text-white hover:bg-[#003D99]"
            >
              Update Password
            </Button>
          </DialogFooter>
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
              This action cannot be undone. This will permanently delete your account, all your models, 
              datasets, and API keys.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-3">
              <p className="text-sm text-red-500">
                You will lose access to:
              </p>
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
              <Input
                id="delete-confirm"
                value={deleteConfirmText}
                onChange={(e) => setDeleteConfirmText(e.target.value)}
                className="border-border bg-background"
                placeholder="DELETE"
              />
            </div>
          </div>
          <DialogFooter>
            <Button 
              variant="outline" 
              onClick={() => {
                setDeleteAccountModalOpen(false)
                setDeleteConfirmText("")
              }} 
              className="bg-transparent"
            >
              Cancel
            </Button>
            <Button
              onClick={handleDeleteAccount}
              disabled={deleteConfirmText !== "DELETE"}
              className="bg-red-500 text-white hover:bg-red-600"
            >
              Delete Account
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

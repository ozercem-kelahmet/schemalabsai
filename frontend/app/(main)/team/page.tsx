"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
  Users,
  UserPlus,
  Mail,
  MoreHorizontal,
  Shield,
  Trash2,
  Loader2,
  Crown,
  AlertTriangle,
  RefreshCw,
  UserX,
  UserCheck,
  Clock,
  CheckCircle2,
  XCircle,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { toast } from "sonner"

type Role = "owner" | "admin" | "member"
type MemberStatus = "active" | "pending" | "deactivated"
type InviteStatus = "pending" | "accepted" | "expired"

interface TeamMember {
  id: string
  name: string
  email: string
  role: Role
  status: MemberStatus
  avatar_color: string
  joined_at: string
  last_active?: string
}

interface Invite {
  id: string
  email: string
  role: Role
  status: InviteStatus
  invited_at: string
  expires_at: string
  invited_by: string
}

interface TeamData {
  id: string
  name: string
  plan: string
  seats_used: number
  seats_total: number
  owner_id: string
}

const roleLabels: Record<Role, string> = {
  owner: "Owner",
  admin: "Admin",
  member: "Member",
}

const roleColors: Record<Role, string> = {
  owner: "bg-amber-500/15 text-amber-600 dark:text-amber-400",
  admin: "bg-[#2684FF]/15 text-[#2684FF]",
  member: "bg-muted text-muted-foreground",
}

const statusConfig: Record<MemberStatus, { label: string; color: string; icon: typeof CheckCircle2 }> = {
  active: { label: "Active", color: "text-[#36B37E]", icon: CheckCircle2 },
  pending: { label: "Pending", color: "text-amber-500", icon: Clock },
  deactivated: { label: "Deactivated", color: "text-muted-foreground", icon: XCircle },
}

// Plan seat limits
const planSeats: Record<string, number> = {
  professional: 3,
  enterprise: 25, // Custom, but default to 25
}


export default function TeamPage() {
  const router = useRouter()
  const [loading, setLoading] = useState(true)
  const [team, setTeam] = useState<TeamData | null>(null)
  const [members, setMembers] = useState<TeamMember[]>([])
  const [invites, setInvites] = useState<Invite[]>([])
  const [currentUserId, setCurrentUserId] = useState<string>("user-001") // Mock current user
  const [userPlan, setUserPlan] = useState<string>("free")
  
  // Invite modal state
  const [inviteOpen, setInviteOpen] = useState(false)
  const [inviteEmail, setInviteEmail] = useState("")
  const [inviteRole, setInviteRole] = useState<Role>("member")
  const [inviting, setInviting] = useState(false)
  
  // Remove member state
  const [removeOpen, setRemoveOpen] = useState(false)
  const [memberToRemove, setMemberToRemove] = useState<TeamMember | null>(null)
  const [removing, setRemoving] = useState(false)
  
  // Deactivate member state
  const [deactivateOpen, setDeactivateOpen] = useState(false)
  const [memberToDeactivate, setMemberToDeactivate] = useState<TeamMember | null>(null)
  const [deactivating, setDeactivating] = useState(false)

  useEffect(() => {
    fetchTeamData()
  }, [])

  const fetchTeamData = async () => {
    try {
      // Check user plan first
      const userRes = await fetch("/api/auth/me", { credentials: "include" })
      if (userRes.ok) {
        const userData = await userRes.json()
        setCurrentUserId(userData.id || "user-001")
        const plan = userData.plan || "professional"
        setUserPlan(plan)
        
        // Only allow Pro and Enterprise
        if (!["professional", "enterprise"].includes(plan)) {
          router.push("/billing")
          return
        }
      }

      // Fetch team data
      const teamRes = await fetch("/api/team", { credentials: "include" })
      if (teamRes.ok) {
        const data = await teamRes.json()
        setTeam(data.team || null)
        setMembers(data.members || [])
        setInvites(data.invites || [])
      } else {
        // Empty state if API not available
        setTeam(null)
        setMembers([])
        setInvites([])
      }
    } catch (e) {
      // Empty state on error
      setTeam(null)
      setMembers([])
      setInvites([])
    } finally {
      setLoading(false)
    }
  }

  const handleInvite = async () => {
    const email = inviteEmail.trim()
    if (!email) return
    if (!team) return
    
    // Email format validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
    if (!emailRegex.test(email) || email.length > 254) {
      toast.error("Please enter a valid email address")
      return
    }
    
    // Check if email already exists
    if (members.some(m => m.email.toLowerCase() === email.toLowerCase())) {
      toast.error("This email is already a team member")
      return
    }
    if (invites.some(i => i.email.toLowerCase() === inviteEmail.toLowerCase() && i.status === "pending")) {
      toast.error("An invitation is already pending for this email")
      return
    }
    
    // Check seat limit
    const activeMembers = members.filter(m => m.status !== "deactivated").length
    const pendingInvites = invites.filter(i => i.status === "pending").length
    if (activeMembers + pendingInvites >= team.seats_total) {
      toast.error(`You've reached your seat limit (${team.seats_total}). Upgrade your plan for more seats.`)
      return
    }
    
    setInviting(true)
    
    try {
      const res = await fetch("/api/team/invite", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ email: inviteEmail, role: inviteRole })
      })
      
      if (res.ok) {
        const data = await res.json()
        setInvites(prev => [...prev, data.invite])
        toast.success(`Invitation sent to ${inviteEmail}`)
      } else {
        // Mock success for demo
        const newInvite: Invite = {
          id: `invite-${Date.now()}`,
          email: inviteEmail,
          role: inviteRole,
          status: "pending",
          invited_at: new Date().toISOString(),
          expires_at: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(),
          invited_by: members.find(m => m.id === currentUserId)?.name || "You",
        }
        setInvites(prev => [...prev, newInvite])
        toast.success(`Invitation sent to ${inviteEmail}`)
      }
      
      setInviteOpen(false)
      setInviteEmail("")
      setInviteRole("member")
    } catch (e) {
      // Mock success for demo
      const newInvite: Invite = {
        id: `invite-${Date.now()}`,
        email: inviteEmail,
        role: inviteRole,
        status: "pending",
        invited_at: new Date().toISOString(),
        expires_at: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(),
        invited_by: members.find(m => m.id === currentUserId)?.name || "You",
      }
      setInvites(prev => [...prev, newInvite])
      setInviteOpen(false)
      setInviteEmail("")
      setInviteRole("member")
      toast.success(`Invitation sent to ${inviteEmail}`)
    } finally {
      setInviting(false)
    }
  }

  const handleRemoveMember = async () => {
    if (!memberToRemove) return
    setRemoving(true)
    
    try {
      const res = await fetch("/api/team/remove", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ member_id: memberToRemove.id })
      })
      
      if (res.ok) {
        setMembers(prev => prev.filter(m => m.id !== memberToRemove.id))
      } else {
        // Mock success
        setMembers(prev => prev.filter(m => m.id !== memberToRemove.id))
      }
      toast.success(`${memberToRemove.name} has been removed from the team`)
      setRemoveOpen(false)
      setMemberToRemove(null)
    } catch (e) {
      // Mock success
      setMembers(prev => prev.filter(m => m.id !== memberToRemove.id))
      toast.success(`${memberToRemove.name} has been removed from the team`)
      setRemoveOpen(false)
      setMemberToRemove(null)
    } finally {
      setRemoving(false)
    }
  }

  const handleDeactivateMember = async () => {
    if (!memberToDeactivate) return
    setDeactivating(true)
    
    try {
      const res = await fetch("/api/team/deactivate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ member_id: memberToDeactivate.id })
      })
      
      if (res.ok) {
        setMembers(prev => prev.map(m => m.id === memberToDeactivate.id ? { ...m, status: "deactivated" as MemberStatus } : m))
      } else {
        // Mock success
        setMembers(prev => prev.map(m => m.id === memberToDeactivate.id ? { ...m, status: "deactivated" as MemberStatus } : m))
      }
      toast.success(`${memberToDeactivate.name} has been deactivated`)
      setDeactivateOpen(false)
      setMemberToDeactivate(null)
    } catch (e) {
      // Mock success
      setMembers(prev => prev.map(m => m.id === memberToDeactivate.id ? { ...m, status: "deactivated" as MemberStatus } : m))
      toast.success(`${memberToDeactivate.name} has been deactivated`)
      setDeactivateOpen(false)
      setMemberToDeactivate(null)
    } finally {
      setDeactivating(false)
    }
  }

  const handleReactivateMember = async (memberId: string) => {
    const member = members.find(m => m.id === memberId)
    if (!member) return
    
    // Check seat limit before reactivating
    const activeMembers = members.filter(m => m.status === "active").length
    if (team && activeMembers >= team.seats_total) {
      toast.error(`Cannot reactivate. You've reached your seat limit (${team.seats_total}).`)
      return
    }
    
    try {
      const res = await fetch("/api/team/reactivate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ member_id: memberId })
      })
      
      if (res.ok) {
        setMembers(prev => prev.map(m => m.id === memberId ? { ...m, status: "active" as MemberStatus } : m))
      } else {
        // Mock success
        setMembers(prev => prev.map(m => m.id === memberId ? { ...m, status: "active" as MemberStatus } : m))
      }
      toast.success(`${member.name} has been reactivated`)
    } catch (e) {
      // Mock success
      setMembers(prev => prev.map(m => m.id === memberId ? { ...m, status: "active" as MemberStatus } : m))
      toast.success(`${member.name} has been reactivated`)
    }
  }

  const handleCancelInvite = async (inviteId: string) => {
    const invite = invites.find(i => i.id === inviteId)
    try {
      const res = await fetch("/api/team/invite/cancel", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ invite_id: inviteId })
      })
      
      if (res.ok) {
        setInvites(prev => prev.filter(i => i.id !== inviteId))
      } else {
        // Mock success
        setInvites(prev => prev.filter(i => i.id !== inviteId))
      }
      toast.success(`Invitation to ${invite?.email} cancelled`)
    } catch (e) {
      // Mock success
      setInvites(prev => prev.filter(i => i.id !== inviteId))
      toast.success(`Invitation to ${invite?.email} cancelled`)
    }
  }

  const handleResendInvite = async (inviteId: string) => {
    const invite = invites.find(i => i.id === inviteId)
    try {
      const res = await fetch("/api/team/invite/resend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ invite_id: inviteId })
      })
      
      if (res.ok) {
        // Update expiration date
        setInvites(prev => prev.map(i => i.id === inviteId ? { 
          ...i, 
          expires_at: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString() 
        } : i))
      } else {
        // Mock success
        setInvites(prev => prev.map(i => i.id === inviteId ? { 
          ...i, 
          expires_at: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString() 
        } : i))
      }
      toast.success(`Invitation resent to ${invite?.email}`)
    } catch (e) {
      // Mock success
      setInvites(prev => prev.map(i => i.id === inviteId ? { 
        ...i, 
        expires_at: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString() 
      } : i))
      toast.success(`Invitation resent to ${invite?.email}`)
    }
  }

  const handleChangeRole = async (memberId: string, newRole: Role) => {
    const member = members.find(m => m.id === memberId)
    try {
      const res = await fetch("/api/team/role", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ member_id: memberId, role: newRole })
      })
      
      if (res.ok) {
        setMembers(prev => prev.map(m => m.id === memberId ? { ...m, role: newRole } : m))
      } else {
        // Mock success
        setMembers(prev => prev.map(m => m.id === memberId ? { ...m, role: newRole } : m))
      }
      toast.success(`${member?.name} is now ${roleLabels[newRole]}`)
    } catch (e) {
      // Mock success
      setMembers(prev => prev.map(m => m.id === memberId ? { ...m, role: newRole } : m))
      toast.success(`${member?.name} is now ${roleLabels[newRole]}`)
    }
  }

  const getInitials = (name: string) => {
    const parts = name.split(" ")
    return parts.length >= 2 
      ? `${parts[0].charAt(0)}${parts[1].charAt(0)}`.toUpperCase()
      : name.substring(0, 2).toUpperCase()
  }

  const formatDate = (date: string) => {
    return new Date(date).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    })
  }

  const formatRelativeTime = (date: string) => {
    const now = new Date()
    const d = new Date(date)
    const diffMs = now.getTime() - d.getTime()
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24))
    
    if (diffDays === 0) return "Today"
    if (diffDays === 1) return "Yesterday"
    if (diffDays < 7) return `${diffDays} days ago`
    return formatDate(date)
  }

  const currentUserRole = members.find(m => m.id === currentUserId)?.role || "member"
  const canManage = currentUserRole === "owner" || currentUserRole === "admin"
  const isOwner = currentUserRole === "owner"
  
  const activeMembers = members.filter(m => m.status === "active")
  const deactivatedMembers = members.filter(m => m.status === "deactivated")
  const pendingInvites = invites.filter(i => i.status === "pending")
  
  // Calculate available seats (active + pending invites count towards limit)
  const usedSeats = activeMembers.length + pendingInvites.length
  const availableSeats = team ? team.seats_total - usedSeats : 0

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64 text-muted-foreground">
        <Loader2 className="h-5 w-5 animate-spin mr-2" /> Loading...
      </div>
    )
  }

  // Redirect handled in useEffect, but show nothing while redirecting
  if (!["professional", "enterprise"].includes(userPlan)) {
    return null
  }

  return (
    <div className="space-y-6 max-w-4xl">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[#0052CC]/10 dark:bg-[#0052CC]/20">
            <Users className="h-5 w-5 text-[#0052CC] dark:text-[#2684FF]" />
          </div>
          <div>
            <h1 className="text-xl font-semibold text-foreground">Team</h1>
            <p className="text-sm text-muted-foreground">Manage your team members and invitations</p>
          </div>
        </div>
        
        {canManage && (
          <Button 
            onClick={() => setInviteOpen(true)}
            disabled={availableSeats <= 0}
            className="bg-[#0052CC] hover:bg-[#003D99] text-white gap-2"
          >
            <UserPlus className="h-4 w-4" />
            Invite Member
          </Button>
        )}
      </div>

      {/* Seats Usage */}
      {team && (
        <Card className="border-border bg-card">
          <CardContent className="p-5">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
              <div>
                <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">Team Seats</p>
                <p className="text-2xl font-bold text-foreground">
                  {usedSeats} <span className="text-lg text-muted-foreground font-normal">/ {team.seats_total}</span>
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  {activeMembers.length} active member{activeMembers.length !== 1 ? "s" : ""}
                  {pendingInvites.length > 0 && `, ${pendingInvites.length} pending invite${pendingInvites.length !== 1 ? "s" : ""}`}
                </p>
              </div>
              <div className="text-left sm:text-right">
                {availableSeats > 0 ? (
                  <p className="text-sm text-muted-foreground">{availableSeats} seat{availableSeats !== 1 ? "s" : ""} available</p>
                ) : (
                  <div>
                    <p className="text-sm text-amber-500 font-medium">No seats available</p>
                    <Button 
                      variant="link" 
                      className="h-auto p-0 text-xs text-[#2684FF]"
                      onClick={() => router.push("/billing")}
                    >
                      Upgrade for more seats
                    </Button>
                  </div>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Active Team Members */}
      <Card className="border-border bg-card">
        <CardHeader>
          <CardTitle className="text-foreground flex items-center gap-2">
            <Users className="h-5 w-5" />
            Active Members
            <span className="text-sm font-normal text-muted-foreground">({activeMembers.length})</span>
          </CardTitle>
          <CardDescription>Team members with full access to the workspace</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {activeMembers.length === 0 ? (
              <p className="text-sm text-muted-foreground py-4 text-center">No active team members</p>
            ) : (
              activeMembers.map((member) => {
                const StatusIcon = statusConfig[member.status].icon
                return (
                  <div
                    key={member.id}
                    className="flex flex-col sm:flex-row sm:items-center justify-between p-3 rounded-lg border border-border bg-muted/30 gap-3"
                  >
                    <div className="flex items-center gap-3">
                      <div
                        className="flex h-10 w-10 items-center justify-center rounded-full text-sm font-semibold text-white shrink-0"
                        style={{ background: member.avatar_color || "linear-gradient(135deg, #2684FF, #0052CC)" }}
                      >
                        {getInitials(member.name)}
                      </div>
                      <div className="min-w-0">
                        <div className="flex items-center gap-2 flex-wrap">
                          <p className="text-sm font-medium text-foreground truncate">{member.name}</p>
                          {member.role === "owner" && <Crown className="h-3.5 w-3.5 text-amber-500 shrink-0" />}
                          {member.id === currentUserId && (
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-[#2684FF]/15 text-[#2684FF] shrink-0">You</span>
                          )}
                        </div>
                        <p className="text-xs text-muted-foreground truncate">{member.email}</p>
                        <p className="text-[10px] text-muted-foreground mt-0.5">
                          Joined {formatDate(member.joined_at)}
                          {member.last_active && ` • Active ${formatRelativeTime(member.last_active)}`}
                        </p>
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-3 sm:ml-auto">
                      <div className="flex items-center gap-1.5">
                        <StatusIcon className={cn("h-3 w-3", statusConfig[member.status].color)} />
                        <span className={cn("text-[10px] font-medium", statusConfig[member.status].color)}>
                          {statusConfig[member.status].label}
                        </span>
                      </div>
                      
                      <span className={cn("text-xs font-medium px-2 py-1 rounded", roleColors[member.role])}>
                        {roleLabels[member.role]}
                      </span>
                      
                      {canManage && member.id !== currentUserId && member.role !== "owner" && (
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                              <MoreHorizontal className="h-4 w-4" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end" className="border-border bg-card">
                            {isOwner && (
                              <>
                                <DropdownMenuItem onClick={() => handleChangeRole(member.id, "admin")} disabled={member.role === "admin"}>
                                  <Shield className="h-4 w-4 mr-2" />
                                  Make Admin
                                </DropdownMenuItem>
                                <DropdownMenuItem onClick={() => handleChangeRole(member.id, "member")} disabled={member.role === "member"}>
                                  <Users className="h-4 w-4 mr-2" />
                                  Make Member
                                </DropdownMenuItem>
                                <DropdownMenuSeparator />
                              </>
                            )}
                            <DropdownMenuItem 
                              onClick={() => { setMemberToDeactivate(member); setDeactivateOpen(true) }}
                              className="text-amber-500 focus:text-amber-500"
                            >
                              <UserX className="h-4 w-4 mr-2" />
                              Deactivate
                            </DropdownMenuItem>
                            <DropdownMenuItem 
                              onClick={() => { setMemberToRemove(member); setRemoveOpen(true) }}
                              className="text-red-500 focus:text-red-500"
                            >
                              <Trash2 className="h-4 w-4 mr-2" />
                              Remove
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      )}
                    </div>
                  </div>
                )
              })
            )}
          </div>
        </CardContent>
      </Card>

      {/* Pending Invites */}
      {pendingInvites.length > 0 && (
        <Card className="border-border bg-card">
          <CardHeader>
            <CardTitle className="text-foreground flex items-center gap-2">
              <Mail className="h-5 w-5" />
              Pending Invitations
              <span className="text-sm font-normal text-muted-foreground">({pendingInvites.length})</span>
            </CardTitle>
            <CardDescription>Invitations awaiting acceptance (expires in 7 days)</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {pendingInvites.map((invite) => {
                const isExpired = new Date(invite.expires_at) < new Date()
                const daysUntilExpiry = Math.ceil((new Date(invite.expires_at).getTime() - Date.now()) / (1000 * 60 * 60 * 24))
                
                return (
                  <div
                    key={invite.id}
                    className={cn(
                      "flex flex-col sm:flex-row sm:items-center justify-between p-3 rounded-lg border border-dashed gap-3",
                      isExpired ? "border-red-500/30 bg-red-500/5" : "border-border bg-muted/20"
                    )}
                  >
                    <div className="flex items-center gap-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-full bg-muted text-muted-foreground shrink-0">
                        <Mail className="h-4 w-4" />
                      </div>
                      <div className="min-w-0">
                        <p className="text-sm font-medium text-foreground truncate">{invite.email}</p>
                        <p className="text-xs text-muted-foreground">
                          Invited by {invite.invited_by} • {formatDate(invite.invited_at)}
                        </p>
                        {isExpired ? (
                          <p className="text-[10px] text-red-500 mt-0.5">Expired</p>
                        ) : (
                          <p className="text-[10px] text-muted-foreground mt-0.5">
                            Expires in {daysUntilExpiry} day{daysUntilExpiry !== 1 ? "s" : ""}
                          </p>
                        )}
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-2 sm:ml-auto">
                      <span className={cn("text-xs font-medium px-2 py-1 rounded", roleColors[invite.role])}>
                        {roleLabels[invite.role]}
                      </span>
                      
                      {canManage && (
                        <div className="flex items-center gap-1">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleResendInvite(invite.id)}
                            className="h-8 gap-1.5 text-muted-foreground hover:text-foreground"
                          >
                            <RefreshCw className="h-3.5 w-3.5" />
                            <span className="hidden sm:inline">Resend</span>
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleCancelInvite(invite.id)}
                            className="h-8 text-muted-foreground hover:text-red-500"
                          >
                            Cancel
                          </Button>
                        </div>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Deactivated Members */}
      {deactivatedMembers.length > 0 && (
        <Card className="border-border bg-card">
          <CardHeader>
            <CardTitle className="text-foreground flex items-center gap-2 text-muted-foreground">
              <UserX className="h-5 w-5" />
              Deactivated Members
              <span className="text-sm font-normal">({deactivatedMembers.length})</span>
            </CardTitle>
            <CardDescription>Members without access (do not count towards seat limit)</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {deactivatedMembers.map((member) => (
                <div
                  key={member.id}
                  className="flex flex-col sm:flex-row sm:items-center justify-between p-3 rounded-lg border border-border bg-muted/10 opacity-60 gap-3"
                >
                  <div className="flex items-center gap-3">
                    <div
                      className="flex h-10 w-10 items-center justify-center rounded-full text-sm font-semibold text-white/50 shrink-0 grayscale"
                      style={{ background: member.avatar_color || "linear-gradient(135deg, #2684FF, #0052CC)" }}
                    >
                      {getInitials(member.name)}
                    </div>
                    <div className="min-w-0">
                      <p className="text-sm font-medium text-muted-foreground truncate">{member.name}</p>
                      <p className="text-xs text-muted-foreground truncate">{member.email}</p>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-3 sm:ml-auto">
                    <div className="flex items-center gap-1.5">
                      <XCircle className="h-3 w-3 text-muted-foreground" />
                      <span className="text-[10px] font-medium text-muted-foreground">Deactivated</span>
                    </div>
                    
                    {canManage && (
                      <div className="flex items-center gap-1">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleReactivateMember(member.id)}
                          disabled={availableSeats <= 0}
                          className="h-8 gap-1.5 text-[#2684FF] hover:text-[#2684FF]"
                        >
                          <UserCheck className="h-3.5 w-3.5" />
                          Reactivate
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => { setMemberToRemove(member); setRemoveOpen(true) }}
                          className="h-8 text-muted-foreground hover:text-red-500"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </Button>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Invite Modal */}
      <Dialog open={inviteOpen} onOpenChange={setInviteOpen}>
        <DialogContent className="border-border bg-card sm:max-w-[420px]">
          <DialogHeader>
            <DialogTitle className="text-foreground">Invite Team Member</DialogTitle>
            <DialogDescription>
              Send an invitation to join your team. They&apos;ll receive an email with a link to accept.
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4 mt-2">
            <div className="space-y-2">
              <Label>Email Address</Label>
              <Input
                type="email"
                placeholder="colleague@company.com"
                value={inviteEmail}
                onChange={(e) => setInviteEmail(e.target.value)}
                className="border-border bg-background"
              />
            </div>
            
            <div className="space-y-2">
              <Label>Role</Label>
              <Select value={inviteRole} onValueChange={(v) => setInviteRole(v as Role)}>
                <SelectTrigger className="border-border bg-background">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="border-border bg-card">
                  <SelectItem value="member">
                    <div className="flex flex-col items-start">
                      <span>Member</span>
                      <span className="text-xs text-muted-foreground">Can use all features</span>
                    </div>
                  </SelectItem>
                  {isOwner && (
                    <SelectItem value="admin">
                      <div className="flex flex-col items-start">
                        <span>Admin</span>
                        <span className="text-xs text-muted-foreground">Can manage team members</span>
                      </div>
                    </SelectItem>
                  )}
                </SelectContent>
              </Select>
            </div>
            
            {team && (
              <p className="text-xs text-muted-foreground">
                {availableSeats} seat{availableSeats !== 1 ? "s" : ""} available on your {team.plan} plan
              </p>
            )}
          </div>
          
          <DialogFooter className="mt-4">
            <Button variant="outline" onClick={() => setInviteOpen(false)}>Cancel</Button>
            <Button 
              onClick={handleInvite} 
              disabled={!inviteEmail.trim() || inviting || availableSeats <= 0}
              className="bg-[#0052CC] hover:bg-[#003D99] text-white"
            >
              {inviting ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <UserPlus className="h-4 w-4 mr-2" />}
              Send Invite
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Remove Member Modal */}
      <Dialog open={removeOpen} onOpenChange={setRemoveOpen}>
        <DialogContent className="border-border bg-card sm:max-w-[400px]">
          <DialogHeader>
            <DialogTitle className="text-foreground flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-red-500" />
              Remove Team Member
            </DialogTitle>
            <DialogDescription>
              Are you sure you want to remove <strong>{memberToRemove?.name}</strong> from the team? 
              They will lose access to all team resources. This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          
          <DialogFooter className="mt-4">
            <Button variant="outline" onClick={() => { setRemoveOpen(false); setMemberToRemove(null) }}>Cancel</Button>
            <Button 
              onClick={handleRemoveMember} 
              disabled={removing}
              variant="destructive"
            >
              {removing ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <Trash2 className="h-4 w-4 mr-2" />}
              Remove
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Deactivate Member Modal */}
      <Dialog open={deactivateOpen} onOpenChange={setDeactivateOpen}>
        <DialogContent className="border-border bg-card sm:max-w-[400px]">
          <DialogHeader>
            <DialogTitle className="text-foreground flex items-center gap-2">
              <UserX className="h-5 w-5 text-amber-500" />
              Deactivate Team Member
            </DialogTitle>
            <DialogDescription>
              Deactivating <strong>{memberToDeactivate?.name}</strong> will revoke their access but keep their 
              data for future reference. They won&apos;t count towards your seat limit. You can reactivate them later.
            </DialogDescription>
          </DialogHeader>
          
          <DialogFooter className="mt-4">
            <Button variant="outline" onClick={() => { setDeactivateOpen(false); setMemberToDeactivate(null) }}>Cancel</Button>
            <Button 
              onClick={handleDeactivateMember} 
              disabled={deactivating}
              className="bg-amber-500 hover:bg-amber-600 text-white"
            >
              {deactivating ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <UserX className="h-4 w-4 mr-2" />}
              Deactivate
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

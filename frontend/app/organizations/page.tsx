"use client"

import { useState, useEffect } from "react"
import { Sidebar } from "@/components/sidebar"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Users, Plus, Settings, Trash2, Mail, Crown, Shield, User, Check, X, Copy, Link, MoreVertical, Building2, UserPlus, ChevronRight, Clock } from "lucide-react"

const API_BASE = "/api"

interface Organization {
  id: string
  name: string
  slug: string
  plan: string
  max_members: number
  member_count: number
  role: string
  is_owner: boolean
  created_at: string
}

interface Member {
  id: string
  email: string
  name: string
  role: string
  status: string
  user_id: string | null
  joined_at: string
  invited_at: string
}

export default function OrganizationsPage() {
  const [organizations, setOrganizations] = useState<Organization[]>([])
  const [selectedOrg, setSelectedOrg] = useState<Organization | null>(null)
  const [members, setMembers] = useState<Member[]>([])
  const [loading, setLoading] = useState(true)
  const [notification, setNotification] = useState<{type: string, message: string} | null>(null)

  // Dialogs
  const [showCreateDialog, setShowCreateDialog] = useState(false)
  const [showInviteDialog, setShowInviteDialog] = useState(false)
  const [showSettingsDialog, setShowSettingsDialog] = useState(false)
  const [showDeleteDialog, setShowDeleteDialog] = useState(false)

  // Form states
  const [newOrgName, setNewOrgName] = useState("")
  const [inviteEmail, setInviteEmail] = useState("")
  const [inviteRole, setInviteRole] = useState("member")
  const [editOrgName, setEditOrgName] = useState("")

  const [userPlan, setUserPlan] = useState("free")
  const [maxTeams, setMaxTeams] = useState(1)

  useEffect(() => {
    fetchOrganizations()
    fetchUserPlan()
  }, [])

  useEffect(() => {
    if (notification) {
      const t = setTimeout(() => setNotification(null), 3000)
      return () => clearTimeout(t)
    }
  }, [notification])

  const notify = (type: string, message: string) => setNotification({ type, message })

  const fetchUserPlan = async () => {
    try {
      const res = await fetch(API_BASE + "/auth/me", { credentials: "include" })
      const data = await res.json()
      setUserPlan(data.plan || "free")
      setMaxTeams(data.plan === "enterprise" ? 100 : data.plan === "pro" ? 5 : 1)
    } catch (e) { console.error(e) }
  }

  const fetchOrganizations = async () => {
    try {
      const res = await fetch(API_BASE + "/organizations", { credentials: "include" })
      const data = await res.json()
      setOrganizations(Array.isArray(data) ? data : [])
      if (data && data.length > 0 && !selectedOrg) {
        selectOrganization(data[0])
      }
    } catch (e) { console.error(e) }
    setLoading(false)
  }

  const selectOrganization = async (org: Organization) => {
    setSelectedOrg(org)
    setEditOrgName(org.name)
    try {
      const res = await fetch(API_BASE + "/organizations/" + org.id + "/members", { credentials: "include" })
      const data = await res.json()
      setMembers(Array.isArray(data) ? data : [])
    } catch (e) { console.error(e) }
  }

  const createOrganization = async () => {
    if (!newOrgName.trim()) return
    try {
      const res = await fetch(API_BASE + "/organizations", {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: newOrgName })
      })
      if (!res.ok) {
        const error = await res.text()
        notify("error", error)
        return
      }
      notify("success", "Organization created")
      setShowCreateDialog(false)
      setNewOrgName("")
      fetchOrganizations()
    } catch (e) { notify("error", "Failed to create organization") }
  }

  const inviteMember = async () => {
    if (!inviteEmail.trim() || !selectedOrg) return
    try {
      const res = await fetch(API_BASE + "/organizations/" + selectedOrg.id + "/members", {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: inviteEmail, role: inviteRole })
      })
      if (!res.ok) {
        const error = await res.text()
        notify("error", error)
        return
      }
      notify("success", "Invitation sent to " + inviteEmail)
      setShowInviteDialog(false)
      setInviteEmail("")
      setInviteRole("member")
      selectOrganization(selectedOrg)
    } catch (e) { notify("error", "Failed to send invitation") }
  }

  const updateOrganization = async () => {
    if (!editOrgName.trim() || !selectedOrg) return
    try {
      await fetch(API_BASE + "/organizations/" + selectedOrg.id, {
        method: "PUT",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: editOrgName })
      })
      notify("success", "Organization updated")
      setShowSettingsDialog(false)
      fetchOrganizations()
    } catch (e) { notify("error", "Failed to update") }
  }

  const deleteOrganization = async () => {
    if (!selectedOrg) return
    try {
      await fetch(API_BASE + "/organizations/" + selectedOrg.id, {
        method: "DELETE",
        credentials: "include"
      })
      notify("success", "Organization deleted")
      setShowDeleteDialog(false)
      setSelectedOrg(null)
      fetchOrganizations()
    } catch (e) { notify("error", "Failed to delete") }
  }

  const updateMemberRole = async (memberId: string, newRole: string) => {
    if (!selectedOrg) return
    try {
      await fetch(API_BASE + "/organizations/" + selectedOrg.id + "/members/" + memberId, {
        method: "PUT",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ role: newRole })
      })
      notify("success", "Role updated")
      selectOrganization(selectedOrg)
    } catch (e) { notify("error", "Failed to update role") }
  }

  const removeMember = async (memberId: string) => {
    if (!selectedOrg) return
    try {
      await fetch(API_BASE + "/organizations/" + selectedOrg.id + "/members/" + memberId, {
        method: "DELETE",
        credentials: "include"
      })
      notify("success", "Member removed")
      selectOrganization(selectedOrg)
    } catch (e) { notify("error", "Failed to remove member") }
  }

  const getInitials = (name: string, email: string) => {
    if (name) return name.split(" ").map(n => n[0]).join("").toUpperCase().slice(0, 2)
    return email?.slice(0, 2).toUpperCase() || "U"
  }

  const getRoleIcon = (role: string) => {
    switch (role) {
      case "owner": return <Crown className="h-3 w-3" />
      case "admin": return <Shield className="h-3 w-3" />
      default: return <User className="h-3 w-3" />
    }
  }

  if (loading) {
    return <Sidebar><div className="flex-1 flex items-center justify-center"><p className="text-muted-foreground">Loading...</p></div></Sidebar>
  }

  return (
    <Sidebar>
      <div className="flex-1 overflow-auto">
        {/* Notification */}
        {notification && (
          <div className={`fixed top-4 right-4 z-50 px-4 py-2.5 rounded-lg shadow-lg flex items-center gap-2 text-sm font-medium ${notification.type === "success" ? "bg-primary text-primary-foreground" : "bg-destructive text-destructive-foreground"}`}>
            {notification.type === "success" ? <Check className="h-4 w-4" /> : <X className="h-4 w-4" />}
            {notification.message}
          </div>
        )}

        {/* Header */}
        <div className="border-b sticky top-0 z-10 bg-background">
          <div className="px-6 py-4">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-xl font-semibold">Organizations</h1>
                <p className="text-sm text-muted-foreground">Manage your teams and members</p>
              </div>
              <div className="flex items-center gap-2">
                <Badge variant="outline">{organizations.length} / {maxTeams} teams</Badge>
                <Button onClick={() => setShowCreateDialog(true)} disabled={organizations.length >= maxTeams}>
                  <Plus className="h-4 w-4 mr-2" />New Team
                </Button>
              </div>
            </div>
          </div>
        </div>

        <div className="p-6">
          {organizations.length === 0 ? (
            <Card className="max-w-lg mx-auto mt-12">
              <CardContent className="pt-12 pb-12 text-center">
                <Building2 className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <h2 className="text-lg font-semibold mb-2">No organizations yet</h2>
                <p className="text-sm text-muted-foreground mb-6">Create your first team to start collaborating with others.</p>
                <Button onClick={() => setShowCreateDialog(true)}>
                  <Plus className="h-4 w-4 mr-2" />Create Organization
                </Button>
              </CardContent>
            </Card>
          ) : (
            <div className="grid grid-cols-4 gap-6">
              {/* Org List */}
              <div className="space-y-2">
                <p className="text-sm font-medium text-muted-foreground mb-3">Your Teams</p>
                {organizations.map((org) => (
                  <button
                    key={org.id}
                    onClick={() => selectOrganization(org)}
                    className={`w-full text-left p-3 rounded-lg border transition-colors ${selectedOrg?.id === org.id ? "border-primary bg-primary/5" : "hover:bg-muted"}`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="h-9 w-9 rounded-lg bg-muted flex items-center justify-center font-semibold text-sm">
                          {org.name.slice(0, 2).toUpperCase()}
                        </div>
                        <div>
                          <p className="font-medium text-sm">{org.name}</p>
                          <p className="text-xs text-muted-foreground">{org.member_count} members</p>
                        </div>
                      </div>
                      {org.is_owner && <Crown className="h-4 w-4 text-muted-foreground" />}
                    </div>
                  </button>
                ))}
                {organizations.length < maxTeams && (
                  <button
                    onClick={() => setShowCreateDialog(true)}
                    className="w-full text-left p-3 rounded-lg border border-dashed hover:bg-muted transition-colors"
                  >
                    <div className="flex items-center gap-3 text-muted-foreground">
                      <Plus className="h-5 w-5" />
                      <span className="text-sm">Add team</span>
                    </div>
                  </button>
                )}
              </div>

              {/* Org Detail */}
              {selectedOrg && (
                <div className="col-span-3 space-y-6">
                  {/* Org Header */}
                  <Card>
                    <CardContent className="pt-6">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4">
                          <div className="h-14 w-14 rounded-xl bg-muted flex items-center justify-center font-bold text-xl">
                            {selectedOrg.name.slice(0, 2).toUpperCase()}
                          </div>
                          <div>
                            <h2 className="text-xl font-semibold">{selectedOrg.name}</h2>
                            <div className="flex items-center gap-2 mt-1">
                              <Badge variant="outline">{selectedOrg.role}</Badge>
                              <span className="text-sm text-muted-foreground">{selectedOrg.member_count} / {selectedOrg.max_members} members</span>
                            </div>
                          </div>
                        </div>
                        {selectedOrg.is_owner && (
                          <div className="flex items-center gap-2">
                            <Button variant="outline" size="sm" onClick={() => setShowSettingsDialog(true)}>
                              <Settings className="h-4 w-4 mr-2" />Settings
                            </Button>
                          </div>
                        )}
                      </div>
                    </CardContent>
                  </Card>

                  {/* Members */}
                  <Card>
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        <div>
                          <CardTitle>Members</CardTitle>
                          <CardDescription>People in this organization</CardDescription>
                        </div>
                        {(selectedOrg.is_owner || selectedOrg.role === "admin") && selectedOrg.member_count < selectedOrg.max_members && (
                          <Button size="sm" onClick={() => setShowInviteDialog(true)}>
                            <UserPlus className="h-4 w-4 mr-2" />Invite
                          </Button>
                        )}
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-1">
                        {members.map((member) => (
                          <div key={member.id} className="flex items-center justify-between py-3 px-3 rounded-lg hover:bg-muted/50">
                            <div className="flex items-center gap-3">
                              <Avatar className="h-9 w-9">
                                <AvatarFallback>{getInitials(member.name, member.email)}</AvatarFallback>
                              </Avatar>
                              <div>
                                <div className="flex items-center gap-2">
                                  <p className="font-medium text-sm">{member.name || member.email.split("@")[0]}</p>
                                  {member.status === "pending" && (
                                    <Badge variant="secondary" className="text-xs">
                                      <Clock className="h-3 w-3 mr-1" />Pending
                                    </Badge>
                                  )}
                                </div>
                                <p className="text-xs text-muted-foreground">{member.email}</p>
                              </div>
                            </div>
                            <div className="flex items-center gap-2">
                              <Badge variant={member.role === "owner" ? "default" : "outline"} className="gap-1">
                                {getRoleIcon(member.role)}
                                {member.role}
                              </Badge>
                              {selectedOrg.is_owner && member.role !== "owner" && (
                                <div className="flex items-center gap-1">
                                  <Select value={member.role} onValueChange={(val) => updateMemberRole(member.id, val)}>
                                    <SelectTrigger className="h-8 w-24">
                                      <SelectValue />
                                    </SelectTrigger>
                                    <SelectContent>
                                      <SelectItem value="admin">Admin</SelectItem>
                                      <SelectItem value="member">Member</SelectItem>
                                    </SelectContent>
                                  </Select>
                                  <Button size="sm" variant="ghost" onClick={() => removeMember(member.id)}>
                                    <Trash2 className="h-4 w-4" />
                                  </Button>
                                </div>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>

                  {/* Quick Stats */}
                  <div className="grid grid-cols-3 gap-4">
                    <Card>
                      <CardContent className="pt-6">
                        <p className="text-sm text-muted-foreground">Members</p>
                        <p className="text-2xl font-bold">{members.filter(m => m.status === "active" || m.role === "owner").length}</p>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="pt-6">
                        <p className="text-sm text-muted-foreground">Pending</p>
                        <p className="text-2xl font-bold">{members.filter(m => m.status === "pending").length}</p>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="pt-6">
                        <p className="text-sm text-muted-foreground">Available Seats</p>
                        <p className="text-2xl font-bold">{selectedOrg.max_members - selectedOrg.member_count}</p>
                      </CardContent>
                    </Card>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Create Dialog */}
        <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Create Organization</DialogTitle>
              <DialogDescription>Create a new team to collaborate with others.</DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div>
                <Label>Organization Name</Label>
                <Input 
                  value={newOrgName} 
                  onChange={(e) => setNewOrgName(e.target.value)} 
                  placeholder="My Team" 
                  className="mt-2"
                />
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setShowCreateDialog(false)}>Cancel</Button>
              <Button onClick={createOrganization}>Create</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Invite Dialog */}
        <Dialog open={showInviteDialog} onOpenChange={setShowInviteDialog}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Invite Member</DialogTitle>
              <DialogDescription>Send an invitation to join {selectedOrg?.name}.</DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div>
                <Label>Email Address</Label>
                <Input 
                  type="email"
                  value={inviteEmail} 
                  onChange={(e) => setInviteEmail(e.target.value)} 
                  placeholder="colleague@company.com" 
                  className="mt-2"
                />
              </div>
              <div>
                <Label>Role</Label>
                <Select value={inviteRole} onValueChange={setInviteRole}>
                  <SelectTrigger className="mt-2">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="admin">Admin - Can manage members</SelectItem>
                    <SelectItem value="member">Member - Standard access</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setShowInviteDialog(false)}>Cancel</Button>
              <Button onClick={inviteMember}><Mail className="h-4 w-4 mr-2" />Send Invite</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Settings Dialog */}
        <Dialog open={showSettingsDialog} onOpenChange={setShowSettingsDialog}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Organization Settings</DialogTitle>
              <DialogDescription>Manage your organization settings.</DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div>
                <Label>Organization Name</Label>
                <Input 
                  value={editOrgName} 
                  onChange={(e) => setEditOrgName(e.target.value)} 
                  className="mt-2"
                />
              </div>
              <div className="pt-4 border-t">
                <p className="text-sm font-medium text-destructive">Danger Zone</p>
                <p className="text-sm text-muted-foreground mt-1">Deleting an organization will remove all members and cannot be undone.</p>
                <Button variant="destructive" className="mt-3" onClick={() => { setShowSettingsDialog(false); setShowDeleteDialog(true); }}>
                  <Trash2 className="h-4 w-4 mr-2" />Delete Organization
                </Button>
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setShowSettingsDialog(false)}>Cancel</Button>
              <Button onClick={updateOrganization}>Save Changes</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Delete Dialog */}
        <Dialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Delete Organization</DialogTitle>
              <DialogDescription>Are you sure you want to delete {selectedOrg?.name}? This action cannot be undone.</DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <Button variant="outline" onClick={() => setShowDeleteDialog(false)}>Cancel</Button>
              <Button variant="destructive" onClick={deleteOrganization}>Delete</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    </Sidebar>
  )
}

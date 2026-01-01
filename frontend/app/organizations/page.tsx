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
import { Users, Plus, Settings, Trash2, Mail, Crown, Shield, User, Check, X, Building2, UserPlus, Clock, FolderPlus, Layers } from "lucide-react"

const API_BASE = ""

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

interface Team {
  id: string
  name: string
  organization_id: string
  member_count: number
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
  const [organization, setOrganization] = useState<Organization | null>(null)
  const [teams, setTeams] = useState<Team[]>([])
  const [members, setMembers] = useState<Member[]>([])
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState("teams")
  const [notification, setNotification] = useState<{type: string, message: string} | null>(null)

  const [showCreateOrgDialog, setShowCreateOrgDialog] = useState(false)
  const [showCreateTeamDialog, setShowCreateTeamDialog] = useState(false)
  const [showInviteDialog, setShowInviteDialog] = useState(false)
  const [showSettingsDialog, setShowSettingsDialog] = useState(false)
  const [showDeleteDialog, setShowDeleteDialog] = useState(false)

  const [newOrgName, setNewOrgName] = useState("")
  const [newTeamName, setNewTeamName] = useState("")
  const [inviteEmail, setInviteEmail] = useState("")
  const [inviteRole, setInviteRole] = useState("member")
  const [editOrgName, setEditOrgName] = useState("")

  const [userPlan, setUserPlan] = useState("free")
  const maxTeamsPerOrg = userPlan === "enterprise" ? 100 : userPlan === "pro" ? 20 : 3

  useEffect(() => {
    fetchOrganization()
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
      const res = await fetch(API_BASE + "/api/auth/me", { credentials: "include" })
      if (!res.ok) return
      const data = await res.json()
      setUserPlan(data.plan || "free")
    } catch (e) { console.error(e) }
  }

  const fetchOrganization = async () => {
    try {
      const res = await fetch(API_BASE + "/api/organizations", { credentials: "include" })
      if (!res.ok) { setLoading(false); return }
      const data = await res.json()
      if (Array.isArray(data) && data.length > 0) {
        setOrganization(data[0])
        setEditOrgName(data[0].name)
        fetchMembers(data[0].id)
        fetchTeams(data[0].id)
      }
    } catch (e) { console.error(e) }
    setLoading(false)
  }

  const fetchMembers = async (orgId: string) => {
    try {
      const res = await fetch(API_BASE + "/api/organizations/" + orgId + "/members", { credentials: "include" })
      if (!res.ok) return
      const data = await res.json()
      setMembers(Array.isArray(data) ? data : [])
    } catch (e) { console.error(e) }
  }

  const fetchTeams = async (orgId: string) => {
    try {
      const res = await fetch(API_BASE + "/api/organizations/" + orgId + "/teams", { credentials: "include" })
      if (!res.ok) { setTeams([]); return }
      const data = await res.json()
      setTeams(Array.isArray(data) ? data : [])
    } catch (e) { setTeams([]) }
  }

  const createOrganization = async () => {
    if (!newOrgName.trim()) return
    try {
      const res = await fetch(API_BASE + "/api/organizations", {
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
      setShowCreateOrgDialog(false)
      setNewOrgName("")
      fetchOrganization()
    } catch (e) { notify("error", "Failed to create organization") }
  }

  const createTeam = async () => {
    if (!newTeamName.trim() || !organization) return
    try {
      const res = await fetch(API_BASE + "/api/organizations/" + organization.id + "/teams", {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: newTeamName })
      })
      if (!res.ok) {
        const error = await res.text()
        notify("error", error)
        return
      }
      notify("success", "Team created")
      setShowCreateTeamDialog(false)
      setNewTeamName("")
      fetchTeams(organization.id)
    } catch (e) { notify("error", "Failed to create team") }
  }

  const inviteMember = async () => {
    if (!inviteEmail.trim() || !organization) return
    try {
      const res = await fetch(API_BASE + "/api/organizations/" + organization.id + "/members", {
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
      fetchMembers(organization.id)
    } catch (e) { notify("error", "Failed to send invitation") }
  }

  const updateOrganization = async () => {
    if (!editOrgName.trim() || !organization) return
    try {
      await fetch(API_BASE + "/api/organizations/" + organization.id, {
        method: "PUT",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: editOrgName })
      })
      notify("success", "Organization updated")
      setShowSettingsDialog(false)
      fetchOrganization()
    } catch (e) { notify("error", "Failed to update") }
  }

  const deleteOrganization = async () => {
    if (!organization) return
    try {
      await fetch(API_BASE + "/api/organizations/" + organization.id, {
        method: "DELETE",
        credentials: "include"
      })
      notify("success", "Organization deleted")
      setShowDeleteDialog(false)
      setOrganization(null)
      setTeams([])
      setMembers([])
    } catch (e) { notify("error", "Failed to delete") }
  }

  const updateMemberRole = async (memberId: string, newRole: string) => {
    if (!organization) return
    try {
      await fetch(API_BASE + "/api/organizations/" + organization.id + "/members/" + memberId, {
        method: "PUT",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ role: newRole })
      })
      notify("success", "Role updated")
      fetchMembers(organization.id)
    } catch (e) { notify("error", "Failed to update role") }
  }

  const removeMember = async (memberId: string) => {
    if (!organization) return
    try {
      await fetch(API_BASE + "/api/organizations/" + organization.id + "/members/" + memberId, {
        method: "DELETE",
        credentials: "include"
      })
      notify("success", "Member removed")
      fetchMembers(organization.id)
    } catch (e) { notify("error", "Failed to remove member") }
  }

  const deleteTeam = async (teamId: string) => {
    if (!organization) return
    try {
      await fetch(API_BASE + "/api/organizations/" + organization.id + "/teams/" + teamId, {
        method: "DELETE",
        credentials: "include"
      })
      notify("success", "Team deleted")
      fetchTeams(organization.id)
    } catch (e) { notify("error", "Failed to delete team") }
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

  // No organization yet - show create screen
  if (!organization) {
    return (
      <Sidebar>
        <div className="flex-1 overflow-auto">
          <div className="border-b sticky top-0 z-10 bg-background">
            <div className="px-6 py-4">
              <h1 className="text-xl font-semibold">Organization</h1>
              <p className="text-sm text-muted-foreground">Create your organization to collaborate with your team</p>
            </div>
          </div>
          <div className="p-6">
            <Card className="max-w-lg mx-auto mt-12">
              <CardContent className="pt-12 pb-12 text-center">
                <Building2 className="h-16 w-16 mx-auto text-muted-foreground mb-4" />
                <h2 className="text-xl font-semibold mb-2">Create Your Organization</h2>
                <p className="text-sm text-muted-foreground mb-6">Set up your organization to invite team members and create teams for better collaboration.</p>
                <Button size="lg" onClick={() => setShowCreateOrgDialog(true)}>
                  <Building2 className="h-4 w-4 mr-2" />Create Organization
                </Button>
              </CardContent>
            </Card>
          </div>

          <Dialog open={showCreateOrgDialog} onOpenChange={setShowCreateOrgDialog}>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Create Organization</DialogTitle>
                <DialogDescription>Enter your organization or company name.</DialogDescription>
              </DialogHeader>
              <div className="space-y-4 py-4">
                <div>
                  <Label>Organization Name</Label>
                  <Input value={newOrgName} onChange={(e) => setNewOrgName(e.target.value)} placeholder="Acme Inc" className="mt-2" />
                </div>
              </div>
              <DialogFooter>
                <Button variant="outline" onClick={() => setShowCreateOrgDialog(false)}>Cancel</Button>
                <Button onClick={createOrganization}>Create Organization</Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>
      </Sidebar>
    )
  }

  // Has organization - show management screen
  return (
    <Sidebar>
      <div className="flex-1 overflow-auto">
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
              <div className="flex items-center gap-4">
                <div className="h-12 w-12 rounded-xl bg-muted flex items-center justify-center font-bold text-lg">
                  {organization.name.slice(0, 2).toUpperCase()}
                </div>
                <div>
                  <h1 className="text-xl font-semibold">{organization.name}</h1>
                  <p className="text-sm text-muted-foreground">{members.length} members · {teams.length} teams</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                {organization.is_owner && (
                  <Button variant="outline" size="sm" onClick={() => setShowSettingsDialog(true)}>
                    <Settings className="h-4 w-4 mr-2" />Settings
                  </Button>
                )}
              </div>
            </div>

            {/* Tabs */}
            <div className="flex gap-1 mt-4 border-b -mb-4">
              <button
                onClick={() => setActiveTab("teams")}
                className={`px-4 py-2 text-sm font-medium border-b-2 -mb-[1px] transition-colors ${activeTab === "teams" ? "border-primary text-foreground" : "border-transparent text-muted-foreground hover:text-foreground"}`}
              >
                Teams
              </button>
              <button
                onClick={() => setActiveTab("members")}
                className={`px-4 py-2 text-sm font-medium border-b-2 -mb-[1px] transition-colors ${activeTab === "members" ? "border-primary text-foreground" : "border-transparent text-muted-foreground hover:text-foreground"}`}
              >
                Members
              </button>
            </div>
          </div>
        </div>

        <div className="p-6">
          {/* Teams Tab */}
          {activeTab === "teams" && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-lg font-semibold">Teams</h2>
                  <p className="text-sm text-muted-foreground">{teams.length} / {maxTeamsPerOrg} teams</p>
                </div>
                {(organization.is_owner || organization.role === "admin") && teams.length < maxTeamsPerOrg && (
                  <Button onClick={() => setShowCreateTeamDialog(true)}>
                    <Plus className="h-4 w-4 mr-2" />New Team
                  </Button>
                )}
              </div>

              {teams.length === 0 ? (
                <Card>
                  <CardContent className="pt-12 pb-12 text-center">
                    <Layers className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                    <h3 className="text-lg font-medium mb-2">No teams yet</h3>
                    <p className="text-sm text-muted-foreground mb-4">Create teams to organize your members into groups.</p>
                    <Button onClick={() => setShowCreateTeamDialog(true)}>
                      <Plus className="h-4 w-4 mr-2" />Create First Team
                    </Button>
                  </CardContent>
                </Card>
              ) : (
                <div className="grid grid-cols-3 gap-4">
                  {teams.map((team) => (
                    <Card key={team.id} className="hover:shadow-md transition-shadow">
                      <CardContent className="pt-6">
                        <div className="flex items-start justify-between">
                          <div className="flex items-center gap-3">
                            <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center">
                              <Layers className="h-5 w-5 text-primary" />
                            </div>
                            <div>
                              <p className="font-semibold">{team.name}</p>
                              <p className="text-sm text-muted-foreground">{team.member_count || 0} members</p>
                            </div>
                          </div>
                          {organization.is_owner && (
                            <Button size="sm" variant="ghost" onClick={() => deleteTeam(team.id)}>
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          )}
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                  {teams.length < maxTeamsPerOrg && (
                    <Card className="border-dashed hover:bg-muted/50 cursor-pointer transition-colors" onClick={() => setShowCreateTeamDialog(true)}>
                      <CardContent className="pt-6 flex items-center justify-center h-full min-h-[100px]">
                        <div className="text-center text-muted-foreground">
                          <Plus className="h-8 w-8 mx-auto mb-2" />
                          <p className="text-sm">Add Team</p>
                        </div>
                      </CardContent>
                    </Card>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Members Tab */}
          {activeTab === "members" && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-lg font-semibold">Members</h2>
                  <p className="text-sm text-muted-foreground">{members.length} / {organization.max_members} members</p>
                </div>
                {(organization.is_owner || organization.role === "admin") && members.length < organization.max_members && (
                  <Button onClick={() => setShowInviteDialog(true)}>
                    <UserPlus className="h-4 w-4 mr-2" />Invite Member
                  </Button>
                )}
              </div>

              <Card>
                <CardContent className="p-0">
                  <div className="divide-y">
                    {members.map((member) => (
                      <div key={member.id} className="flex items-center justify-between p-4 hover:bg-muted/50">
                        <div className="flex items-center gap-3">
                          <Avatar className="h-10 w-10">
                            <AvatarFallback>{getInitials(member.name, member.email)}</AvatarFallback>
                          </Avatar>
                          <div>
                            <div className="flex items-center gap-2">
                              <p className="font-medium">{member.name || member.email.split("@")[0]}</p>
                              {member.status === "pending" && (
                                <Badge variant="secondary" className="text-xs">
                                  <Clock className="h-3 w-3 mr-1" />Pending
                                </Badge>
                              )}
                            </div>
                            <p className="text-sm text-muted-foreground">{member.email}</p>
                          </div>
                        </div>
                        <div className="flex items-center gap-3">
                          <Badge variant={member.role === "owner" ? "default" : "outline"} className="gap-1">
                            {getRoleIcon(member.role)}
                            {member.role}
                          </Badge>
                          {organization.is_owner && member.role !== "owner" && (
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
            </div>
          )}
        </div>

        {/* Create Team Dialog */}
        <Dialog open={showCreateTeamDialog} onOpenChange={setShowCreateTeamDialog}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Create Team</DialogTitle>
              <DialogDescription>Create a new team within {organization.name}.</DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div>
                <Label>Team Name</Label>
                <Input value={newTeamName} onChange={(e) => setNewTeamName(e.target.value)} placeholder="Engineering, Marketing, Sales..." className="mt-2" />
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setShowCreateTeamDialog(false)}>Cancel</Button>
              <Button onClick={createTeam}>Create Team</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Invite Dialog */}
        <Dialog open={showInviteDialog} onOpenChange={setShowInviteDialog}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Invite Member</DialogTitle>
              <DialogDescription>Invite someone to join {organization.name}.</DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div>
                <Label>Email Address</Label>
                <Input type="email" value={inviteEmail} onChange={(e) => setInviteEmail(e.target.value)} placeholder="colleague@company.com" className="mt-2" />
              </div>
              <div>
                <Label>Role</Label>
                <Select value={inviteRole} onValueChange={setInviteRole}>
                  <SelectTrigger className="mt-2">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="admin">Admin - Can manage teams and members</SelectItem>
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
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div>
                <Label>Organization Name</Label>
                <Input value={editOrgName} onChange={(e) => setEditOrgName(e.target.value)} className="mt-2" />
              </div>
              <div className="pt-4 border-t">
                <p className="text-sm font-medium text-destructive">Danger Zone</p>
                <p className="text-sm text-muted-foreground mt-1">This will delete all teams and remove all members.</p>
                <Button variant="destructive" className="mt-3" onClick={() => { setShowSettingsDialog(false); setShowDeleteDialog(true); }}>
                  <Trash2 className="h-4 w-4 mr-2" />Delete Organization
                </Button>
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setShowSettingsDialog(false)}>Cancel</Button>
              <Button onClick={updateOrganization}>Save</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Delete Dialog */}
        <Dialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Delete Organization</DialogTitle>
              <DialogDescription>Are you sure? This will delete all teams and remove all members. This cannot be undone.</DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <Button variant="outline" onClick={() => setShowDeleteDialog(false)}>Cancel</Button>
              <Button variant="destructive" onClick={deleteOrganization}>Delete Organization</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    </Sidebar>
  )
}

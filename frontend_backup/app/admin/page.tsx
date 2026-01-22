"use client"

import { useState, useEffect } from "react"
import { Sidebar } from "@/components/sidebar"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Progress } from "@/components/ui/progress"
import { Users, Database, Key, FileUp, Brain, Trash2, Shield, RefreshCw, Save, ChevronLeft, ChevronRight, Check, X, Edit, HardDrive, ShieldX, Zap, Globe, Server, Mail } from "lucide-react"

const API_BASE = "/api"

export default function AdminPage() {
  const [isAdmin, setIsAdmin] = useState(false)
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState("overview")
  const [notification, setNotification] = useState<{type: string, message: string} | null>(null)
  const [searchQuery, setSearchQuery] = useState("")
  
  const [users, setUsers] = useState<any[]>([])
  const [models, setModels] = useState<any[]>([])
  const [apiKeys, setApiKeys] = useState<any[]>([])
  const [files, setFiles] = useState<any[]>([])
  
  const [userPage, setUserPage] = useState(1)
  const [modelPage, setModelPage] = useState(1)
  const [filePage, setFilePage] = useState(1)
  const pageSize = 10
  
  const [stats, setStats] = useState({ totalUsers: 0, totalModels: 0, totalKeys: 0, totalFiles: 0, totalStorage: 0, totalRequests: 0 })

  const [config, setConfig] = useState({
    modelPath: "./checkpoints/model_12sector.pt",
    smtpEmail: "hello@schemalabs.ai",
    smtpHost: "smtp.gmail.com",
    smtpPort: 587,
    maxFileSize: 50,
    maxStorage: 1024,
    maxModelsPerUser: 100,
    maxKeysPerUser: 10,
    maintenanceMode: false,
    allowSignups: true,
    requireEmailVerification: true
  })

  const [plans, setPlans] = useState([
    { name: "Free", price: "$0", description: "For individuals", features: ["1,000 API calls/month", "5 GB storage", "3 models", "Community support"] },
    { name: "Pro", price: "$49", description: "For teams", features: ["50,000 API calls/month", "50 GB storage", "Unlimited models", "Priority support", "Advanced analytics"] },
    { name: "Enterprise", price: "Custom", description: "For organizations", features: ["Unlimited API calls", "Unlimited storage", "24/7 support", "Custom SLA", "SSO & SAML"] }
  ])

  const [editingPlan, setEditingPlan] = useState<number | null>(null)
  const [newFeature, setNewFeature] = useState("")

  const tabs = [
    { id: "overview", label: "Overview" },
    { id: "users", label: "Users" },
    { id: "plans", label: "Plans" },
    { id: "models", label: "Models" },
    { id: "storage", label: "Storage" },
    { id: "settings", label: "Settings" },
  ]

  useEffect(() => { checkAdmin() }, [])
  useEffect(() => { if (notification) { const t = setTimeout(() => setNotification(null), 3000); return () => clearTimeout(t) } }, [notification])

  const notify = (type: string, message: string) => setNotification({ type, message })

  const checkAdmin = async () => {
    try {
      const res = await fetch(API_BASE + "/auth/me", { credentials: "include" })
      const data = await res.json()
      if (data && data.role === "admin") { setIsAdmin(true); fetchAllData() }
      else { setIsAdmin(false); setLoading(false) }
    } catch { setIsAdmin(false); setLoading(false) }
  }

  const fetchAllData = async () => {
    try {
      const [usersRes, modelsRes, keysRes, filesRes, configRes] = await Promise.all([
        fetch(API_BASE + "/admin/users", { credentials: "include" }),
        fetch(API_BASE + "/admin/models", { credentials: "include" }),
        fetch(API_BASE + "/admin/keys", { credentials: "include" }),
        fetch(API_BASE + "/admin/files", { credentials: "include" }),
        fetch(API_BASE + "/admin/config", { credentials: "include" })
      ])
      const [usersData, modelsData, keysData, filesData, configData] = await Promise.all([usersRes.json(), modelsRes.json(), keysRes.json(), filesRes.json(), configRes.json()])

      setUsers(Array.isArray(usersData) ? usersData : [])
      setModels(Array.isArray(modelsData) ? modelsData : [])
      setApiKeys(Array.isArray(keysData) ? keysData : [])
      setFiles(Array.isArray(filesData) ? filesData : [])
      if (configData && !configData.error) setConfig(prev => ({ ...prev, ...configData }))

      const totalStorage = Array.isArray(filesData) ? filesData.reduce((acc, f) => acc + (f.size || 0), 0) / (1024 * 1024) : 0
      const totalRequests = Array.isArray(keysData) ? keysData.reduce((acc, k) => acc + (k.requests || 0), 0) : 0
      setStats({
        totalUsers: Array.isArray(usersData) ? usersData.length : 0,
        totalModels: Array.isArray(modelsData) ? modelsData.length : 0,
        totalKeys: Array.isArray(keysData) ? keysData.length : 0,
        totalFiles: Array.isArray(filesData) ? filesData.length : 0,
        totalStorage: parseFloat(totalStorage.toFixed(2)),
        totalRequests
      })
    } catch (e) { console.error(e) }
    setLoading(false)
  }

  const toggleAdmin = async (id: string, role: string) => {
    const newRole = role === "admin" ? "user" : "admin"
    try {
      await fetch(API_BASE + "/admin/users/" + id + "/role", { method: "PUT", credentials: "include", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ role: newRole }) })
      notify("success", newRole === "admin" ? "User promoted to admin" : "Admin rights revoked")
      fetchAllData()
    } catch { notify("error", "Operation failed") }
  }

  const deleteUser = async (id: string, email: string) => { try { await fetch(API_BASE + "/admin/users/" + id, { method: "DELETE", credentials: "include" }); notify("success", email + " deleted"); fetchAllData() } catch { notify("error", "Delete failed") } }
  const deleteModel = async (id: string) => { try { await fetch(API_BASE + "/admin/models/" + id, { method: "DELETE", credentials: "include" }); notify("success", "Model deleted"); fetchAllData() } catch { notify("error", "Delete failed") } }
  const deleteFile = async (id: string) => { try { await fetch(API_BASE + "/admin/files/" + id, { method: "DELETE", credentials: "include" }); notify("success", "File deleted"); fetchAllData() } catch { notify("error", "Delete failed") } }
  const deleteKey = async (id: string) => { try { await fetch(API_BASE + "/admin/keys/" + id, { method: "DELETE", credentials: "include" }); notify("success", "API key deleted"); fetchAllData() } catch { notify("error", "Delete failed") } }
  const saveConfig = async () => { try { await fetch(API_BASE + "/admin/config", { method: "PUT", credentials: "include", headers: { "Content-Type": "application/json" }, body: JSON.stringify(config) }); notify("success", "Settings saved") } catch { notify("error", "Save failed") } }
  const savePlans = async () => { try { await fetch(API_BASE + "/admin/plans", { method: "PUT", credentials: "include", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ plans }) }); notify("success", "Plans saved") } catch { notify("error", "Save failed") } }

  const addFeature = (i: number) => { if (!newFeature.trim()) return; const u = [...plans]; u[i].features.push(newFeature.trim()); setPlans(u); setNewFeature("") }
  const removeFeature = (pi: number, fi: number) => { const u = [...plans]; u[pi].features.splice(fi, 1); setPlans(u) }
  const updatePlan = (i: number, field: string, value: string) => { const u = [...plans] as any; u[i][field] = value; setPlans(u) }

  const paginate = (items: any[], page: number) => items.slice((page - 1) * pageSize, page * pageSize)
  const totalPages = (items: any[]) => Math.ceil(items.length / pageSize)
  const filteredUsers = users.filter(u => u.email?.toLowerCase().includes(searchQuery.toLowerCase()) || u.name?.toLowerCase().includes(searchQuery.toLowerCase()))
  const getInitials = (name: string, email: string) => name ? name.split(" ").map(n => n[0]).join("").toUpperCase().slice(0, 2) : email?.slice(0, 2).toUpperCase() || "U"

  if (loading) return <Sidebar><div className="flex-1 flex items-center justify-center"><p className="text-muted-foreground">Loading...</p></div></Sidebar>
  if (!isAdmin) return <Sidebar><div className="flex-1 flex items-center justify-center"><div className="text-center"><ShieldX className="h-12 w-12 mx-auto text-muted-foreground mb-4" /><h1 className="text-xl font-semibold mb-2">Access Denied</h1><p className="text-sm text-muted-foreground">Admin privileges required.</p></div></div></Sidebar>

  return (
    <Sidebar>
      <div className="flex-1 overflow-auto">
        {notification && (
          <div className={`fixed top-4 right-4 z-50 px-4 py-2.5 rounded-lg shadow-lg flex items-center gap-2 text-sm font-medium ${notification.type === "success" ? "bg-primary text-primary-foreground" : "bg-destructive text-destructive-foreground"}`}>
            {notification.type === "success" ? <Check className="h-4 w-4" /> : <X className="h-4 w-4" />}
            {notification.message}
          </div>
        )}

        <div className="border-b sticky top-0 z-10 bg-background">
          <div className="px-6 py-4">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h1 className="text-xl font-semibold">Admin Panel</h1>
                <p className="text-sm text-muted-foreground">Manage your platform</p>
              </div>
              <Button onClick={fetchAllData} variant="outline" size="sm">
                <RefreshCw className="h-4 w-4 mr-2" />Refresh
              </Button>
            </div>
            <div className="flex gap-1 border-b -mb-4 pb-0">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors -mb-[2px] ${
                    activeTab === tab.id
                      ? "border-primary text-foreground"
                      : "border-transparent text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="p-6">
          
          {activeTab === "overview" && (
            <div className="space-y-6">
              <div className="grid grid-cols-4 gap-4">
                <Card><CardContent className="pt-6"><p className="text-sm text-muted-foreground">Users</p><p className="text-3xl font-bold">{stats.totalUsers}</p></CardContent></Card>
                <Card><CardContent className="pt-6"><p className="text-sm text-muted-foreground">API Requests</p><p className="text-3xl font-bold">{stats.totalRequests.toLocaleString()}</p></CardContent></Card>
                <Card><CardContent className="pt-6"><p className="text-sm text-muted-foreground">Models</p><p className="text-3xl font-bold">{stats.totalModels}</p></CardContent></Card>
                <Card><CardContent className="pt-6"><p className="text-sm text-muted-foreground">Storage</p><p className="text-3xl font-bold">{stats.totalStorage.toFixed(1)} <span className="text-base font-normal text-muted-foreground">MB</span></p></CardContent></Card>
              </div>

              <div className="grid grid-cols-3 gap-6">
                <Card className="col-span-2">
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div><CardTitle>Recent Users</CardTitle><CardDescription>Latest registered accounts</CardDescription></div>
                      <Button variant="ghost" size="sm" onClick={() => setActiveTab("users")}>View All</Button>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {users.slice(0, 6).map((user) => (
                        <div key={user.id} className="flex items-center justify-between py-2 border-b last:border-0">
                          <div className="flex items-center gap-3">
                            <Avatar className="h-9 w-9"><AvatarFallback>{getInitials(user.name, user.email)}</AvatarFallback></Avatar>
                            <div>
                              <p className="font-medium text-sm">{user.name || user.email?.split("@")[0]}</p>
                              <p className="text-xs text-muted-foreground">{user.email}</p>
                            </div>
                          </div>
                          <div className="flex items-center gap-3">
                            <Badge variant={user.role === "admin" ? "default" : "secondary"}>{user.role || "user"}</Badge>
                            <span className="text-xs text-muted-foreground">{user.created_at ? new Date(user.created_at).toLocaleDateString() : ""}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                <div className="space-y-6">
                  <Card>
                    <CardHeader><CardTitle>Storage</CardTitle></CardHeader>
                    <CardContent className="space-y-4">
                      <div className="text-center py-4">
                        <p className="text-4xl font-bold">{Math.round((stats.totalStorage / config.maxStorage) * 100)}%</p>
                        <p className="text-sm text-muted-foreground">used</p>
                      </div>
                      <Progress value={(stats.totalStorage / config.maxStorage) * 100} className="h-2" />
                      <p className="text-sm text-muted-foreground text-center">{stats.totalStorage.toFixed(1)} / {config.maxStorage} MB</p>
                      <div className="grid grid-cols-2 gap-4 pt-2">
                        <div className="text-center p-3 border rounded-lg"><p className="text-xl font-bold">{stats.totalFiles}</p><p className="text-xs text-muted-foreground">Files</p></div>
                        <div className="text-center p-3 border rounded-lg"><p className="text-xl font-bold">{stats.totalKeys}</p><p className="text-xs text-muted-foreground">API Keys</p></div>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-6">
                <Card>
                  <CardHeader><CardTitle>API Keys</CardTitle><CardDescription>Request counts</CardDescription></CardHeader>
                  <CardContent>
                    {apiKeys.length === 0 ? (
                      <p className="text-sm text-muted-foreground text-center py-8">No API keys</p>
                    ) : (
                      <div className="space-y-3">
                        {apiKeys.slice(0, 5).map((k) => (
                          <div key={k.id} className="flex items-center justify-between py-2 border-b last:border-0">
                            <div>
                              <p className="text-sm font-medium">{k.name}</p>
                              <p className="text-xs text-muted-foreground font-mono">{k.key?.substring(0, 16)}...</p>
                            </div>
                            <p className="font-semibold">{k.requests || 0}</p>
                          </div>
                        ))}
                      </div>
                    )}
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader><CardTitle>Recent Files</CardTitle><CardDescription>Latest uploads</CardDescription></CardHeader>
                  <CardContent>
                    {files.length === 0 ? (
                      <p className="text-sm text-muted-foreground text-center py-8">No files</p>
                    ) : (
                      <div className="space-y-3">
                        {files.slice(0, 5).map((f) => (
                          <div key={f.id} className="flex items-center justify-between py-2 border-b last:border-0">
                            <div className="min-w-0">
                              <p className="text-sm font-medium truncate">{f.filename}</p>
                              <p className="text-xs text-muted-foreground">{f.user_email}</p>
                            </div>
                            <span className="text-sm text-muted-foreground shrink-0">{((f.size || 0) / 1024).toFixed(1)} KB</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            </div>
          )}

          {activeTab === "users" && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <div><h2 className="text-lg font-semibold">User Management</h2><p className="text-sm text-muted-foreground">{users.length} total users</p></div>
                <Input placeholder="Search users..." className="w-64" value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} />
              </div>

              <div className="grid grid-cols-4 gap-4">
                <Card><CardContent className="pt-4 pb-4"><p className="text-2xl font-bold">{stats.totalUsers}</p><p className="text-xs text-muted-foreground">Total Users</p></CardContent></Card>
                <Card><CardContent className="pt-4 pb-4"><p className="text-2xl font-bold">{users.filter(u => u.role === "admin").length}</p><p className="text-xs text-muted-foreground">Admins</p></CardContent></Card>
                <Card><CardContent className="pt-4 pb-4"><p className="text-2xl font-bold">{users.filter(u => u.role !== "admin").length}</p><p className="text-xs text-muted-foreground">Regular Users</p></CardContent></Card>
                <Card><CardContent className="pt-4 pb-4"><p className="text-2xl font-bold">{stats.totalKeys}</p><p className="text-xs text-muted-foreground">API Keys</p></CardContent></Card>
              </div>

              <Card>
                <CardContent className="p-0">
                  <table className="w-full">
                    <thead><tr className="border-b"><th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">User</th><th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Role</th><th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Joined</th><th className="text-right py-3 px-4 text-sm font-medium text-muted-foreground">Actions</th></tr></thead>
                    <tbody>
                      {paginate(filteredUsers, userPage).map((user) => (
                        <tr key={user.id} className="border-b last:border-0">
                          <td className="py-3 px-4">
                            <div className="flex items-center gap-3">
                              <Avatar className="h-9 w-9"><AvatarFallback>{getInitials(user.name, user.email)}</AvatarFallback></Avatar>
                              <div><p className="font-medium">{user.name || "-"}</p><p className="text-sm text-muted-foreground">{user.email}</p></div>
                            </div>
                          </td>
                          <td className="py-3 px-4"><Badge variant={user.role === "admin" ? "default" : "secondary"}>{user.role || "user"}</Badge></td>
                          <td className="py-3 px-4 text-sm text-muted-foreground">{user.created_at ? new Date(user.created_at).toLocaleDateString() : "-"}</td>
                          <td className="py-3 px-4 text-right">
                            <Button size="sm" variant={user.role === "admin" ? "default" : "outline"} className="mr-2" onClick={() => toggleAdmin(user.id, user.role)}><Shield className="h-3 w-3 mr-1" />{user.role === "admin" ? "Admin" : "Make Admin"}</Button>
                            <Button size="sm" variant="ghost" onClick={() => deleteUser(user.id, user.email)}><Trash2 className="h-4 w-4" /></Button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </CardContent>
              </Card>

              {totalPages(filteredUsers) > 1 && (
                <div className="flex items-center justify-between">
                  <p className="text-sm text-muted-foreground">Page {userPage} of {totalPages(filteredUsers)}</p>
                  <div className="flex gap-1">
                    <Button size="sm" variant="outline" disabled={userPage === 1} onClick={() => setUserPage(p => p - 1)}><ChevronLeft className="h-4 w-4" /></Button>
                    <Button size="sm" variant="outline" disabled={userPage >= totalPages(filteredUsers)} onClick={() => setUserPage(p => p + 1)}><ChevronRight className="h-4 w-4" /></Button>
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === "plans" && (
            <div className="space-y-6">
              <div><h2 className="text-lg font-semibold">Subscription Plans</h2><p className="text-sm text-muted-foreground">Manage pricing and features</p></div>

              <div className="grid grid-cols-3 gap-6">
                {plans.map((plan, i) => (
                  <Card key={plan.name} className={i === 1 ? "border-primary" : ""}>
                    {i === 1 && <div className="bg-primary text-primary-foreground text-xs text-center py-1">POPULAR</div>}
                    <CardHeader className="pb-2">
                      <div className="flex items-center justify-between">
                        <CardTitle>{plan.name}</CardTitle>
                        <Button size="sm" variant="ghost" onClick={() => setEditingPlan(editingPlan === i ? null : i)}><Edit className="h-4 w-4" /></Button>
                      </div>
                      {editingPlan === i ? <Input value={plan.description} onChange={(e) => updatePlan(i, "description", e.target.value)} /> : <CardDescription>{plan.description}</CardDescription>}
                    </CardHeader>
                    <CardContent className="space-y-4">
                      {editingPlan === i ? <Input value={plan.price} onChange={(e) => updatePlan(i, "price", e.target.value)} className="text-xl font-bold" /> : <p className="text-3xl font-bold">{plan.price}<span className="text-sm font-normal text-muted-foreground">{plan.name !== "Enterprise" && "/mo"}</span></p>}
                      <div className="space-y-2 pt-4 border-t">
                        {plan.features.map((f, fi) => (
                          <div key={fi} className="flex items-center justify-between text-sm">
                            <span className="flex items-center gap-2"><Check className="h-4 w-4" />{f}</span>
                            {editingPlan === i && <Button size="sm" variant="ghost" className="h-6 w-6 p-0" onClick={() => removeFeature(i, fi)}><X className="h-3 w-3" /></Button>}
                          </div>
                        ))}
                      </div>
                      {editingPlan === i && <div className="flex gap-2"><Input placeholder="Add feature..." value={newFeature} onChange={(e) => setNewFeature(e.target.value)} /><Button size="sm" onClick={() => addFeature(i)}>Add</Button></div>}
                    </CardContent>
                  </Card>
                ))}
              </div>
              <Button onClick={savePlans}><Save className="h-4 w-4 mr-2" />Save Plans</Button>
            </div>
          )}

          {activeTab === "models" && (
            <div className="space-y-6">
              <div><h2 className="text-lg font-semibold">Fine-tuned Models</h2><p className="text-sm text-muted-foreground">{models.length} models</p></div>

              <div className="grid grid-cols-4 gap-4">
                <Card><CardContent className="pt-4 pb-4"><p className="text-2xl font-bold">{stats.totalModels}</p><p className="text-xs text-muted-foreground">Total</p></CardContent></Card>
                <Card><CardContent className="pt-4 pb-4"><p className="text-2xl font-bold">{models.reduce((a, m) => a + (m.request_count || 0), 0)}</p><p className="text-xs text-muted-foreground">Requests</p></CardContent></Card>
                <Card><CardContent className="pt-4 pb-4"><p className="text-2xl font-bold">{models.length > 0 ? (models.reduce((a, m) => a + (m.accuracy || 0), 0) / models.length).toFixed(1) : 0}%</p><p className="text-xs text-muted-foreground">Avg Accuracy</p></CardContent></Card>
                <Card><CardContent className="pt-4 pb-4"><p className="text-2xl font-bold">{new Set(models.map(m => m.user_id)).size}</p><p className="text-xs text-muted-foreground">Users</p></CardContent></Card>
              </div>

              <Card>
                <CardContent className="p-0">
                  {models.length === 0 ? (
                    <div className="text-center py-16"><Brain className="h-12 w-12 mx-auto text-muted-foreground mb-3" /><p className="text-muted-foreground">No models yet</p></div>
                  ) : (
                    <table className="w-full">
                      <thead><tr className="border-b"><th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Model</th><th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Owner</th><th className="text-right py-3 px-4 text-sm font-medium text-muted-foreground">Accuracy</th><th className="text-right py-3 px-4 text-sm font-medium text-muted-foreground">Requests</th><th className="text-right py-3 px-4 text-sm font-medium text-muted-foreground">Actions</th></tr></thead>
                      <tbody>
                        {paginate(models, modelPage).map((m) => (
                          <tr key={m.id} className="border-b last:border-0">
                            <td className="py-3 px-4 font-medium">{m.name}</td>
                            <td className="py-3 px-4 text-muted-foreground">{m.user_email}</td>
                            <td className="py-3 px-4 text-right"><Badge variant="outline">{m.accuracy?.toFixed(1)}%</Badge></td>
                            <td className="py-3 px-4 text-right">{m.request_count || 0}</td>
                            <td className="py-3 px-4 text-right"><Button size="sm" variant="ghost" onClick={() => deleteModel(m.id)}><Trash2 className="h-4 w-4" /></Button></td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </CardContent>
              </Card>

              {totalPages(models) > 1 && (
                <div className="flex items-center justify-between">
                  <p className="text-sm text-muted-foreground">Page {modelPage} of {totalPages(models)}</p>
                  <div className="flex gap-1">
                    <Button size="sm" variant="outline" disabled={modelPage === 1} onClick={() => setModelPage(p => p - 1)}><ChevronLeft className="h-4 w-4" /></Button>
                    <Button size="sm" variant="outline" disabled={modelPage >= totalPages(models)} onClick={() => setModelPage(p => p + 1)}><ChevronRight className="h-4 w-4" /></Button>
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === "storage" && (
            <div className="space-y-6">
              <div><h2 className="text-lg font-semibold">Storage Management</h2><p className="text-sm text-muted-foreground">Files and API keys</p></div>

              <div className="grid grid-cols-4 gap-4">
                <Card><CardContent className="pt-4 pb-4"><p className="text-2xl font-bold">{stats.totalFiles}</p><p className="text-xs text-muted-foreground">Files</p></CardContent></Card>
                <Card><CardContent className="pt-4 pb-4"><p className="text-2xl font-bold">{stats.totalStorage.toFixed(2)}</p><p className="text-xs text-muted-foreground">MB Used</p></CardContent></Card>
                <Card><CardContent className="pt-4 pb-4"><p className="text-2xl font-bold">{stats.totalKeys}</p><p className="text-xs text-muted-foreground">API Keys</p></CardContent></Card>
                <Card><CardContent className="pt-4 pb-4"><p className="text-2xl font-bold">{(config.maxStorage - stats.totalStorage).toFixed(0)}</p><p className="text-xs text-muted-foreground">MB Available</p></CardContent></Card>
              </div>

              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-center justify-between mb-3">
                    <span className="font-medium">Storage Usage</span>
                    <span className="text-muted-foreground">{stats.totalStorage.toFixed(2)} / {config.maxStorage} MB</span>
                  </div>
                  <Progress value={(stats.totalStorage / config.maxStorage) * 100} className="h-2" />
                </CardContent>
              </Card>

              <div className="grid grid-cols-2 gap-6">
                <Card>
                  <CardHeader><CardTitle>Files ({files.length})</CardTitle></CardHeader>
                  <CardContent>
                    {files.length === 0 ? (
                      <p className="text-sm text-muted-foreground text-center py-8">No files</p>
                    ) : (
                      <div className="space-y-2">
                        {paginate(files, filePage).map((f) => (
                          <div key={f.id} className="flex items-center justify-between py-2 border-b last:border-0">
                            <div className="min-w-0"><p className="text-sm font-medium truncate">{f.filename}</p><p className="text-xs text-muted-foreground">{f.user_email}</p></div>
                            <div className="flex items-center gap-2 shrink-0">
                              <span className="text-sm text-muted-foreground">{((f.size || 0) / 1024).toFixed(1)} KB</span>
                              <Button size="sm" variant="ghost" onClick={() => deleteFile(f.id)}><Trash2 className="h-4 w-4" /></Button>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                    {totalPages(files) > 1 && (
                      <div className="flex justify-center gap-1 mt-4 pt-4 border-t">
                        <Button size="sm" variant="outline" disabled={filePage === 1} onClick={() => setFilePage(p => p - 1)}><ChevronLeft className="h-4 w-4" /></Button>
                        <Button size="sm" variant="outline" disabled={filePage >= totalPages(files)} onClick={() => setFilePage(p => p + 1)}><ChevronRight className="h-4 w-4" /></Button>
                      </div>
                    )}
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader><CardTitle>API Keys ({apiKeys.length})</CardTitle></CardHeader>
                  <CardContent>
                    {apiKeys.length === 0 ? (
                      <p className="text-sm text-muted-foreground text-center py-8">No API keys</p>
                    ) : (
                      <div className="space-y-2">
                        {apiKeys.map((k) => (
                          <div key={k.id} className="flex items-center justify-between py-2 border-b last:border-0">
                            <div className="min-w-0"><p className="text-sm font-medium">{k.name}</p><p className="text-xs text-muted-foreground font-mono">{k.key?.substring(0, 20)}...</p></div>
                            <div className="flex items-center gap-2 shrink-0">
                              <Badge variant="outline">{k.requests || 0}</Badge>
                              <Button size="sm" variant="ghost" onClick={() => deleteKey(k.id)}><Trash2 className="h-4 w-4" /></Button>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            </div>
          )}

          {activeTab === "settings" && (
            <div className="space-y-6">
              <div><h2 className="text-lg font-semibold">System Settings</h2><p className="text-sm text-muted-foreground">Configure your platform</p></div>

              <div className="grid grid-cols-2 gap-6">
                <Card>
                  <CardHeader><CardTitle>Model Configuration</CardTitle><CardDescription>Base model settings</CardDescription></CardHeader>
                  <CardContent className="space-y-4">
                    <div><Label>Base Model Path</Label><Input value={config.modelPath} onChange={(e) => setConfig({...config, modelPath: e.target.value})} className="mt-2 font-mono text-sm" /></div>
                    <div><Label>Max Models Per User</Label><Input type="number" value={config.maxModelsPerUser} onChange={(e) => setConfig({...config, maxModelsPerUser: parseInt(e.target.value)})} className="mt-2" /></div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader><CardTitle>Email Configuration</CardTitle><CardDescription>SMTP settings</CardDescription></CardHeader>
                  <CardContent className="space-y-4">
                    <div><Label>SMTP Email</Label><Input value={config.smtpEmail} onChange={(e) => setConfig({...config, smtpEmail: e.target.value})} className="mt-2" /></div>
                    <div className="grid grid-cols-2 gap-4">
                      <div><Label>Host</Label><Input value={config.smtpHost} onChange={(e) => setConfig({...config, smtpHost: e.target.value})} className="mt-2" /></div>
                      <div><Label>Port</Label><Input type="number" value={config.smtpPort} onChange={(e) => setConfig({...config, smtpPort: parseInt(e.target.value)})} className="mt-2" /></div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader><CardTitle>Storage Limits</CardTitle><CardDescription>File restrictions</CardDescription></CardHeader>
                  <CardContent className="space-y-4">
                    <div><Label>Max File Size (MB)</Label><Input type="number" value={config.maxFileSize} onChange={(e) => setConfig({...config, maxFileSize: parseInt(e.target.value)})} className="mt-2" /></div>
                    <div><Label>Max Total Storage (MB)</Label><Input type="number" value={config.maxStorage} onChange={(e) => setConfig({...config, maxStorage: parseInt(e.target.value)})} className="mt-2" /></div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader><CardTitle>API Limits</CardTitle><CardDescription>User restrictions</CardDescription></CardHeader>
                  <CardContent className="space-y-4">
                    <div><Label>Max API Keys Per User</Label><Input type="number" value={config.maxKeysPerUser} onChange={(e) => setConfig({...config, maxKeysPerUser: parseInt(e.target.value)})} className="mt-2" /></div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader><CardTitle>General Settings</CardTitle><CardDescription>Platform settings</CardDescription></CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex items-center justify-between py-2">
                      <div><p className="font-medium text-sm">Allow Signups</p><p className="text-xs text-muted-foreground">Enable registration</p></div>
                      <Button variant={config.allowSignups ? "default" : "outline"} size="sm" onClick={() => setConfig({...config, allowSignups: !config.allowSignups})}>{config.allowSignups ? "Enabled" : "Disabled"}</Button>
                    </div>
                    <div className="flex items-center justify-between py-2">
                      <div><p className="font-medium text-sm">Email Verification</p><p className="text-xs text-muted-foreground">Require verification</p></div>
                      <Button variant={config.requireEmailVerification ? "default" : "outline"} size="sm" onClick={() => setConfig({...config, requireEmailVerification: !config.requireEmailVerification})}>{config.requireEmailVerification ? "Required" : "Optional"}</Button>
                    </div>
                    <div className="flex items-center justify-between py-2">
                      <div><p className="font-medium text-sm">Maintenance Mode</p><p className="text-xs text-muted-foreground">Disable access</p></div>
                      <Button variant={config.maintenanceMode ? "default" : "outline"} size="sm" onClick={() => setConfig({...config, maintenanceMode: !config.maintenanceMode})}>{config.maintenanceMode ? "Active" : "Inactive"}</Button>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader><CardTitle>Actions</CardTitle><CardDescription>Save changes</CardDescription></CardHeader>
                  <CardContent className="space-y-2">
                    <Button onClick={saveConfig} className="w-full"><Save className="h-4 w-4 mr-2" />Save All Settings</Button>
                    <Button variant="outline" className="w-full" onClick={fetchAllData}><RefreshCw className="h-4 w-4 mr-2" />Refresh Data</Button>
                  </CardContent>
                </Card>
              </div>
            </div>
          )}

        </div>
      </div>
    </Sidebar>
  )
}

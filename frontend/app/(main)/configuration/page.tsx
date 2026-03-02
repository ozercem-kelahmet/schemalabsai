"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Checkbox } from "@/components/ui/checkbox"
import { Settings, Key, Globe, Plus, Copy, Eye, EyeOff, Trash2, CheckCircle2, Play, Brain, Check, AlertTriangle, Info, Layers } from "lucide-react"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

interface APIKey {
  id: string
  name: string
  key: string
  created_at: string
  last_used: string | null
  requests: number
  permissions: string[]
  rate_limit: string
  finetuned_model: string
  llm_provider: string
  llm_model: string
}

interface Endpoint {
  id: string
  name: string
  path: string
  fine_tuned_model_id: string
  fine_tuned_model_name?: string
  llm_model: string
  description: string
  calls: number
  status: string
  created_at: string
  vertical_config_id?: string
  vertical_config_name?: string
}

interface VConfig {
  id: string
  name: string
  model_id: string
  enabled: boolean
}

interface FineTunedModel {
  id: string
  name: string
}

const llmModels = [
  { id: "claude-3-5-sonnet-20241022", name: "Claude 3.5 Sonnet", provider: "claude" },
  { id: "claude-3-opus-20240229", name: "Claude 3 Opus", provider: "claude" },
  { id: "claude-3-5-haiku-20241022", name: "Claude 3.5 Haiku", provider: "claude" },
  { id: "gpt-4o", name: "GPT-4o", provider: "openai" },
  { id: "gpt-4o-mini", name: "GPT-4o Mini", provider: "openai" },
  { id: "gemini-1.5-pro", name: "Gemini 1.5 Pro", provider: "google" },
]

const rateLimitOptions = [
  { value: "1000/min", label: "1,000 requests/min (Alpha)" },
]

export default function ConfigurationPage() {
  const [apiKeys, setApiKeys] = useState<APIKey[]>([])
  const [apiKeyPage, setApiKeyPage] = useState(1)
  const apiKeysPerPage = 5
  const [openTooltip, setOpenTooltip] = useState<string | null>(null)
  const [copiedApi, setCopiedApi] = useState<string | null>(null)
  const [endpointPage, setEndpointPage] = useState(1)
  const endpointsPerPage = 5
  const [endpoints, setEndpoints] = useState<Endpoint[]>([])
  const [fineTunedModels, setFineTunedModels] = useState<FineTunedModel[]>([])
  const [loading, setLoading] = useState(true)
  const [showKeys, setShowKeys] = useState<Record<string, boolean>>({})
  const [copiedKey, setCopiedKey] = useState<string | null>(null)

  const [createKeyModalOpen, setCreateKeyModalOpen] = useState(false)
  const [newKeyName, setNewKeyName] = useState("")
  const [newKeyRateLimit, setNewKeyRateLimit] = useState("1000/min")
  const [selectedFineTunedModel, setSelectedFineTunedModel] = useState("")
  const [selectedLLMModel, setSelectedLLMModel] = useState("claude-3-5-sonnet-20241022")
  const [permissions, setPermissions] = useState({ read: true, write: false, query: true, delete: false })
  const [generatedKey, setGeneratedKey] = useState<string | null>(null)
  const [creatingKey, setCreatingKey] = useState(false)

  const [createEndpointModalOpen, setCreateEndpointModalOpen] = useState(false)
  const [deleteNotification, setDeleteNotification] = useState<{type: "key" | "endpoint", name: string} | null>(null)
  const [newEndpointName, setNewEndpointName] = useState("")
  const [newEndpointPath, setNewEndpointPath] = useState("")
  const [newEndpointModel, setNewEndpointModel] = useState("")
  const [newEndpointBaseModel, setNewEndpointBaseModel] = useState("schema-v0")
  const [newEndpointLLM, setNewEndpointLLM] = useState("")
  const [newEndpointDescription, setNewEndpointDescription] = useState("")
  const [newEndpointVertical, setNewEndpointVertical] = useState("")
  const [newEndpointType, setNewEndpointType] = useState("query")
  const [verticalConfigs, setVerticalConfigs] = useState<VConfig[]>([])
  const [creatingEndpoint, setCreatingEndpoint] = useState(false)
  
  const fetchVerticals = async (modelId?: string) => {
    try {
      const url = modelId ? `/api/vertical/configs?model_id=${modelId}` : `/api/vertical/configs`
      const res = await fetch(url, { credentials: "include" })
      if (res.ok) {
        const configs = await res.json() || []
        setVerticalConfigs(configs.filter((c: VConfig) => c.enabled))
      }
    } catch { setVerticalConfigs([]) }
  }

  const [testModalOpen, setTestModalOpen] = useState(false)
  const [testEndpoint, setTestEndpoint] = useState<Endpoint | null>(null)
  const [testQuery, setTestQuery] = useState("")
  const [testResponse, setTestResponse] = useState<string | null>(null)
  const [testing, setTesting] = useState(false)

  useEffect(() => { fetchData() }, [])

  const fetchData = async () => {
    try {
      const [keysRes, endpointsRes, modelsRes] = await Promise.all([
        fetch("/api/keys", { credentials: "include" }),
        fetch("/api/endpoints", { credentials: "include" }),
        fetch("/api/models/finetuned", { credentials: "include" })
      ])
      if (keysRes.ok) {
        const data = await keysRes.json()
        setApiKeys((data.keys || []).map((k: any) => ({ ...k, requests: k.requests || 0, permissions: k.permissions || ["read", "query"], rate_limit: k.rate_limit || "1000/min" })))
      }
      if (endpointsRes.ok) {
        const data = await endpointsRes.json()
        setEndpoints(data || [])
      }
      if (modelsRes.ok) {
        const data = await modelsRes.json()
        setFineTunedModels(data.models || [])
      }
    } catch (e) {
      console.error("Failed to fetch:", e)
    } finally {
      setLoading(false)
    }
  }

  const toggleKeyVisibility = (id: string) => setShowKeys(prev => ({ ...prev, [id]: !prev[id] }))

  const copyToClipboard = (text: string, id?: string) => {
    navigator.clipboard.writeText(text)
    if (id) { setCopiedKey(id); setTimeout(() => setCopiedKey(null), 2000) }
  }

  const getModelName = (id: string) => fineTunedModels.find(m => m.id === id)?.name || id?.slice(0, 8) + "..."
  const getLLMName = (id: string) => llmModels.find(m => m.id === id)?.name || id

  const createAPIKey = async () => {
    if (!newKeyName || !selectedFineTunedModel) return
    if (apiKeys.find((k: any) => k.name === newKeyName)) {
      setDeleteNotification({ type: "key", name: "Key name already exists: " + newKeyName })
      setTimeout(() => setDeleteNotification(null), 3000)
      return
    }
    setCreatingKey(true)
    try {
      const res = await fetch("/api/keys/create", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({
          name: newKeyName,
          rate_limit: "1000/min",
          permissions: Object.entries(permissions).filter(([, v]) => v).map(([k]) => k),
          finetuned_model: selectedFineTunedModel
        })
      })
      const data = await res.json()
      setGeneratedKey(data.key)
      fetchData()
    } catch (e) {
      console.error("Failed to create key:", e)
    } finally {
      setCreatingKey(false)
    }
  }

  const closeKeyModal = () => {
    setCreateKeyModalOpen(false)
    setGeneratedKey(null)
    setNewKeyName("")
    setSelectedFineTunedModel("")
    setNewKeyRateLimit("1000/min")
    setPermissions({ read: true, write: false, query: true, delete: false })
  }

  const deleteAPIKey = async (id: string) => {
    try {
      await fetch("/api/keys/delete?id=" + id, { method: "DELETE", credentials: "include" })
      setApiKeys(prev => prev.filter(k => k.id !== id))
    } catch (e) {
      console.error("Failed to delete key:", e)
    }
  }

  const createEndpoint = async () => {
    if (!newEndpointName || !newEndpointPath || (newEndpointType === "query" && !newEndpointModel)) return
    if (endpoints.find(e => e.path === newEndpointPath)) {
      setDeleteNotification({ type: "endpoint", name: "Path already exists: " + newEndpointPath })
      setTimeout(() => setDeleteNotification(null), 3000)
      return
    }
    setCreatingEndpoint(true)
    try {
      const res = await fetch("/api/endpoints/create", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({
          name: newEndpointName,
          path: newEndpointPath,
          fine_tuned_model_id: newEndpointType === "query" ? newEndpointModel : "",
          description: newEndpointDescription,
          endpoint_type: newEndpointType,
          vertical_config_id: newEndpointVertical && newEndpointVertical !== "none" ? newEndpointVertical : ""
        })
      })
      if (res.ok) {
        setCreateEndpointModalOpen(false)
        setCreatingEndpoint(false)
        setDeleteNotification({ type: "endpoint", name: "Endpoint created: " + newEndpointName })
        setTimeout(() => setDeleteNotification(null), 3000)
        setNewEndpointName("")
        setNewEndpointPath("")
        setNewEndpointModel("")
        setNewEndpointLLM("")
        setNewEndpointDescription("")
        setNewEndpointVertical(""); setNewEndpointType("query")
        fetch("/api/endpoints", { credentials: "include" }).then(r => r.ok ? r.json() : []).then(d => setEndpoints(d || []))
      }
    } catch (e) {
      console.error("Failed to create endpoint:", e)
    } finally {
      setCreatingEndpoint(false)
    }
  }

  const deleteEndpoint = async (id: string) => {
    try {
      const ep = endpoints.find(e => e.id === id)
      await fetch("/api/endpoints/delete", { method: "POST", headers: { "Content-Type": "application/json" }, credentials: "include", body: JSON.stringify({ id }) })
      setEndpoints(prev => prev.filter(e => e.id !== id))
      if (ep) {
        setDeleteNotification({ type: "endpoint", name: ep.name })
        setTimeout(() => setDeleteNotification(null), 3000)
      }
    } catch (e) {
      console.error("Failed to delete endpoint:", e)
    }
  }

  const openTestModal = (endpoint: Endpoint) => {
    setTestEndpoint(endpoint)
    setTestQuery("")
    setTestResponse(null)
    setTestModalOpen(true)
  }

  const runTest = async () => {
    if (!testEndpoint || !testQuery) return
    setTesting(true)
    setTestResponse(null)
    try {
      const keysRes = await fetch("/api/keys", { credentials: "include" })
      const keys = await keysRes.json()
      const apiKey = keys?.keys?.[0]?.key
      if (!apiKey) { setTestResponse(JSON.stringify({ error: "No API key found" }, null, 2)); setTesting(false); return }
      
      const res = await fetch(`/v1/query/${testEndpoint.path}`, {
        method: "POST",
        headers: { "Authorization": `Bearer ${apiKey}`, "Content-Type": "application/json" },
        body: JSON.stringify({ query: testQuery, data: {} })
      })
      const data = await res.json()
      setTestResponse(JSON.stringify(data, null, 2))
    } catch (e: any) {
      setTestResponse(JSON.stringify({ error: e.message }, null, 2))
    } finally {
      setTesting(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Delete Notification */}
      {deleteNotification && (
        <div className="fixed bottom-4 right-4 z-50 animate-in slide-in-from-bottom-2 fade-in duration-300">
          <div className="flex items-center gap-3 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 px-4 py-3 rounded-lg shadow-lg">
            <CheckCircle2 className="h-5 w-5" />
            <span className="text-sm font-medium">{deleteNotification.type === 'key' ? 'API Key' : 'Endpoint'} "{deleteNotification.name}" deleted successfully</span>
          </div>
        </div>
      )}
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[#0052CC]/10 dark:bg-[#0052CC]/20">
          <Settings className="h-5 w-5 text-[#0052CC] dark:text-[#2684FF]" />
        </div>
        <div>
          <h1 className="text-xl font-semibold text-foreground">Configuration</h1>
          <p className="text-sm text-muted-foreground">Manage API keys and endpoints</p>
        </div>
      </div>

      <Card className="border-border bg-card">
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2 text-foreground"><Key className="h-5 w-5" /> API Keys</CardTitle>
            <CardDescription>Manage your API keys for accessing Schema models</CardDescription>
          </div>
          <Button onClick={() => setCreateKeyModalOpen(true)} disabled={fineTunedModels.length === 0} className="gap-2 bg-[#0052CC] text-white hover:bg-[#003D99]">
            <Plus className="h-4 w-4" /> Create Key
          </Button>
        </CardHeader>
        <CardContent>
          {loading ? <div className="py-8 text-center text-muted-foreground">Loading...</div> : apiKeys.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-8 text-center">
              <Key className="h-12 w-12 text-muted-foreground" />
              <p className="mt-4 text-muted-foreground">No API keys created yet</p>
            </div>
          ) : (
            <div className="space-y-3">
              {apiKeys.slice((apiKeyPage - 1) * apiKeysPerPage, apiKeyPage * apiKeysPerPage).map((apiKey) => (
                <div key={apiKey.id} className="group flex items-center justify-between rounded-lg border border-border bg-muted/30 p-4 hover:border-border/80 transition-colors">
                  <div className="space-y-1.5 flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="font-medium text-foreground">{apiKey.name}</span>
                      <span className="flex items-center gap-1 rounded bg-emerald-500/10 px-2 py-0.5 text-xs text-emerald-500"><CheckCircle2 className="h-3 w-3" /> Active</span>
                    </div>
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <span className="font-mono text-xs truncate max-w-[200px]">{showKeys[apiKey.id] ? apiKey.key : apiKey.key.slice(0, 12) + "..." + apiKey.key.slice(-4)}</span>
                      <button onClick={() => toggleKeyVisibility(apiKey.id)} className="hover:text-foreground">{showKeys[apiKey.id] ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}</button>
                      <button onClick={() => copyToClipboard(apiKey.key, apiKey.id)} className="hover:text-foreground">{copiedKey === apiKey.id ? <Check className="h-4 w-4 text-emerald-500" /> : <Copy className="h-4 w-4" />}</button>
                    </div>
                    <div className="flex items-center gap-2 flex-wrap text-xs text-muted-foreground">
                      <span className="flex items-center gap-1"><Brain className="h-3 w-3" /> {getModelName(apiKey.finetuned_model)}</span>
                      <span>|</span>
                      <span>{apiKey.rate_limit}</span>
                    </div>
                  </div>
                  <div className="relative">
                    <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground hover:text-foreground opacity-0 group-hover:opacity-100 transition-opacity" onClick={() => setOpenTooltip(openTooltip === apiKey.id ? null : apiKey.id)}>
                      <Info className="h-4 w-4" />
                    </Button>
                    {openTooltip === apiKey.id && (
                      <>
                        <div className="fixed inset-0 z-40" onClick={() => setOpenTooltip(null)} />
                        <div className="absolute right-0 top-10 z-50 w-[520px] p-4 rounded-lg border border-border bg-card shadow-xl">
                          <div className="flex items-center justify-between mb-3">
                            <p className="text-sm font-semibold text-foreground">API Usage</p>
                            <Button variant="ghost" size="sm" className="h-7 px-2 gap-1" onClick={(e) => { e.stopPropagation(); navigator.clipboard.writeText(`curl -X POST https://api.schemalabs.ai/v1/analyze \\n  -H "Authorization: Bearer ${apiKey.key}" \\n  -F "file=@data.csv" \\n  -F "query=Analyze this data"`); setCopiedApi(apiKey.id); setTimeout(() => setCopiedApi(null), 2000); }}>
                              {copiedApi === apiKey.id ? <><Check className="h-3.5 w-3.5 text-emerald-500" /><span className="text-xs text-emerald-500">Copied!</span></> : <Copy className="h-3.5 w-3.5" />}
                            </Button>
                          </div>
                          <div className="bg-muted p-3 rounded-md font-mono text-[11px] text-muted-foreground leading-6">
                            <div>curl -X POST https://api.schemalabs.ai/v1/analyze \</div>
                            <div className="pl-4">-H "Authorization: Bearer {apiKey.key.slice(0,20)}..." \</div>
                            <div className="pl-4">-F "file=@yourdata.csv" \</div>
                            <div className="pl-4">-F "query=Analyze this data"</div>
                          </div>
                          <div className="mt-3 pt-3 border-t border-border text-xs text-muted-foreground">
                            <span className="font-medium text-foreground">Response: </span>JSON with file_info, statistics, predictions
                          </div>
                        </div>
                      </>
                    )}
                  </div>
                  <Button variant="ghost" size="icon" className="h-8 w-8 text-red-500 hover:text-red-600 hover:bg-red-500/10 opacity-0 group-hover:opacity-100 transition-opacity" onClick={() => deleteAPIKey(apiKey.id)}>
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              ))}
              {apiKeys.length > apiKeysPerPage && (
                <div className="flex items-center justify-center mt-4 pt-4 border-t border-border gap-1">
                  {Array.from({ length: Math.ceil(apiKeys.length / apiKeysPerPage) }, (_, i) => (
                    <Button key={i} variant={apiKeyPage === i + 1 ? "default" : "outline"} size="sm" className="h-8 w-8 p-0" onClick={() => setApiKeyPage(i + 1)}>
                      {i + 1}
                    </Button>
                  ))}
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      <Card className="border-border bg-card">
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2 text-foreground"><Globe className="h-5 w-5" /> Endpoints</CardTitle>
            <CardDescription>Create and manage your model API endpoints</CardDescription>
          </div>
          <Button onClick={() => setCreateEndpointModalOpen(true)} disabled={fineTunedModels.length === 0} className="gap-2 bg-[#0052CC] text-white hover:bg-[#003D99]">
            <Plus className="h-4 w-4" /> Create Endpoint
          </Button>
        </CardHeader>
        <CardContent>
          {loading ? <div className="py-8 text-center text-muted-foreground">Loading...</div> : endpoints.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-8 text-center">
              <Globe className="h-12 w-12 text-muted-foreground" />
              <p className="mt-4 text-muted-foreground">No endpoints created yet</p>
            </div>
          ) : (
            <div className="space-y-3">
              {endpoints.slice((endpointPage - 1) * endpointsPerPage, endpointPage * endpointsPerPage).map((endpoint) => (
                <div key={endpoint.id} className="group flex items-center justify-between rounded-lg border border-border bg-muted/30 p-4 hover:border-border/80 transition-colors">
                  <div className="space-y-1.5 flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="font-medium text-foreground">{endpoint.name}</span>
                      <span className="flex items-center gap-1 rounded bg-emerald-500/10 px-2 py-0.5 text-xs text-emerald-500"><CheckCircle2 className="h-3 w-3" /> Active</span>
                    </div>
                    <code className="text-xs text-[#0052CC] dark:text-[#2684FF]">https://api.schemalabs.ai/v1/{endpoint.endpoint_type === "analyze" ? "analyze" : "query"}/{endpoint.path}</code>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      <span className="flex items-center gap-1"><Brain className="h-3 w-3" /> {getModelName(endpoint.fine_tuned_model_id)}</span>
                      <span>|</span>
                      {endpoint.vertical_config_id && <><span className="flex items-center gap-1 text-purple-400"><Layers className="h-3 w-3" /> Vertical</span><span>|</span></>}
                      <span>{endpoint.calls?.toLocaleString() || 0} calls</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="relative">
                      <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground hover:text-foreground opacity-0 group-hover:opacity-100 transition-opacity" onClick={() => setOpenTooltip(openTooltip === "ep-" + endpoint.id ? null : "ep-" + endpoint.id)}>
                        <Info className="h-4 w-4" />
                      </Button>
                      {openTooltip === "ep-" + endpoint.id && (
                        <>
                          <div className="fixed inset-0 z-40" onClick={() => setOpenTooltip(null)} />
                          <div className="absolute right-0 top-10 z-50 w-[520px] p-4 rounded-lg border border-border bg-card shadow-xl">
                            <div className="flex items-center justify-between mb-3">
                              <p className="text-sm font-semibold text-foreground">Endpoint Usage</p>
                              <Button variant="ghost" size="sm" className="h-7 px-2 gap-1" onClick={(e) => { e.stopPropagation(); navigator.clipboard.writeText(`curl -X POST https://api.schemalabs.ai/v1/query/${endpoint.path} -H "Authorization: Bearer YOUR_API_KEY" -H "Content-Type: application/json" -d '{\"query\": \"Your question\"}'`); setCopiedApi("ep-" + endpoint.id); setTimeout(() => setCopiedApi(null), 2000); }}>
                                {copiedApi === "ep-" + endpoint.id ? <><Check className="h-3.5 w-3.5 text-emerald-500" /><span className="text-xs text-emerald-500">Copied!</span></> : <Copy className="h-3.5 w-3.5" />}
                              </Button>
                            </div>
                            <div className="bg-muted p-3 rounded-md font-mono text-[11px] text-muted-foreground leading-6">
                              <div>curl -X POST https://api.schemalabs.ai/v1/query/{endpoint.path} \</div>
                              <div className="pl-4">-H "Authorization: Bearer YOUR_API_KEY" \</div>
                              <div className="pl-4">-H "Content-Type: application/json" \</div>
                              <div className="pl-4">-d '\"query\": \"Your question\"'</div>
                            </div>
                            <div className="mt-3 pt-3 border-t border-border text-xs text-muted-foreground">
                              <span className="font-medium text-foreground">Response: </span>JSON with prediction and analysis
                            </div>
                          </div>
                        </>
                      )}
                    </div>
                    <Button variant="ghost" size="icon" className="h-8 w-8 text-red-500 hover:text-red-600 hover:bg-red-500/10 opacity-0 group-hover:opacity-100 transition-opacity" onClick={() => deleteEndpoint(endpoint.id)}>
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      <Dialog open={createKeyModalOpen} onOpenChange={(open) => { if (!open) closeKeyModal(); else setCreateKeyModalOpen(true) }}>
        <DialogContent className="border-border bg-card sm:max-w-[500px]">
          <DialogHeader>
            <DialogTitle>Create API Key</DialogTitle>
            <DialogDescription>Create a new API key to access your Schema models</DialogDescription>
          </DialogHeader>
          {generatedKey ? (
            <div className="space-y-4">
              <div className="p-3 bg-amber-500/10 border border-amber-500/30 rounded-lg flex items-start gap-2">
                <AlertTriangle className="h-4 w-4 text-amber-500 mt-0.5" />
                <p className="text-xs text-amber-600 dark:text-amber-400">Make sure to copy your API key now. You won't be able to see it again!</p>
              </div>
              <div className="p-3 bg-muted rounded-lg">
                <p className="text-xs text-muted-foreground mb-1">Your API Key</p>
                <div className="flex items-center gap-2">
                  <code className="flex-1 text-xs font-mono break-all">{generatedKey}</code>
                  <Button variant="outline" size="sm" onClick={() => copyToClipboard(generatedKey)}><Copy className="h-4 w-4" /></Button>
                </div>
              </div>
              <Button onClick={closeKeyModal} className="w-full">Done</Button>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="space-y-2">
                <Label>Key Name</Label>
                <Input placeholder="My API Key" value={newKeyName} onChange={(e) => setNewKeyName(e.target.value)} className="border-border bg-background" />
              </div>
              <div className="space-y-2">
                <Label>Base Model</Label>
                <Select value={selectedFineTunedModel} onValueChange={setSelectedFineTunedModel}>
                  <SelectTrigger className="border-border bg-background"><SelectValue placeholder="Select model" /></SelectTrigger>
                  <SelectContent><SelectItem value="schema-v0">schema-v0</SelectItem></SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Rate Limit</Label>
                <Select value={newKeyRateLimit} onValueChange={setNewKeyRateLimit}>
                  <SelectTrigger className="border-border bg-background"><SelectValue /></SelectTrigger>
                  <SelectContent>{rateLimitOptions.map(o => <SelectItem key={o.value} value={o.value}>{o.label}</SelectItem>)}</SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Permissions</Label>
                <div className="grid grid-cols-2 gap-2">
                  {Object.entries(permissions).map(([key, value]) => (
                    <div key={key} className="flex items-center space-x-2">
                      <Checkbox id={key} checked={value} onCheckedChange={(c) => setPermissions(prev => ({ ...prev, [key]: !!c }))} />
                      <label htmlFor={key} className="text-sm capitalize">{key}</label>
                    </div>
                  ))}
                </div>
              </div>
              <DialogFooter>
                <Button variant="outline" onClick={closeKeyModal}>Cancel</Button>
                <Button onClick={createAPIKey} disabled={creatingKey || !newKeyName || !selectedFineTunedModel} className="bg-[#0052CC] text-white hover:bg-[#003D99]">
                  {creatingKey ? "Creating..." : "Create Key"}
                </Button>
              </DialogFooter>
            </div>
          )}
        </DialogContent>
      </Dialog>

      <Dialog open={createEndpointModalOpen} onOpenChange={setCreateEndpointModalOpen}>
        <DialogContent className="border-border bg-card sm:max-w-[500px]">
          <DialogHeader>
            <DialogTitle>Create Endpoint</DialogTitle>
            <DialogDescription>Create a new API endpoint for your model</DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Endpoint Type</Label>
              <Select value={newEndpointType} onValueChange={v => { setNewEndpointType(v); if (v === "analyze") fetchVerticals(); }}>
                <SelectTrigger className="border-border bg-background"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="query">Query — use your trained model</SelectItem>
                  <SelectItem value="analyze">Analyze — accept external data files</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Base Model</Label>
              <Select value={newEndpointBaseModel} onValueChange={setNewEndpointBaseModel}>
                <SelectTrigger className="border-border bg-background"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="schema-v0">schema-v0</SelectItem>
                </SelectContent>
              </Select>
            </div>
            {newEndpointType === "query" && <div className="space-y-2">
              <Label>Fine-tuned Model</Label>
              <Select value={newEndpointModel} onValueChange={v => { setNewEndpointModel(v); fetchVerticals(v); setNewEndpointVertical("") }}>
                <SelectTrigger className="border-border bg-background"><SelectValue placeholder="Select your trained model" /></SelectTrigger>
                <SelectContent>{fineTunedModels.map(m => <SelectItem key={m.id} value={m.id}>{m.name}</SelectItem>)}</SelectContent>
              </Select>
            </div>}
            <div className="space-y-2">
              <Label>Endpoint Name</Label>
              <Input placeholder="Sales Prediction API" value={newEndpointName} onChange={(e) => setNewEndpointName(e.target.value)} className="border-border bg-background" />
            </div>
            <div className="space-y-2">
              <Label>URL Path</Label>
              <div className="flex items-center">
                <span className="rounded-l-md border border-r-0 border-border bg-muted px-3 py-2 text-sm text-muted-foreground">{newEndpointType === "analyze" ? "/v1/analyze/" : "/v1/query/"}</span>
                <Input placeholder="sales-prediction" value={newEndpointPath} onChange={(e) => setNewEndpointPath(e.target.value.toLowerCase().replace(/\s+/g, "-").replace(/[^a-z0-9-]/g, ""))} className="border-border bg-background rounded-l-none" />
              </div>
            </div>
            <div className="space-y-2">
              <Label>Vertical AI Runtime (Optional)</Label>
              <Select value={newEndpointVertical} onValueChange={setNewEndpointVertical}>
                <SelectTrigger className="border-border bg-background"><SelectValue placeholder="None - no vertical processing" /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">None</SelectItem>
                  {verticalConfigs.map(v => <SelectItem key={v.id} value={v.id}>{v.name}{v.enabled ? " (Active)" : ""}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Description (Optional)</Label>
              <Textarea placeholder="What does this endpoint do?" value={newEndpointDescription} onChange={(e) => setNewEndpointDescription(e.target.value)} className="border-border bg-background" rows={2} />
            </div>
            <div className="rounded-md bg-zinc-900 border border-zinc-700 p-3 relative">
              <button type="button" className="absolute top-2 right-2 p-1 rounded hover:bg-zinc-700 text-zinc-400 hover:text-zinc-200 transition-colors" onClick={(e) => {
                const btn = e.currentTarget;
                const p = newEndpointPath || "{path}";
                const txt = newEndpointType === "analyze"
                  ? `curl -X POST https://api.schemalabs.ai/v1/analyze/${p} \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "file=@data.csv" \
  -F "query=Analyze this data"`
                  : `curl -X POST https://api.schemalabs.ai/v1/query/${p} \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "your question"}'`;
                navigator.clipboard.writeText(txt);
                btn.innerHTML = '<span class="text-[10px] text-green-400">Copied!</span>';
                setTimeout(() => { btn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>'; }, 1500)
              }}><Copy className="h-3 w-3" /></button>
              {newEndpointType === "analyze" ? (
                <pre className="text-[11px] font-mono leading-relaxed text-zinc-300 whitespace-pre-wrap">{`curl -X POST /v1/analyze/${newEndpointPath || "{path}"}
  -H "Authorization: Bearer YOUR_API_KEY"
  -F "file=@data.csv"
  -F "query=Analyze this data"

# Response: file_info, statistics, sector, predictions`}</pre>
              ) : (
                <pre className="text-[11px] font-mono leading-relaxed text-zinc-300 whitespace-pre-wrap">{`curl -X POST /v1/query/${newEndpointPath || "{path}"}
  -H "Authorization: Bearer YOUR_API_KEY"
  -H "Content-Type: application/json"
  -d '{"query": "your question"}'

# Response: analysis, predictions, sector`}</pre>
              )}
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateEndpointModalOpen(false)}>Cancel</Button>
            <Button onClick={createEndpoint} disabled={creatingEndpoint || (newEndpointType === "query" && !newEndpointModel) || !newEndpointName || !newEndpointPath} className="bg-[#0052CC] text-white hover:bg-[#003D99]">
              {creatingEndpoint ? "Creating..." : "Create Endpoint"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={testModalOpen} onOpenChange={setTestModalOpen}>
        <DialogContent className="border-border bg-card sm:max-w-[600px]">
          <DialogHeader>
            <DialogTitle>Test Endpoint</DialogTitle>
            <DialogDescription>Test your endpoint with a custom query</DialogDescription>
          </DialogHeader>
          {testEndpoint && (
            <div className="space-y-4">
              <div className="rounded-lg border border-border bg-muted/30 p-4 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="font-medium">{testEndpoint.name}</span>
                  <span className="text-xs text-muted-foreground">{getLLMName(testEndpoint.llm_model)}</span>
                </div>
                <code className="text-xs text-[#0052CC]">POST /v1/query/{testEndpoint.path}</code>
              </div>
              <div className="space-y-2">
                <Label>Query</Label>
                <Textarea placeholder="What insights can you provide?" value={testQuery} onChange={(e) => setTestQuery(e.target.value)} className="border-border bg-background" rows={3} />
              </div>
              {testResponse && (
                <div className="space-y-2">
                  <Label>Response</Label>
                  <div className="rounded-lg border border-border bg-muted/50 p-3 max-h-[200px] overflow-auto">
                    <pre className="font-mono text-xs whitespace-pre-wrap">{testResponse}</pre>
                  </div>
                </div>
              )}
            </div>
          )}
          <DialogFooter>
            <Button variant="outline" onClick={() => setTestModalOpen(false)}>Close</Button>
            <Button onClick={runTest} disabled={!testQuery || testing} className="bg-[#0052CC] text-white hover:bg-[#003D99]">
              {testing ? "Running..." : "Run Test"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
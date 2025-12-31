"use client"

import { useState, useEffect } from "react"
import { Sidebar } from "@/components/sidebar"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Checkbox } from "@/components/ui/checkbox"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogClose,
} from "@/components/ui/dialog"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Key, Plus, Copy, Trash2, Check, AlertTriangle, Brain } from "lucide-react"
import { toast, Toaster } from "sonner"
import { API_BASE } from "@/lib/config"

interface FineTunedModel {
  id: string
  name: string
  status: string
}

interface ApiKey {
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

export default function ApiKeysPage() {
  const [existingKeys, setExistingKeys] = useState<ApiKey[]>([])
  const [fineTunedModels, setFineTunedModels] = useState<FineTunedModel[]>([])
  const [loading, setLoading] = useState(true)
  const [copiedKey, setCopiedKey] = useState<string | null>(null)
  const [newKeyName, setNewKeyName] = useState("")
  const [newKeyRateLimit, setNewKeyRateLimit] = useState("1000/min")
  const [selectedFineTunedModel, setSelectedFineTunedModel] = useState("")
  const [selectedLLMProvider, setSelectedLLMProvider] = useState("claude")
  const [selectedLLMModel, setSelectedLLMModel] = useState("claude-3-5-sonnet-20241022")
  const [permissions, setPermissions] = useState({
    read: true,
    write: false,
    query: true,
    delete: false,
  })
  const [isDialogOpen, setIsDialogOpen] = useState(false)
  const [generatedKey, setGeneratedKey] = useState<string | null>(null)
  const [generatedKeyCopied, setGeneratedKeyCopied] = useState(false)
  const [creating, setCreating] = useState(false)
  const [selectedKeyId, setSelectedKeyId] = useState<string>("")

  useEffect(() => {
    loadKeys()
    loadFineTunedModels()
  }, [])

  const loadKeys = async () => {
    try {
      const res = await fetch(API_BASE + '/keys', { credentials: 'include' })
      const data = await res.json()
      const keys = (data.keys || []).map((k: any) => ({
        ...k,
        requests: k.requests || 0,
        permissions: k.permissions || ['read', 'query'],
        rate_limit: k.rate_limit || '1000/min',
        finetuned_model: k.finetuned_model || '',
        llm_provider: k.llm_provider || 'claude',
        llm_model: k.llm_model || ''
      }))
      setExistingKeys(keys)
    } catch (error) {
      console.error("Failed to load keys:", error)
    } finally {
      setLoading(false)
    }
  }

  const loadFineTunedModels = async () => {
    try {
      const res = await fetch(API_BASE + '/models/finetuned', { credentials: 'include' })
      const data = await res.json()
      setFineTunedModels(data.models || [])
    } catch (error) {
      console.error("Failed to load models:", error)
    }
  }

  const copyKey = (id: string, key: string) => {
    navigator.clipboard.writeText(key)
    setCopiedKey(id)
    setTimeout(() => setCopiedKey(null), 2000)
    toast.success("Copied!", { description: "API key copied to clipboard." })
  }

  const copyGeneratedKey = () => {
    if (generatedKey) {
      navigator.clipboard.writeText(generatedKey)
      setGeneratedKeyCopied(true)
      setTimeout(() => setGeneratedKeyCopied(false), 2000)
      toast.success("Copied!", { description: "API key copied to clipboard." })
    }
  }

  const handleGenerateKey = async () => {
    if (creating) return
    if (!selectedFineTunedModel) {
      toast.error("Error", { description: "Please select a fine-tuned model." })
      return
    }
    setCreating(true)
    try {
      const selectedPermissions = Object.entries(permissions)
        .filter(([, value]) => value)
        .map(([key]) => key)

      const res = await fetch(API_BASE + '/keys/create', {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          name: newKeyName || "Untitled Key",
          rate_limit: newKeyRateLimit,
          permissions: selectedPermissions,
          finetuned_model: selectedFineTunedModel,
          llm_provider: selectedLLMProvider,
          llm_model: selectedLLMModel
        })
      })
      const data = await res.json()
      setGeneratedKey(data.key)
      loadKeys()
      toast.success("API Key Created", { description: "Your new API key has been created." })
    } catch (error) {
      toast.error("Error", { description: "Failed to create API key." })
    } finally {
      setCreating(false)
    }
  }

  const handleCloseDialog = () => {
    setIsDialogOpen(false)
    setGeneratedKey(null)
    setNewKeyName("")
    setNewKeyRateLimit("1000/min")
    setSelectedFineTunedModel("")
    setSelectedLLMProvider("claude")
    setSelectedLLMModel("claude-3-5-sonnet-20241022")
    setPermissions({ read: true, write: false, query: true, delete: false })
    setGeneratedKeyCopied(false)
  }

  const deleteKey = async (id: string, name: string) => {
    try {
      await fetch(API_BASE + '/keys/delete?id=' + id, {
        method: 'DELETE',
        credentials: 'include'
      })
      setExistingKeys(existingKeys.filter((k) => k.id !== id))
      toast.success("API Key Deleted", { description: `"${name}" has been deleted.` })
    } catch (error) {
      toast.error("Error", { description: "Failed to delete API key." })
    }
  }

  const formatDate = (date: string) => {
    return new Date(date).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })
  }

  const formatLastUsed = (date: string | null) => {
    if (!date) return "Never"
    const d = new Date(date)
    const now = new Date()
    const diff = now.getTime() - d.getTime()
    const minutes = Math.floor(diff / 60000)
    const hours = Math.floor(diff / 3600000)
    const days = Math.floor(diff / 86400000)
    if (minutes < 1) return "Just now"
    if (minutes < 60) return `${minutes}m ago`
    if (hours < 24) return `${hours}h ago`
    if (days < 7) return `${days}d ago`
    return formatDate(date)
  }

  const getLLMModels = (provider: string) => {
    if (provider === 'claude') {
      return [
        { value: 'claude-3-5-sonnet-20241022', label: 'Claude 3.5 Sonnet' },
        { value: 'claude-3-opus-20240229', label: 'Claude 3 Opus' },
        { value: 'claude-3-5-haiku-20241022', label: 'Claude 3.5 Haiku' },
      ]
    } else if (provider === 'openai') {
      return [
        { value: 'gpt-4o', label: 'GPT-4o' },
        { value: 'gpt-4o-mini', label: 'GPT-4o Mini' },
        { value: 'gpt-4-turbo', label: 'GPT-4 Turbo' },
      ]
    } else if (provider === 'google') {
      return [
        { value: 'gemini-1.5-pro', label: 'Gemini 1.5 Pro' },
        { value: 'gemini-1.5-flash', label: 'Gemini 1.5 Flash' },
      ]
    }
    return []
  }

  const getModelName = (modelId: string) => {
    const model = fineTunedModels.find(m => m.id === modelId)
    return model?.name || modelId?.slice(0, 8) + '...'
  }

  const getLLMDisplayName = (model: string) => {
    const allModels = [
      ...getLLMModels('claude'),
      ...getLLMModels('openai'),
      ...getLLMModels('google')
    ]
    return allModels.find(m => m.value === model)?.label || model
  }

  const selectedKey = existingKeys.find(k => k.id === selectedKeyId) || existingKeys[0]
  const displayKey = selectedKey?.key && selectedKey.key.length > 20 ? selectedKey.key : "YOUR_API_KEY"

  return (
    <div className="flex min-h-screen">
      <Toaster position="top-right" />
      <Sidebar>
        <div className="p-4 sm:p-6 pt-8 sm:pt-12 space-y-4 sm:space-y-6">
          <Card className="bg-card border-border">
            <CardHeader className="p-4 sm:p-6">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div>
                  <CardTitle className="text-foreground text-lg sm:text-xl">API Keys</CardTitle>
                  <CardDescription className="text-xs sm:text-sm">Manage your API keys for accessing SchemaLabs AI</CardDescription>
                </div>
                <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
                  <DialogTrigger asChild>
                    <Button className="w-full sm:w-auto" disabled={fineTunedModels.length === 0}>
                      <Plus className="h-4 w-4 mr-2" />
                      Create API Key
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="max-w-md">
                    <DialogHeader>
                      <DialogTitle>Create New API Key</DialogTitle>
                      <DialogDescription>Generate a new API key for your application</DialogDescription>
                    </DialogHeader>
                    {generatedKey ? (
                      <div className="space-y-4">
                        <div className="p-3 bg-amber-500/10 border border-amber-500/30 rounded-lg flex items-start gap-2">
                          <AlertTriangle className="h-4 w-4 text-amber-500 mt-0.5 flex-shrink-0" />
                          <p className="text-xs text-amber-600 dark:text-amber-400">
                            Make sure to copy your API key now. You won't be able to see it again!
                          </p>
                        </div>
                        <div className="p-3 bg-muted rounded-lg">
                          <p className="text-xs text-muted-foreground mb-1">Your API Key</p>
                          <div className="flex items-center gap-2">
                            <code className="flex-1 text-xs font-mono break-all">{generatedKey}</code>
                            <Button variant="outline" size="sm" onClick={copyGeneratedKey}>
                              {generatedKeyCopied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                            </Button>
                          </div>
                        </div>
                        <DialogFooter>
                          <Button onClick={handleCloseDialog} className="w-full">Done</Button>
                        </DialogFooter>
                      </div>
                    ) : (
                      <div className="space-y-4">
                        <div className="space-y-2">
                          <Label>Key Name <span className="text-destructive">*</span></Label>
                          <Input 
                            placeholder="My API Key" 
                            value={newKeyName}
                            onChange={(e) => setNewKeyName(e.target.value)}
                          />
                        </div>
                        <div className="space-y-2">
                          <Label>Fine-tuned Model <span className="text-destructive">*</span></Label>
                          <Select value={selectedFineTunedModel} onValueChange={setSelectedFineTunedModel}>
                            <SelectTrigger>
                              <SelectValue placeholder="Select a model" />
                            </SelectTrigger>
                            <SelectContent>
                              {fineTunedModels.map((model) => (
                                <SelectItem key={model.id} value={model.id}>
                                  <div className="flex items-center gap-2">
                                    <Brain className="h-4 w-4 text-primary" />
                                    {model.name}
                                  </div>
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                          {fineTunedModels.length === 0 && (
                            <p className="text-xs text-muted-foreground">No fine-tuned models available. Train one first.</p>
                          )}
                        </div>
                        <div className="grid grid-cols-2 gap-4">
                          <div className="space-y-2">
                            <Label>LLM Provider <span className="text-destructive">*</span></Label>
                            <Select value={selectedLLMProvider} onValueChange={(v) => {
                              setSelectedLLMProvider(v)
                              setSelectedLLMModel(getLLMModels(v)[0]?.value || '')
                            }}>
                              <SelectTrigger>
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="claude">Claude</SelectItem>
                                <SelectItem value="openai">OpenAI</SelectItem>
                                <SelectItem value="google">Google</SelectItem>
                              </SelectContent>
                            </Select>
                          </div>
                          <div className="space-y-2">
                            <Label>LLM Model <span className="text-destructive">*</span></Label>
                            <Select value={selectedLLMModel} onValueChange={setSelectedLLMModel}>
                              <SelectTrigger>
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent>
                                {getLLMModels(selectedLLMProvider).map((m) => (
                                  <SelectItem key={m.value} value={m.value}>{m.label}</SelectItem>
                                ))}
                              </SelectContent>
                            </Select>
                          </div>
                        </div>
                        <div className="space-y-2">
                          <Label>Rate Limit</Label>
                          <Select value={newKeyRateLimit} onValueChange={setNewKeyRateLimit}>
                            <SelectTrigger>
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="100/min">100/min</SelectItem>
                              <SelectItem value="1000/min">1000/min</SelectItem>
                              <SelectItem value="10000/min">10000/min</SelectItem>
                              <SelectItem value="unlimited">Unlimited</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                        <div className="space-y-2">
                          <Label>Permissions</Label>
                          <div className="grid grid-cols-2 gap-2">
                            {Object.entries(permissions).map(([key, value]) => (
                              <div key={key} className="flex items-center space-x-2">
                                <Checkbox
                                  id={key}
                                  checked={value}
                                  onCheckedChange={(checked) => 
                                    setPermissions(prev => ({ ...prev, [key]: !!checked }))
                                  }
                                />
                                <label htmlFor={key} className="text-sm capitalize">{key}</label>
                              </div>
                            ))}
                          </div>
                        </div>
                        <DialogFooter>
                          <DialogClose asChild>
                            <Button variant="outline">Cancel</Button>
                          </DialogClose>
                          <Button onClick={handleGenerateKey} disabled={creating || !selectedFineTunedModel}>
                            {creating ? "Creating..." : "Create Key"}
                          </Button>
                        </DialogFooter>
                      </div>
                    )}
                  </DialogContent>
                </Dialog>
              </div>
            </CardHeader>
            <CardContent className="p-4 sm:p-6 pt-0">
              {loading ? (
                <div className="text-center py-8 text-muted-foreground">Loading...</div>
              ) : existingKeys.length === 0 ? (
                <div className="text-center py-8">
                  <Key className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                  <p className="text-muted-foreground">No API keys yet. Create one to get started.</p>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow className="border-border">
                        <TableHead className="text-muted-foreground">Name</TableHead>
                        <TableHead className="text-muted-foreground">Key</TableHead>
                        <TableHead className="text-muted-foreground hidden md:table-cell">Fine-tuned Model</TableHead>
                        <TableHead className="text-muted-foreground hidden lg:table-cell">LLM</TableHead>
                        <TableHead className="text-muted-foreground hidden sm:table-cell">Permissions</TableHead>
                        <TableHead className="text-muted-foreground hidden xl:table-cell">Rate Limit</TableHead>
                        <TableHead className="text-muted-foreground hidden xl:table-cell">Requests</TableHead>
                        <TableHead className="text-muted-foreground hidden sm:table-cell">Last Used</TableHead>
                        <TableHead className="text-muted-foreground text-right">Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {existingKeys.map((apiKey) => (
                        <TableRow 
                          key={apiKey.id} 
                          className={"border-border cursor-pointer transition-all " + (selectedKeyId === apiKey.id ? "bg-primary/5" : "hover:bg-muted/50")} 
                          onClick={() => setSelectedKeyId(apiKey.id)}
                        >
                          <TableCell className="font-medium text-foreground">
                            <div className="flex items-center gap-2">
                              <Key className="h-4 w-4 text-muted-foreground" />
                              {apiKey.name}
                            </div>
                          </TableCell>
                          <TableCell>
                            <code className="text-xs bg-muted px-2 py-1 rounded">
                              {apiKey.key.slice(0, 12)}...{apiKey.key.slice(-4)}
                            </code>
                          </TableCell>
                          <TableCell className="hidden md:table-cell">
                            {apiKey.finetuned_model ? (
                              <div className="flex items-center gap-1">
                                <Brain className="h-3 w-3 text-primary" />
                                <span className="text-xs">{getModelName(apiKey.finetuned_model)}</span>
                              </div>
                            ) : (
                              <span className="text-xs text-muted-foreground">-</span>
                            )}
                          </TableCell>
                          <TableCell className="hidden lg:table-cell">
                            <span className="text-xs">{getLLMDisplayName(apiKey.llm_model)}</span>
                          </TableCell>
                          <TableCell className="hidden sm:table-cell">
                            <div className="flex flex-wrap gap-1">
                              {apiKey.permissions.map((perm) => (
                                <Badge key={perm} variant="outline" className="text-[10px] px-1.5 py-0">
                                  {perm}
                                </Badge>
                              ))}
                            </div>
                          </TableCell>
                          <TableCell className="text-muted-foreground text-xs hidden xl:table-cell">
                            {apiKey.rate_limit}
                          </TableCell>
                          <TableCell className="text-muted-foreground text-xs hidden xl:table-cell">
                            {apiKey.requests.toLocaleString()}
                          </TableCell>
                          <TableCell className="text-muted-foreground text-xs hidden sm:table-cell">
                            {formatLastUsed(apiKey.last_used)}
                          </TableCell>
                          <TableCell className="text-right">
                            <div className="flex items-center justify-end gap-1">
                              <Button 
                                variant="ghost" 
                                size="icon" 
                                className="h-8 w-8"
                                onClick={(e) => { e.stopPropagation(); copyKey(apiKey.id, apiKey.key) }}
                              >
                                {copiedKey === apiKey.id ? (
                                  <Check className="h-4 w-4 text-green-500" />
                                ) : (
                                  <Copy className="h-4 w-4" />
                                )}
                              </Button>
                              <Button 
                                variant="ghost" 
                                size="icon" 
                                className="h-8 w-8 text-destructive hover:text-destructive"
                                onClick={(e) => { e.stopPropagation(); deleteKey(apiKey.id, apiKey.name) }}
                              >
                                <Trash2 className="h-4 w-4" />
                              </Button>
                            </div>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="bg-card border-border">
            <CardHeader className="p-4 sm:p-6">
              <CardTitle className="text-foreground text-base sm:text-lg">Quick Start</CardTitle>
              <CardDescription className="text-xs sm:text-sm">
                {selectedKeyId 
                  ? `Code example for "${existingKeys.find(k => k.id === selectedKeyId)?.name}"`
                  : "Click on an API key above to see its code example"
                }
              </CardDescription>
            </CardHeader>
            <CardContent className="p-4 sm:p-6 pt-0 sm:pt-0">
              {!selectedKey ? (
                <p className="text-muted-foreground text-sm">Create an API key to see example code</p>
              ) : (
                <div className="space-y-4">
                  <div className="rounded-lg bg-muted border border-border p-3 sm:p-4 overflow-x-auto">
                    <div className="flex items-center justify-between mb-2">
                      <p className="text-[10px] sm:text-xs text-muted-foreground">Chat with LLM + Fine-tuned Model</p>
                      <Button 
                        variant="ghost" 
                        size="sm" 
                        className="h-6 px-2"
                        onClick={() => {
                          const cmd = `curl -X POST https://api.schemalabs.ai/v1/chat \\\n  -H "Authorization: Bearer ${displayKey}" \\\n  -H "Content-Type: application/json" \\\n  -d '{"message": "Explain the prediction"}'`
                          navigator.clipboard.writeText(cmd)
                          toast.success("Copied!", { description: "cURL command copied to clipboard." })
                        }}
                      >
                        <Copy className="h-3 w-3" />
                      </Button>
                    </div>
                    <pre className="text-xs sm:text-sm font-mono text-foreground whitespace-pre-wrap">{`curl -X POST https://api.schemalabs.ai/v1/chat \\
  -H "Authorization: Bearer ${displayKey}" \\
  -H "Content-Type: application/json" \\
  -d '{"message": "Explain the prediction"}'`}</pre>
                  </div>
                  <div className="rounded-lg bg-muted border border-border p-3 sm:p-4 overflow-x-auto">
                    <div className="flex items-center justify-between mb-2">
                      <p className="text-[10px] sm:text-xs text-muted-foreground">Prediction Only (No LLM)</p>
                      <Button 
                        variant="ghost" 
                        size="sm" 
                        className="h-6 px-2"
                        onClick={() => {
                          const cmd = `curl -X POST https://api.schemalabs.ai/v1/predict \\\n  -H "Authorization: Bearer ${displayKey}" \\\n  -H "Content-Type: application/json" \\\n  -d '{"data": {"feature1": 0.5, "feature2": 1.2}}'`
                          navigator.clipboard.writeText(cmd)
                          toast.success("Copied!", { description: "cURL command copied to clipboard." })
                        }}
                      >
                        <Copy className="h-3 w-3" />
                      </Button>
                    </div>
                    <pre className="text-xs sm:text-sm font-mono text-foreground whitespace-pre-wrap">{`curl -X POST https://api.schemalabs.ai/v1/predict \\
  -H "Authorization: Bearer ${displayKey}" \\
  -H "Content-Type: application/json" \\
  -d '{"data": {"feature1": 0.5, "feature2": 1.2}}'`}</pre>
                  </div>
                  {selectedKey.finetuned_model && (
                    <div className="flex items-center gap-4 text-xs text-muted-foreground pt-2 border-t border-border">
                      <span className="flex items-center gap-1">
                        <Brain className="h-3 w-3" />
                        Fine-tuned: {getModelName(selectedKey.finetuned_model)}
                      </span>
                      <span>|</span>
                      <span>LLM: {getLLMDisplayName(selectedKey.llm_model)}</span>
                      <span>|</span>
                      <span>Rate: {selectedKey.rate_limit}</span>
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </Sidebar>
    </div>
  )
}

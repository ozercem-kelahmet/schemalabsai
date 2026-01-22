"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Checkbox } from "@/components/ui/checkbox"
import {
  Settings,
  Key,
  Globe,
  Plus,
  Copy,
  Eye,
  EyeOff,
  Trash2,
  MoreHorizontal,
  CheckCircle2,
  AlertCircle,
  ExternalLink,
  ChevronDown,
} from "lucide-react"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
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
import { mockModels } from "@/lib/mock-data"

interface APIKey {
  id: string
  name: string
  key: string
  baseModel: string
  rateLimit: string
  permissions: string[]
  createdAt: Date
  lastUsed: Date | null
  status: "active" | "revoked"
}

interface Endpoint {
  id: string
  name: string
  url: string
  modelId: string
  modelName: string
  description: string
  status: "active" | "inactive"
  requests: number
  createdAt: Date
}

const mockAPIKeys: APIKey[] = [
  {
    id: "key-1",
    name: "Production Key",
    key: "sk-schema-prod-xxxxxxxxxxxxxxxxxxxx",
    baseModel: "schema-v0",
    rateLimit: "1000/min",
    permissions: ["read", "query"],
    createdAt: new Date("2024-01-10"),
    lastUsed: new Date("2024-01-18"),
    status: "active",
  },
  {
    id: "key-2",
    name: "Development Key",
    key: "sk-schema-dev-xxxxxxxxxxxxxxxxxxxx",
    baseModel: "schema-v0",
    rateLimit: "100/min",
    permissions: ["read", "query", "write"],
    createdAt: new Date("2024-01-05"),
    lastUsed: new Date("2024-01-17"),
    status: "active",
  },
]

const mockEndpoints: Endpoint[] = [
  {
    id: "ep-1",
    name: "Churn Prediction",
    url: "/v1/models/cust-intl/predict",
    modelId: "model-001",
    modelName: "Customer Intelligence",
    description: "Predict customer churn probability",
    status: "active",
    requests: 12847,
    createdAt: new Date("2024-01-16"),
  },
  {
    id: "ep-2",
    name: "Risk Assessment",
    url: "/v1/models/fin-risk/assess",
    modelId: "model-002",
    modelName: "Financial Risk Model",
    description: "Assess financial risk for loan applications",
    status: "active",
    requests: 8432,
    createdAt: new Date("2024-01-14"),
  },
]

const rateLimitOptions = [
  { value: "100/min", label: "100 requests/min" },
  { value: "1000/min", label: "1,000 requests/min" },
  { value: "10000/min", label: "10,000 requests/min" },
  { value: "unlimited", label: "Unlimited" },
]

const permissionOptions = [
  { value: "read", label: "Read", description: "View model information and metadata" },
  { value: "query", label: "Query", description: "Send queries to models" },
  { value: "write", label: "Write", description: "Update model configurations" },
  { value: "delete", label: "Delete", description: "Delete models and data" },
]

export default function ConfigurationPage() {
  const [apiKeys, setApiKeys] = useState<APIKey[]>(mockAPIKeys)
  const [endpoints, setEndpoints] = useState<Endpoint[]>(mockEndpoints)
  const [showKeys, setShowKeys] = useState<Record<string, boolean>>({})
  
  // Create Key Modal State
  const [createKeyModalOpen, setCreateKeyModalOpen] = useState(false)
  const [newKeyName, setNewKeyName] = useState("")
  const [newKeyBaseModel, setNewKeyBaseModel] = useState("schema-v0")
  const [newKeyRateLimit, setNewKeyRateLimit] = useState("1000/min")
  const [newKeyPermissions, setNewKeyPermissions] = useState<string[]>(["read", "query"])
  
  // Create Endpoint Modal State
  const [createEndpointModalOpen, setCreateEndpointModalOpen] = useState(false)
  const [newEndpointModel, setNewEndpointModel] = useState("")
  const [newEndpointName, setNewEndpointName] = useState("")
  const [newEndpointPath, setNewEndpointPath] = useState("")
  const [newEndpointDescription, setNewEndpointDescription] = useState("")
  
  // Test Endpoint Modal State
  const [testEndpointModalOpen, setTestEndpointModalOpen] = useState(false)
  const [testingEndpoint, setTestingEndpoint] = useState<Endpoint | null>(null)
  const [testQuery, setTestQuery] = useState("")
  const [testResponse, setTestResponse] = useState<string | null>(null)
  const [isTesting, setIsTesting] = useState(false)

  const toggleKeyVisibility = (keyId: string) => {
    setShowKeys((prev) => ({ ...prev, [keyId]: !prev[keyId] }))
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
  }

  const formatDate = (date: Date) => {
    return new Intl.DateTimeFormat("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    }).format(date)
  }

  const formatNumber = (num: number) => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`
    return num.toString()
  }

  const togglePermission = (permission: string) => {
    setNewKeyPermissions((prev) =>
      prev.includes(permission)
        ? prev.filter((p) => p !== permission)
        : [...prev, permission]
    )
  }

  const createAPIKey = () => {
    if (newKeyName.trim() && newKeyPermissions.length > 0) {
      const newKey: APIKey = {
        id: `key-${Date.now()}`,
        name: newKeyName.trim(),
        key: `sk-schema-${Math.random().toString(36).substring(2, 10)}-${"x".repeat(20)}`,
        baseModel: newKeyBaseModel,
        rateLimit: newKeyRateLimit,
        permissions: newKeyPermissions,
        createdAt: new Date(),
        lastUsed: null,
        status: "active",
      }
      setApiKeys((prev) => [newKey, ...prev])
      // Reset form
      setNewKeyName("")
      setNewKeyBaseModel("schema-v0")
      setNewKeyRateLimit("1000/min")
      setNewKeyPermissions(["read", "query"])
      setCreateKeyModalOpen(false)
    }
  }

  const createEndpoint = () => {
    if (newEndpointModel && newEndpointName.trim() && newEndpointPath.trim()) {
      const selectedModel = mockModels.find((m) => m.id === newEndpointModel)
      const newEndpoint: Endpoint = {
        id: `ep-${Date.now()}`,
        name: newEndpointName.trim(),
        url: newEndpointPath.startsWith("/") ? newEndpointPath : `/${newEndpointPath}`,
        modelId: newEndpointModel,
        modelName: selectedModel?.name || "Unknown Model",
        description: newEndpointDescription.trim(),
        status: "active",
        requests: 0,
        createdAt: new Date(),
      }
      setEndpoints((prev) => [newEndpoint, ...prev])
      // Reset form
      setNewEndpointModel("")
      setNewEndpointName("")
      setNewEndpointPath("")
      setNewEndpointDescription("")
      setCreateEndpointModalOpen(false)
    }
  }

  const revokeKey = (keyId: string) => {
    setApiKeys((prev) =>
      prev.map((k) => (k.id === keyId ? { ...k, status: "revoked" as const } : k))
    )
  }

  const deleteKey = (keyId: string) => {
    setApiKeys((prev) => prev.filter((k) => k.id !== keyId))
  }

  const deleteEndpoint = (endpointId: string) => {
    setEndpoints((prev) => prev.filter((e) => e.id !== endpointId))
  }

  const openTestEndpoint = (endpoint: Endpoint) => {
    setTestingEndpoint(endpoint)
    setTestQuery("")
    setTestResponse(null)
    setTestEndpointModalOpen(true)
  }

  const runEndpointTest = async () => {
    if (!testQuery.trim() || !testingEndpoint) return
    
    setIsTesting(true)
    setTestResponse(null)
    
    // Simulate API call
    setTimeout(() => {
      const mockResponse = {
        status: 200,
        latency: `${Math.floor(Math.random() * 200) + 50}ms`,
        data: {
          query: testQuery.trim(),
          model: testingEndpoint.modelName,
          endpoint: testingEndpoint.url,
          result: {
            prediction: Math.random() > 0.5 ? "positive" : "negative",
            confidence: (Math.random() * 0.3 + 0.7).toFixed(3),
            processed_at: new Date().toISOString(),
          },
          tokens_used: Math.floor(Math.random() * 500) + 100,
        },
      }
      setTestResponse(JSON.stringify(mockResponse, null, 2))
      setIsTesting(false)
    }, 1000)
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[#0052CC]/10 dark:bg-[#0052CC]/20">
          <Settings className="h-5 w-5 text-[#0052CC] dark:text-[#2684FF]" />
        </div>
        <div>
          <h1 className="text-xl font-semibold text-foreground">Configuration</h1>
          <p className="text-sm text-muted-foreground">Manage API keys and endpoints</p>
        </div>
      </div>

      {/* API Keys Section */}
      <Card className="border-border bg-card">
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2 text-foreground">
              <Key className="h-5 w-5" />
              API Keys
            </CardTitle>
            <CardDescription className="text-muted-foreground">
              Manage your API keys for accessing Schema models
            </CardDescription>
          </div>
          <Button
            onClick={() => setCreateKeyModalOpen(true)}
            className="gap-2 bg-[#0052CC] text-white hover:bg-[#003D99]"
          >
            <Plus className="h-4 w-4" />
            Create Key
          </Button>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {apiKeys.map((apiKey) => (
              <div
                key={apiKey.id}
                className={`flex items-center justify-between rounded-lg border p-4 ${
                  apiKey.status === "revoked"
                    ? "border-red-500/30 bg-red-500/5"
                    : "border-border bg-muted/30"
                }`}
              >
                <div className="space-y-1.5 flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="font-medium text-foreground">{apiKey.name}</span>
                    <span className="rounded bg-muted px-2 py-0.5 text-xs text-muted-foreground">
                      {apiKey.baseModel}
                    </span>
                    <span className="rounded bg-muted px-2 py-0.5 text-xs text-muted-foreground">
                      {apiKey.rateLimit}
                    </span>
                    {apiKey.status === "active" ? (
                      <span className="flex items-center gap-1 rounded bg-emerald-500/10 px-2 py-0.5 text-xs text-emerald-500">
                        <CheckCircle2 className="h-3 w-3" />
                        Active
                      </span>
                    ) : (
                      <span className="flex items-center gap-1 rounded bg-red-500/10 px-2 py-0.5 text-xs text-red-500">
                        <AlertCircle className="h-3 w-3" />
                        Revoked
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-4 text-sm text-muted-foreground">
                    <span className="font-mono text-xs truncate max-w-[300px]">
                      {showKeys[apiKey.id] ? apiKey.key : apiKey.key.replace(/./g, "•").slice(0, 30)}
                    </span>
                    <button
                      onClick={() => toggleKeyVisibility(apiKey.id)}
                      className="hover:text-foreground transition-colors shrink-0"
                    >
                      {showKeys[apiKey.id] ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </button>
                    <button
                      onClick={() => copyToClipboard(apiKey.key)}
                      className="hover:text-foreground transition-colors shrink-0"
                    >
                      <Copy className="h-4 w-4" />
                    </button>
                  </div>
                  <div className="flex items-center gap-2 flex-wrap">
                    {apiKey.permissions.map((perm) => (
                      <span key={perm} className="rounded bg-[#0052CC]/10 px-2 py-0.5 text-xs text-[#0052CC] dark:text-[#2684FF]">
                        {perm}
                      </span>
                    ))}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Created {formatDate(apiKey.createdAt)}
                    {apiKey.lastUsed && ` • Last used ${formatDate(apiKey.lastUsed)}`}
                  </p>
                </div>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground shrink-0">
                      <MoreHorizontal className="h-4 w-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="border-border bg-popover">
                    {apiKey.status === "active" && (
                      <DropdownMenuItem
                        onClick={() => revokeKey(apiKey.id)}
                        className="text-amber-500 focus:text-amber-500"
                      >
                        <AlertCircle className="mr-2 h-4 w-4" />
                        Revoke Key
                      </DropdownMenuItem>
                    )}
                    <DropdownMenuItem
                      onClick={() => deleteKey(apiKey.id)}
                      className="text-red-500 focus:text-red-500"
                    >
                      <Trash2 className="mr-2 h-4 w-4" />
                      Delete
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            ))}

            {apiKeys.length === 0 && (
              <div className="flex flex-col items-center justify-center py-8 text-center">
                <Key className="h-12 w-12 text-muted-foreground" />
                <p className="mt-4 text-muted-foreground">No API keys created yet</p>
                <p className="text-sm text-muted-foreground">
                  Create an API key to access your Schema models
                </p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Endpoints Section */}
      <Card className="border-border bg-card">
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2 text-foreground">
              <Globe className="h-5 w-5" />
              Endpoints
            </CardTitle>
            <CardDescription className="text-muted-foreground">
              Create and manage your model API endpoints
            </CardDescription>
          </div>
          <Button
            onClick={() => setCreateEndpointModalOpen(true)}
            className="gap-2 bg-[#0052CC] text-white hover:bg-[#003D99]"
          >
            <Plus className="h-4 w-4" />
            Create Endpoint
          </Button>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {endpoints.map((endpoint) => (
              <div
                key={endpoint.id}
                className="flex items-center justify-between rounded-lg border border-border bg-muted/30 p-4"
              >
                <div className="space-y-1.5 flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="font-medium text-foreground">{endpoint.name}</span>
                    <span className="rounded bg-muted px-2 py-0.5 text-xs text-muted-foreground">
                      {endpoint.modelName}
                    </span>
                    {endpoint.status === "active" ? (
                      <span className="flex items-center gap-1 rounded bg-emerald-500/10 px-2 py-0.5 text-xs text-emerald-500">
                        <CheckCircle2 className="h-3 w-3" />
                        Active
                      </span>
                    ) : (
                      <span className="rounded bg-muted px-2 py-0.5 text-xs text-muted-foreground">
                        Inactive
                      </span>
                    )}
                  </div>
                  {endpoint.description && (
                    <p className="text-sm text-muted-foreground">{endpoint.description}</p>
                  )}
                  <div className="flex items-center gap-2 text-sm">
                    <code className="rounded bg-muted px-2 py-0.5 font-mono text-xs text-[#0052CC] dark:text-[#2684FF] truncate max-w-[400px]">
                      https://api.schemalabs.ai{endpoint.url}
                    </code>
                    <button
                      onClick={() => copyToClipboard(`https://api.schemalabs.ai${endpoint.url}`)}
                      className="text-muted-foreground hover:text-foreground transition-colors shrink-0"
                    >
                      <Copy className="h-3.5 w-3.5" />
                    </button>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {formatNumber(endpoint.requests)} requests • Created {formatDate(endpoint.createdAt)}
                  </p>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <Button 
                    variant="outline" 
                    size="sm" 
                    className="gap-2 bg-transparent"
                    onClick={() => openTestEndpoint(endpoint)}
                  >
                    <ExternalLink className="h-3.5 w-3.5" />
                    Test
                  </Button>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground">
                        <MoreHorizontal className="h-4 w-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end" className="border-border bg-popover">
                      <DropdownMenuItem
                        onClick={() => deleteEndpoint(endpoint.id)}
                        className="text-red-500 focus:text-red-500"
                      >
                        <Trash2 className="mr-2 h-4 w-4" />
                        Delete
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              </div>
            ))}

            {endpoints.length === 0 && (
              <div className="flex flex-col items-center justify-center py-8 text-center">
                <Globe className="h-12 w-12 text-muted-foreground" />
                <p className="mt-4 text-muted-foreground">No endpoints created yet</p>
                <p className="text-sm text-muted-foreground">
                  Create an endpoint to expose your models via API
                </p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Create API Key Modal */}
      <Dialog open={createKeyModalOpen} onOpenChange={setCreateKeyModalOpen}>
        <DialogContent className="border-border bg-card sm:max-w-[500px]">
          <DialogHeader>
            <DialogTitle className="text-foreground">Create API Key</DialogTitle>
            <DialogDescription className="text-muted-foreground">
              Create a new API key to access your Schema models
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="key-name" className="text-foreground">
                Key Name
              </Label>
              <Input
                id="key-name"
                placeholder="e.g., Production Key"
                value={newKeyName}
                onChange={(e) => setNewKeyName(e.target.value)}
                className="border-border bg-background text-foreground"
              />
            </div>

            <div className="space-y-2">
              <Label className="text-foreground">Base Model</Label>
              <Select value={newKeyBaseModel} onValueChange={setNewKeyBaseModel}>
                <SelectTrigger className="border-border bg-background text-foreground">
                  <SelectValue placeholder="Select base model" />
                </SelectTrigger>
                <SelectContent className="border-border bg-popover">
                  <SelectItem value="schema-v0">schema-v0</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label className="text-foreground">Rate Limit</Label>
              <Select value={newKeyRateLimit} onValueChange={setNewKeyRateLimit}>
                <SelectTrigger className="border-border bg-background text-foreground">
                  <SelectValue placeholder="Select rate limit" />
                </SelectTrigger>
                <SelectContent className="border-border bg-popover">
                  {rateLimitOptions.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label className="text-foreground">Permissions</Label>
              <div className="space-y-2">
                {permissionOptions.map((perm) => (
                  <div
                    key={perm.value}
                    className="flex items-start gap-3 rounded-lg border border-border bg-muted/30 p-3 cursor-pointer hover:bg-muted/50 transition-colors"
                    onClick={() => togglePermission(perm.value)}
                  >
                    <Checkbox
                      id={`perm-${perm.value}`}
                      checked={newKeyPermissions.includes(perm.value)}
                      onCheckedChange={() => togglePermission(perm.value)}
                      className="mt-0.5"
                    />
                    <div className="flex-1">
                      <label
                        htmlFor={`perm-${perm.value}`}
                        className="text-sm font-medium text-foreground cursor-pointer"
                      >
                        {perm.label}
                      </label>
                      <p className="text-xs text-muted-foreground">{perm.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateKeyModalOpen(false)} className="bg-transparent">
              Cancel
            </Button>
            <Button
              onClick={createAPIKey}
              disabled={!newKeyName.trim() || newKeyPermissions.length === 0}
              className="bg-[#0052CC] text-white hover:bg-[#003D99]"
            >
              Create Key
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Create Endpoint Modal */}
      <Dialog open={createEndpointModalOpen} onOpenChange={setCreateEndpointModalOpen}>
        <DialogContent className="border-border bg-card sm:max-w-[500px]">
          <DialogHeader>
            <DialogTitle className="text-foreground">Create Endpoint</DialogTitle>
            <DialogDescription className="text-muted-foreground">
              Create a new API endpoint for your model
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label className="text-foreground">Model</Label>
              <Select value={newEndpointModel} onValueChange={setNewEndpointModel}>
                <SelectTrigger className="border-border bg-background text-foreground">
                  <SelectValue placeholder="Select a model" />
                </SelectTrigger>
                <SelectContent className="border-border bg-popover">
                  {mockModels.map((model) => (
                    <SelectItem key={model.id} value={model.id}>
                      {model.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="endpoint-name" className="text-foreground">
                Endpoint Name
              </Label>
              <Input
                id="endpoint-name"
                placeholder="e.g., Churn Prediction"
                value={newEndpointName}
                onChange={(e) => setNewEndpointName(e.target.value)}
                className="border-border bg-background text-foreground"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="endpoint-path" className="text-foreground">
                URL Path
              </Label>
              <div className="flex items-center">
                <span className="rounded-l-md border border-r-0 border-border bg-muted px-3 py-2 text-sm text-muted-foreground">
                  https://api.schemalabs.ai
                </span>
                <Input
                  id="endpoint-path"
                  placeholder="/v1/models/my-model/predict"
                  value={newEndpointPath}
                  onChange={(e) => setNewEndpointPath(e.target.value)}
                  className="border-border bg-background text-foreground rounded-l-none"
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="endpoint-description" className="text-foreground">
                Description (Optional)
              </Label>
              <Textarea
                id="endpoint-description"
                placeholder="Describe what this endpoint does..."
                value={newEndpointDescription}
                onChange={(e) => setNewEndpointDescription(e.target.value)}
                className="border-border bg-background text-foreground resize-none"
                rows={3}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateEndpointModalOpen(false)} className="bg-transparent">
              Cancel
            </Button>
            <Button
              onClick={createEndpoint}
              disabled={!newEndpointModel || !newEndpointName.trim() || !newEndpointPath.trim()}
              className="bg-[#0052CC] text-white hover:bg-[#003D99]"
            >
              Create Endpoint
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Test Endpoint Modal */}
      <Dialog open={testEndpointModalOpen} onOpenChange={setTestEndpointModalOpen}>
        <DialogContent className="border-border bg-card sm:max-w-[600px]">
          <DialogHeader>
            <DialogTitle className="text-foreground">Test Endpoint</DialogTitle>
            <DialogDescription className="text-muted-foreground">
              Test your endpoint with a custom query
            </DialogDescription>
          </DialogHeader>
          
          {testingEndpoint && (
            <div className="space-y-4">
              {/* Endpoint Info Card */}
              <div className="rounded-lg border border-border bg-muted/30 p-4 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="font-medium text-foreground">{testingEndpoint.name}</span>
                  <span className="rounded bg-muted px-2 py-0.5 text-xs text-muted-foreground">
                    {testingEndpoint.modelName}
                  </span>
                </div>
                <code className="block rounded bg-muted px-2 py-1 font-mono text-xs text-[#0052CC] dark:text-[#2684FF]">
                  POST https://api.schemalabs.ai{testingEndpoint.url}
                </code>
                {testingEndpoint.description && (
                  <p className="text-sm text-muted-foreground">{testingEndpoint.description}</p>
                )}
              </div>

              {/* Query Input */}
              <div className="space-y-2">
                <Label htmlFor="test-query" className="text-foreground">
                  Query
                </Label>
                <Textarea
                  id="test-query"
                  placeholder='{"input": "your query here", "options": {}}'
                  value={testQuery}
                  onChange={(e) => setTestQuery(e.target.value)}
                  className="border-border bg-background text-foreground font-mono text-sm resize-none"
                  rows={4}
                />
              </div>

              {/* Response Output */}
              {(testResponse || isTesting) && (
                <div className="space-y-2">
                  <Label className="text-foreground">Response</Label>
                  <div className="rounded-lg border border-border bg-muted/50 p-3 max-h-[200px] overflow-auto">
                    {isTesting ? (
                      <div className="flex items-center gap-2 text-muted-foreground">
                        <div className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                        <span className="text-sm">Running test...</span>
                      </div>
                    ) : (
                      <pre className="font-mono text-xs text-foreground whitespace-pre-wrap">
                        {testResponse}
                      </pre>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
          
          <DialogFooter>
            <Button 
              variant="outline" 
              onClick={() => setTestEndpointModalOpen(false)} 
              className="bg-transparent"
            >
              Close
            </Button>
            <Button
              onClick={runEndpointTest}
              disabled={!testQuery.trim() || isTesting}
              className="bg-[#0052CC] text-white hover:bg-[#003D99]"
            >
              {isTesting ? "Running..." : "Run Test"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

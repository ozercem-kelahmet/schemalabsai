"use client"

import { useState, useEffect } from "react"
import { Sidebar } from "@/components/sidebar"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
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
import { Plus, ExternalLink, Copy, Trash2, Check, Brain, Loader2, Play, Code } from "lucide-react"
import { useToast } from "@/components/ui/toast"
import { API_BASE } from "@/lib/config"

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
}

interface FineTunedModel {
  id: string
  name: string
}

const llmModels = [
  { id: "gpt-4o", name: "GPT-4o" },
  { id: "gpt-4o-mini", name: "GPT-4o Mini" },
  { id: "claude-3-5-sonnet", name: "Claude 3.5 Sonnet" },
  { id: "claude-3-5-haiku", name: "Claude 3.5 Haiku" },
  { id: "gemini-1.5-pro", name: "Gemini 1.5 Pro" },
]

export default function EndpointsPage() {
  const [endpoints, setEndpoints] = useState<Endpoint[]>([])
  const [fineTunedModels, setFineTunedModels] = useState<FineTunedModel[]>([])
  const [loading, setLoading] = useState(true)
  const [creating, setCreating] = useState(false)
  const [copiedUrl, setCopiedUrl] = useState<string | null>(null)
  const [isDialogOpen, setIsDialogOpen] = useState(false)
  const { addToast } = useToast()
  
  // Form state
  const [newEndpointName, setNewEndpointName] = useState("")
  const [newEndpointPath, setNewEndpointPath] = useState("")
  const [selectedFineTunedModel, setSelectedFineTunedModel] = useState("")
  const [selectedLLMModel, setSelectedLLMModel] = useState("")
  const [newEndpointDescription, setNewEndpointDescription] = useState("")

  // Test Modal state
  const [testModalOpen, setTestModalOpen] = useState(false)
  const [selectedExampleEndpoint, setSelectedExampleEndpoint] = useState<string>("")
  const [testEndpoint, setTestEndpoint] = useState<Endpoint | null>(null)
  const [testQuery, setTestQuery] = useState("What insights can you provide?")
  const [testData, setTestData] = useState('{\n  "example_field": "value"\n}')
  const [testResponse, setTestResponse] = useState<any>(null)
  const [testing, setTesting] = useState(false)

  useEffect(() => {
    fetchEndpoints()
    fetchFineTunedModels()
  }, [])

  const fetchEndpoints = async () => {
    try {
      const res = await fetch(API_BASE + "/endpoints", { credentials: "include" })
      if (res.ok) {
        const data = await res.json()
        setEndpoints(data)
      }
    } catch (error) {
      console.error("Failed to fetch endpoints:", error)
    } finally {
      setLoading(false)
    }
  }

  const fetchFineTunedModels = async () => {
    try {
      const res = await fetch(API_BASE + "/models/finetuned", { credentials: "include" })
      if (res.ok) {
        const data = await res.json()
        setFineTunedModels(data.models || data || [])
      }
    } catch (error) {
      console.error("Failed to fetch models:", error)
    }
  }

  const copyUrl = (id: string, path: string) => {
    navigator.clipboard.writeText(`https://api.schemalabs.ai/v1/query/${path}`)
    setCopiedUrl(id)
    setTimeout(() => setCopiedUrl(null), 2000)
  }

  const createEndpoint = async () => {
    if (!newEndpointName || !newEndpointPath || !selectedFineTunedModel || !selectedLLMModel) return

    setCreating(true)
    try {
      const res = await fetch(API_BASE + "/endpoints/create", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({
          name: newEndpointName,
          path: newEndpointPath,
          fine_tuned_model_id: selectedFineTunedModel,
          llm_model: selectedLLMModel,
          description: newEndpointDescription,
        }),
      })

      if (res.ok) {
        const newEndpoint = await res.json()
        setEndpoints([newEndpoint, ...endpoints])
        addToast({ title: "Success", description: "Endpoint created successfully", variant: "success" })
        resetForm()
        setIsDialogOpen(false)
      } else {
        const error = await res.text()
        addToast({ title: "Error", description: error, variant: "error" })
      }
    } catch (error) {
      console.error("Failed to create endpoint:", error)
    } finally {
      setCreating(false)
    }
  }

  const deleteEndpoint = async (id: string, name: string) => {
    try {
      const res = await fetch(API_BASE + "/endpoints/delete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ id }),
      })

      if (res.ok) {
        setEndpoints(endpoints.filter((e) => e.id !== id))
        addToast({ title: "Deleted", description: "Endpoint deleted successfully", variant: "success" })
      }
    } catch (error) {
      console.error("Failed to delete endpoint:", error)
    }
  }

  const resetForm = () => {
    setNewEndpointName("")
    setNewEndpointPath("")
    setSelectedFineTunedModel("")
    setSelectedLLMModel("")
    setNewEndpointDescription("")
  }

  const getLLMDisplayName = (id: string) => {
    return llmModels.find((m) => m.id === id)?.name || id
  }

  const getFineTunedModelName = (id: string) => {
    return fineTunedModels.find((m) => m.id === id)?.name || id.slice(0, 8) + "..."
  }

  const openTestModal = (endpoint: Endpoint) => {
    setTestEndpoint(endpoint)
    setTestQuery("What insights can you provide?")
    setTestData('{\n  "example_field": "value"\n}')
    setTestResponse(null)
    setTestModalOpen(true)
  }

  const runTest = async () => {
    if (!testEndpoint) return
    setTesting(true)
    setTestResponse(null)
    
    try {
      // Get user's API key first
      const keysRes = await fetch(API_BASE + "/keys", { credentials: "include" })
      if (!keysRes.ok) {
        addToast({ title: "Error", description: "Failed to fetch API keys", variant: "error" })
        setTesting(false)
        return
      }
      const keys = await keysRes.json().catch(() => null)
      console.log("API Keys response:", keys)
      
      const keysList = keys?.keys || keys || []; if (!keysList || keysList.length === 0) {
        addToast({ title: "Error", description: "No API key found. Create one in API Keys page first.", variant: "error" })
        setTesting(false)
        return
      }
      
      const apiKey = keysList[0]?.key
      if (!apiKey) {
        addToast({ title: "Error", description: "API key not found or invalid", variant: "error" })
        setTesting(false)
        return
      }
      
      let parsedData = {}
      try {
        parsedData = JSON.parse(testData || "{}")
      } catch {
        addToast({ title: "Error", description: "Invalid JSON in data field", variant: "error" })
        setTesting(false)
        return
      }
      
      const res = await fetch(`${window.location.origin}/v1/query/${testEndpoint.path}`, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${apiKey}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          query: testQuery,
          data: parsedData
        })
      })
      
      const data = await res.json()
      setTestResponse(data)
      
      if (data.error) {
        addToast({ title: "Error", description: data.error, variant: "error" })
      } else {
        addToast({ title: "Success", description: "Test completed successfully", variant: "success" })
      }
    } catch (error: any) {
      setTestResponse({ error: error.message })
      addToast({ title: "Error", description: error.message, variant: "error" })
    } finally {
      setTesting(false)
    }
  }

  return (
    <div className="flex min-h-screen">
      <Sidebar>
        <div className="p-4 sm:p-6 pt-8 sm:pt-12 space-y-4 sm:space-y-6">
          {/* Header Card */}
          <Card className="bg-card border-border">
            <CardHeader className="p-4 sm:p-6">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div>
                  <CardTitle className="text-foreground text-lg sm:text-xl">Your Endpoints</CardTitle>
                  <CardDescription className="text-sm">
                    Create API endpoints powered by your fine-tuned models
                  </CardDescription>
                </div>
                <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
                  <DialogTrigger asChild>
                    <Button disabled={fineTunedModels.length === 0}>
                      <Plus className="h-4 w-4 mr-2" />
                      Create Endpoint
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="sm:max-w-lg">
                    <DialogHeader>
                      <DialogTitle>Create New Endpoint</DialogTitle>
                      <DialogDescription>
                        Create an API endpoint powered by your fine-tuned model
                      </DialogDescription>
                    </DialogHeader>
                    <div className="space-y-4 py-4">
                      <div className="space-y-2">
                        <Label>Fine-tuned Model <span className="text-destructive">*</span></Label>
                        <Select value={selectedFineTunedModel} onValueChange={setSelectedFineTunedModel}>
                          <SelectTrigger className="bg-secondary">
                            <SelectValue placeholder="Select your trained model" />
                          </SelectTrigger>
                          <SelectContent>
                            {fineTunedModels.map((model) => (
                              <SelectItem key={model.id} value={model.id}>
                                <div className="flex items-center gap-2">
                                  <Brain className="h-4 w-4 text-primary" />
                                  <span>{model.name}</span>
                                </div>
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="space-y-2">
                        <Label>LLM Model <span className="text-destructive">*</span></Label>
                        <Select value={selectedLLMModel} onValueChange={setSelectedLLMModel}>
                          <SelectTrigger className="bg-secondary">
                            <SelectValue placeholder="Select LLM for responses" />
                          </SelectTrigger>
                          <SelectContent>
                            {llmModels.map((model) => (
                              <SelectItem key={model.id} value={model.id}>
                                {model.name}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="space-y-2">
                        <Label>Endpoint Name <span className="text-destructive">*</span></Label>
                        <Input
                          placeholder="e.g., Sales Prediction API"
                          className="bg-secondary"
                          value={newEndpointName}
                          onChange={(e) => setNewEndpointName(e.target.value)}
                        />
                      </div>

                      <div className="space-y-2">
                        <Label>URL Path <span className="text-destructive">*</span></Label>
                        <div className="flex items-center gap-2">
                          <span className="text-sm text-muted-foreground whitespace-nowrap">/v1/query/</span>
                          <Input
                            placeholder="sales-prediction"
                            className="bg-secondary"
                            value={newEndpointPath}
                            onChange={(e) => setNewEndpointPath(e.target.value.toLowerCase().replace(/\s+/g, "-").replace(/[^a-z0-9-]/g, ""))}
                          />
                        </div>
                      </div>

                      <div className="space-y-2">
                        <Label>Description (Optional)</Label>
                        <Input
                          placeholder="What does this endpoint do?"
                          className="bg-secondary"
                          value={newEndpointDescription}
                          onChange={(e) => setNewEndpointDescription(e.target.value)}
                        />
                      </div>
                    </div>
                    <DialogFooter>
                      <DialogClose asChild>
                        <Button variant="outline" onClick={resetForm}>Cancel</Button>
                      </DialogClose>
                      <Button
                        onClick={createEndpoint}
                        disabled={creating || !selectedFineTunedModel || !selectedLLMModel || !newEndpointName || !newEndpointPath}
                      >
                        {creating ? (
                          <>
                            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                            Creating...
                          </>
                        ) : (
                          "Create Endpoint"
                        )}
                      </Button>
                    </DialogFooter>
                  </DialogContent>
                </Dialog>
              </div>
            </CardHeader>
            <CardContent className="p-4 sm:p-6 pt-0">
              {loading ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                </div>
              ) : endpoints.length === 0 ? (
                <div className="text-center py-12">
                  <Brain className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                  <h3 className="text-lg font-medium text-foreground mb-2">No endpoints yet</h3>
                  <p className="text-sm text-muted-foreground mb-4">
                    {fineTunedModels.length === 0 
                      ? "Train a model first, then create an endpoint"
                      : "Create your first API endpoint to get started"
                    }
                  </p>
                </div>
              ) : (
                <div className="space-y-4">
                  {endpoints.map((endpoint) => (
                    <div
                      key={endpoint.id}
                      className={"flex flex-col lg:flex-row lg:items-center justify-between gap-4 rounded-lg border p-4 cursor-pointer transition-all " + (selectedExampleEndpoint === endpoint.id ? "border-primary bg-primary/5" : "border-border bg-secondary/30 hover:border-primary/50")}
                      onClick={() => setSelectedExampleEndpoint(endpoint.id)}
                    >
                      <div className="space-y-2 min-w-0 flex-1">
                        <div className="flex items-center gap-3 flex-wrap">
                          <p className="font-medium text-foreground">{endpoint.name}</p>
                          <Badge variant="outline" className="bg-emerald-500/10 text-emerald-600 border-emerald-500/30">
                            Active
                          </Badge>
                        </div>
                        <div className="flex items-center gap-2 text-sm">
                          <Badge variant="outline" className="font-mono text-xs">POST</Badge>
                          <code className="bg-muted px-2 py-0.5 rounded text-xs sm:text-sm truncate">
                            https://api.schemalabs.ai/v1/query/{endpoint.path}
                          </code>
                        </div>
                        <div className="flex items-center gap-4 text-xs sm:text-sm text-muted-foreground flex-wrap">
                          <span className="flex items-center gap-1">
                            <Brain className="h-3 w-3" />
                            {getFineTunedModelName(endpoint.fine_tuned_model_id)}
                          </span>
                          <span>|</span>
                          <span>LLM: {getLLMDisplayName(endpoint.llm_model)}</span>
                        </div>
                      </div>
                      <div className="flex items-center gap-4">
                        <div className="text-right">
                          <p className="text-sm font-medium text-foreground">{endpoint.calls.toLocaleString()}</p>
                          <p className="text-xs text-muted-foreground">API calls</p>
                        </div>
                        <div className="flex items-center gap-1">
                          <Button variant="outline" size="sm" onClick={() => copyUrl(endpoint.id, endpoint.path)}>
                            {copiedUrl === endpoint.id ? (
                              <Check className="h-4 w-4 mr-1 sm:mr-2 text-emerald-500" />
                            ) : (
                              <Copy className="h-4 w-4 mr-1 sm:mr-2" />
                            )}
                            <span className="hidden sm:inline">{copiedUrl === endpoint.id ? "Copied" : "Copy"}</span>
                          </Button>
                          <Button variant="outline" size="sm" onClick={() => openTestModal(endpoint)}>
                            <Play className="h-4 w-4 sm:mr-2" />
                            <span className="hidden sm:inline">Test</span>
                          </Button>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8"
                            onClick={() => deleteEndpoint(endpoint.id, endpoint.name)}
                          >
                            <Trash2 className="h-4 w-4 text-muted-foreground hover:text-destructive" />
                          </Button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Code Example */}
          <Card className="bg-card border-border">
            <CardHeader className="p-4 sm:p-6">
              <CardTitle className="text-foreground text-base sm:text-lg">Example Request</CardTitle>
              <CardDescription className="text-sm">
                {selectedExampleEndpoint 
                  ? `Code example for "${endpoints.find(e => e.id === selectedExampleEndpoint)?.name}"`
                  : "Click on an endpoint above to see its code example"
                }
              </CardDescription>
            </CardHeader>
            <CardContent className="p-4 sm:p-6 pt-0">
              {(() => {
                const exampleEp = endpoints.find(e => e.id === selectedExampleEndpoint) || endpoints[0]
                if (!exampleEp) return <p className="text-muted-foreground text-sm">Create an endpoint to see example code</p>
                return (
                  <>
                    <div className="rounded-lg bg-muted border border-border p-3 sm:p-4 overflow-x-auto">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-xs text-muted-foreground">cURL</span>
                        <Button variant="ghost" size="sm" className="h-6 px-2" onClick={() => {
                          navigator.clipboard.writeText(`curl -X POST https://api.schemalabs.ai/v1/query/${exampleEp.path} \\\n  -H "Authorization: Bearer YOUR_API_KEY" \\\n  -H "Content-Type: application/json" \\\n  -d '{\n    "query": "What insights can you provide?",\n    "data": {"example_field": "value"}\n  }'`)
                          addToast({ title: "Copied", description: "cURL command copied to clipboard", variant: "success" })
                        }}>
                          <Copy className="h-3 w-3" />
                        </Button>
                      </div>
                      <pre className="text-xs sm:text-sm font-mono text-foreground whitespace-pre-wrap">
{`curl -X POST https://api.schemalabs.ai/v1/query/${exampleEp.path} \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "query": "What insights can you provide?",
    "data": {"example_field": "value"}
  }'`}
                      </pre>
                    </div>
                    <div className="mt-4 rounded-lg bg-muted border border-border p-3 sm:p-4 overflow-x-auto">
                      <p className="text-xs text-muted-foreground mb-2">Expected Response</p>
                      <pre className="text-xs sm:text-sm font-mono text-emerald-600 whitespace-pre-wrap">
{`{
  "answer": "Based on the analysis of your data...",
  "prediction": "Model prediction result",
  "model": "${exampleEp.llm_model}",
  "fine_tuned": "${exampleEp.fine_tuned_model_id.slice(0,8)}...",
  "tokens_used": 156
}`}
                      </pre>
                    </div>
                    <div className="mt-4 flex items-center gap-2 text-xs text-muted-foreground">
                      <Brain className="h-3 w-3" />
                      <span>Fine-tuned: {getFineTunedModelName(exampleEp.fine_tuned_model_id)}</span>
                      <span>|</span>
                      <span>LLM: {getLLMDisplayName(exampleEp.llm_model)}</span>
                    </div>
                  </>
                )
              })()}
            </CardContent>
          </Card>
        </div>
      </Sidebar>

      {/* Test Modal */}
      <Dialog open={testModalOpen} onOpenChange={setTestModalOpen}>
        <DialogContent className="sm:max-w-2xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Play className="h-5 w-5 text-primary" />
              Test Endpoint
            </DialogTitle>
            <DialogDescription>
              Test your endpoint with custom query and data
            </DialogDescription>
          </DialogHeader>

          {testEndpoint && (
            <div className="space-y-4 py-4">
              {/* Endpoint Info */}
              <div className="rounded-lg bg-muted/50 border border-border p-4 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-foreground">{testEndpoint.name}</span>
                  <Badge variant="outline" className="bg-emerald-500/10 text-emerald-600 border-emerald-500/30">
                    Active
                  </Badge>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant="outline" className="font-mono text-xs">POST</Badge>
                  <code className="text-xs text-muted-foreground">/v1/query/{testEndpoint.path}</code>
                </div>
                <div className="flex items-center gap-4 text-xs text-muted-foreground">
                  <span className="flex items-center gap-1">
                    <Brain className="h-3 w-3" />
                    {getFineTunedModelName(testEndpoint.fine_tuned_model_id)}
                  </span>
                  <span>|</span>
                  <span>LLM: {getLLMDisplayName(testEndpoint.llm_model)}</span>
                </div>
              </div>

              {/* Query Input */}
              <div className="space-y-2">
                <Label>Query</Label>
                <Textarea
                  placeholder="Enter your question or analysis request..."
                  className="bg-secondary min-h-[80px]"
                  value={testQuery}
                  onChange={(e) => setTestQuery(e.target.value)}
                />
              </div>

              {/* Data Input */}
              <div className="space-y-2">
                <Label>Data (JSON)</Label>
                <Textarea
                  placeholder='{"field": "value"}'
                  className="bg-secondary font-mono text-sm min-h-[100px]"
                  value={testData}
                  onChange={(e) => setTestData(e.target.value)}
                />
                <p className="text-xs text-muted-foreground">Optional: Add input data for prediction</p>
              </div>

              {/* Run Test Button */}
              <Button onClick={runTest} disabled={testing || !testQuery} className="w-full">
                {testing ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Running Test...
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 mr-2" />
                    Run Test
                  </>
                )}
              </Button>

              {/* Response */}
              {testResponse && (
                <div className="space-y-2">
                  <Label className="flex items-center gap-2">
                    <Code className="h-4 w-4" />
                    Response
                  </Label>
                  <div className="rounded-lg bg-muted border border-border p-4 overflow-x-auto">
                    {testResponse.error ? (
                      <pre className="text-xs sm:text-sm font-mono text-red-500 whitespace-pre-wrap">
                        {JSON.stringify(testResponse, null, 2)}
                      </pre>
                    ) : (
                      <div className="space-y-4">
                        {/* Answer */}
                        <div>
                          <p className="text-xs text-muted-foreground mb-1">Answer</p>
                          <p className="text-sm text-foreground">{testResponse.answer}</p>
                        </div>
                        
                        {/* Prediction if exists */}
                        {testResponse.prediction && (
                          <div>
                            <p className="text-xs text-muted-foreground mb-1">Fine-tuned Model Prediction</p>
                            <pre className="text-sm font-mono text-primary bg-primary/5 p-2 rounded">
                              {typeof testResponse.prediction === 'object' 
                                ? JSON.stringify(testResponse.prediction, null, 2)
                                : testResponse.prediction}
                            </pre>
                          </div>
                        )}

                        {/* Metadata */}
                        <div className="flex items-center gap-4 text-xs text-muted-foreground pt-2 border-t border-border">
                          <span>Model: {testResponse.model}</span>
                          <span>|</span>
                          <span>Tokens: {testResponse.tokens_used}</span>
                        </div>

                        {/* Raw JSON */}
                        <details className="text-xs">
                          <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
                            View Raw JSON
                          </summary>
                          <pre className="mt-2 p-2 bg-background rounded text-xs font-mono overflow-x-auto">
                            {JSON.stringify(testResponse, null, 2)}
                          </pre>
                        </details>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          <DialogFooter>
            <Button variant="outline" onClick={() => setTestModalOpen(false)}>
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

"use client"

import React from "react"

import { useState, useCallback } from "react"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Upload, Globe, Database, FileUp, Check, Cloud, ArrowLeft, X } from "lucide-react"
import { cn } from "@/lib/utils"

interface ConnectModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onConnect?: (connection: ConnectionData) => void
}

interface ConnectionData {
  type: "upload" | "api" | "database" | "cloud"
  subType?: string
  name: string
  config: Record<string, string>
  files?: File[]
}

type DatabaseProvider = "postgresql" | "mysql" | "supabase" | "mongodb" | "databricks" | "snowflake" | "pinecone" | "weaviate" | "chroma" | "lancedb"
type CloudProvider = "google-drive" | "gcs" | "aws-s3"

const databaseProviders: { id: DatabaseProvider; name: string; icon: React.ReactNode; description: string }[] = [
  { id: "postgresql", name: "PostgreSQL", description: "Open-source relational database", icon: <Database className="h-5 w-5 text-[#336791]" /> },
  { id: "mysql", name: "MySQL", description: "Popular relational database", icon: <Database className="h-5 w-5 text-[#4479A1]" /> },
  { id: "supabase", name: "Supabase", description: "Open-source Firebase alternative", icon: <svg viewBox="0 0 24 24" className="h-5 w-5 text-[#3ECF8E]" fill="currentColor"><path d="M21.362 9.354H12V.396a.396.396 0 00-.716-.233L2.203 12.424l-.401.562a1.04 1.04 0 00.836 1.659H12v8.959a.396.396 0 00.716.233l9.081-12.261.401-.562a1.04 1.04 0 00-.836-1.66z" /></svg> },
  { id: "mongodb", name: "MongoDB", description: "NoSQL document database", icon: <Database className="h-5 w-5 text-[#47A248]" /> },
  { id: "databricks", name: "Databricks", description: "Unified analytics platform", icon: <svg viewBox="0 0 24 24" className="h-5 w-5 text-[#FF3621]" fill="currentColor"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" /></svg> },
  { id: "snowflake", name: "Snowflake", description: "Cloud data warehouse", icon: <Database className="h-5 w-5 text-[#29B5E8]" /> },
  { id: "pinecone", name: "Pinecone", description: "Vector database for AI", icon: <Database className="h-5 w-5 text-[#7B61FF]" /> },
  { id: "weaviate", name: "Weaviate", description: "Open-source vector database", icon: <Database className="h-5 w-5 text-[#00C8A8]" /> },
  { id: "chroma", name: "Chroma", description: "Embedding database", icon: <Database className="h-5 w-5 text-[#FFD700]" /> },
  { id: "lancedb", name: "LanceDB", description: "Serverless vector database", icon: <Database className="h-5 w-5 text-[#3B82F6]" /> },
]

const cloudProviders: { id: CloudProvider; name: string; description: string; icon: React.ReactNode }[] = [
  { id: "google-drive", name: "Google Drive", description: "Cloud file storage", icon: <svg viewBox="0 0 24 24" className="h-5 w-5"><path d="M12 2L4 14h4l4-7 4 7h4L12 2z" fill="#4285F4" /><path d="M4 14l4 8h8l4-8H4z" fill="#FBBC04" /></svg> },
  { id: "gcs", name: "Google Cloud Storage", description: "GCP object storage", icon: <Cloud className="h-5 w-5 text-[#4285F4]" /> },
  { id: "aws-s3", name: "AWS S3", description: "Amazon object storage", icon: <Cloud className="h-5 w-5 text-[#FF9900]" /> },
]

export function ConnectModal({ open, onOpenChange, onConnect }: ConnectModalProps) {
  const [selectedProvider, setSelectedProvider] = useState<string | null>(null)
  const [dragActive, setDragActive] = useState(false)
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([])
  
  // Form states
  const [connectionName, setConnectionName] = useState("")
  const [endpoint, setEndpoint] = useState("")
  const [authToken, setAuthToken] = useState("")
  const [apiKey, setApiKey] = useState("")
  const [bucket, setBucket] = useState("")
  const [region, setRegion] = useState("")
  const [projectId, setProjectId] = useState("")
  const [accessKeyId, setAccessKeyId] = useState("")
  const [secretAccessKey, setSecretAccessKey] = useState("")
  const [dbHost, setDbHost] = useState("")
  const [dbPort, setDbPort] = useState("")
  const [dbName, setDbName] = useState("")
  const [dbUser, setDbUser] = useState("")
  const [dbPassword, setDbPassword] = useState("")
  const [apiType, setApiType] = useState<"rest" | "graphql">("rest")

  const resetForm = () => {
    setSelectedProvider(null)
    setUploadedFiles([])
    setConnectionName("")
    setEndpoint("")
    setAuthToken("")
    setApiKey("")
    setBucket("")
    setRegion("")
    setProjectId("")
    setAccessKeyId("")
    setSecretAccessKey("")
    setDbHost("")
    setDbPort("")
    setDbName("")
    setDbUser("")
    setDbPassword("")
    setApiType("rest")
  }

  const handleClose = () => {
    resetForm()
    onOpenChange(false)
  }

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const files = Array.from(e.dataTransfer.files).filter(file => 
        file.name.endsWith('.csv') || 
        file.name.endsWith('.xlsx') || 
        file.name.endsWith('.xls') || 
        file.name.endsWith('.json')
      )
      setUploadedFiles(prev => [...prev, ...files])
    }
  }, [])

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files)
      setUploadedFiles(prev => [...prev, ...files])
    }
  }

  const handleConnect = (type: string) => {
    const connection: ConnectionData = {
      type: type as ConnectionData["type"],
      name: connectionName,
      config: {},
    }
    
    if (type === "upload") {
      connection.files = uploadedFiles
      connection.name = connectionName || "Uploaded Files"
    } else if (type === "api") {
      connection.subType = apiType
      connection.config = { endpoint, authToken }
    } else if (databaseProviders.find(p => p.id === selectedProvider)) {
      connection.type = "database"
      connection.subType = selectedProvider || undefined
      const isRelationalDB = ["postgresql", "mysql", "supabase", "mongodb", "snowflake", "databricks"].includes(selectedProvider || "")
      if (isRelationalDB) {
        connection.config = { host: dbHost, port: dbPort, database: dbName, username: dbUser, password: dbPassword, ssl: selectedProvider === "supabase" }
      } else {
        connection.config = { apiKey, endpoint }
      }
    } else if (cloudProviders.find(p => p.id === selectedProvider)) {
      connection.type = "cloud"
      connection.subType = selectedProvider || undefined
      if (selectedProvider === "google-drive") {
        connection.config = { projectId }
      } else if (selectedProvider === "gcs") {
        connection.config = { projectId, bucket }
      } else if (selectedProvider === "aws-s3") {
        connection.config = { bucket, region, accessKeyId, secretAccessKey }
      }
    }
    
    onConnect?.(connection)
    handleClose()
  }

  const renderProviderForm = () => {
    if (!selectedProvider) return null

    const dbProvider = databaseProviders.find(p => p.id === selectedProvider)
    const cloudProvider = cloudProviders.find(p => p.id === selectedProvider)

    if (dbProvider) {
      const isRelational = ["postgresql", "mysql", "supabase", "mongodb", "snowflake", "databricks"].includes(selectedProvider || "")
      const isVectorDB = ["pinecone", "weaviate", "chroma", "lancedb"].includes(selectedProvider || "")
      return (
        <div className="space-y-3 animate-in fade-in-50 duration-200">
          <div className="flex items-center gap-3 pb-3 border-b border-border">
            <Button variant="ghost" size="icon" onClick={() => setSelectedProvider(null)} className="h-8 w-8">
              <ArrowLeft className="h-4 w-4" />
            </Button>
            <div className="flex items-center gap-2">
              {dbProvider.icon}
              <span className="font-medium text-foreground">{dbProvider.name}</span>
            </div>
          </div>
          
          <div className="space-y-2">
            <Label className="text-foreground text-xs">Connection Name</Label>
            <Input value={connectionName} onChange={(e) => setConnectionName(e.target.value)}
              placeholder={`My ${dbProvider.name}`} className="bg-card border-border text-foreground" />
          </div>

          {isRelational ? (
            <>
              <div className="grid grid-cols-3 gap-2">
                <div className="col-span-2 space-y-1">
                  <Label className="text-foreground text-xs">Host</Label>
                  <Input value={dbHost} onChange={(e) => setDbHost(e.target.value)}
                    placeholder={selectedProvider === "supabase" ? "db.xxxx.supabase.co" : "localhost"} className="bg-card border-border text-foreground" />
                </div>
                <div className="space-y-1">
                  <Label className="text-foreground text-xs">Port</Label>
                  <Input value={dbPort} onChange={(e) => setDbPort(e.target.value)}
                    placeholder={selectedProvider === "mongodb" ? "27017" : selectedProvider === "mysql" ? "3306" : "5432"} className="bg-card border-border text-foreground" />
                </div>
              </div>
              <div className="space-y-1">
                <Label className="text-foreground text-xs">Database</Label>
                <Input value={dbName} onChange={(e) => setDbName(e.target.value)}
                  placeholder={selectedProvider === "supabase" ? "postgres" : "mydb"} className="bg-card border-border text-foreground" />
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div className="space-y-1">
                  <Label className="text-foreground text-xs">Username</Label>
                  <Input value={dbUser} onChange={(e) => setDbUser(e.target.value)}
                    placeholder={selectedProvider === "supabase" ? "postgres" : "user"} className="bg-card border-border text-foreground" />
                </div>
                <div className="space-y-1">
                  <Label className="text-foreground text-xs">Password</Label>
                  <Input type="password" value={dbPassword} onChange={(e) => setDbPassword(e.target.value)}
                    placeholder="••••••••" className="bg-card border-border text-foreground" />
                </div>
              </div>
              {(selectedProvider === "supabase") && (
                <div className="flex items-center gap-2">
                  <input type="checkbox" id="ssl-toggle" checked={true} readOnly className="rounded" />
                  <Label htmlFor="ssl-toggle" className="text-xs text-muted-foreground">SSL Required (Supabase)</Label>
                </div>
              )}
              <Button onClick={() => handleConnect("database")}
                disabled={!connectionName || !dbHost || !dbName}
                className="w-full bg-[#0052CC] hover:bg-[#003D99] text-white">
                Connect {dbProvider.name}
              </Button>
            </>
          ) : (
            <>
              <div className="space-y-1">
                <Label className="text-foreground text-xs">{isVectorDB ? "API Key" : "API Key / Token"}</Label>
                <Input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)}
                  placeholder="Enter your API key" className="bg-card border-border text-foreground" />
              </div>
              <div className="space-y-1">
                <Label className="text-foreground text-xs">Endpoint URL</Label>
                <Input value={endpoint} onChange={(e) => setEndpoint(e.target.value)}
                  placeholder={selectedProvider === "pinecone" ? "https://index-xxx.svc.xxx.pinecone.io" : "https://your-endpoint.com"} className="bg-card border-border text-foreground" />
              </div>
              <Button onClick={() => handleConnect("database")}
                disabled={!connectionName || !apiKey}
                className="w-full bg-[#0052CC] hover:bg-[#003D99] text-white">
                Connect {dbProvider.name}
              </Button>
            </>
          )}
        </div>
      )
    }

    if (cloudProvider) {
      return (
        <div className="space-y-4 animate-in fade-in-50 duration-200">
          <div className="flex items-center gap-3 pb-3 border-b border-border">
            <Button variant="ghost" size="icon" onClick={() => setSelectedProvider(null)} className="h-8 w-8">
              <ArrowLeft className="h-4 w-4" />
            </Button>
            <div className="flex items-center gap-2">
              {cloudProvider.icon}
              <span className="font-medium text-foreground">{cloudProvider.name}</span>
            </div>
          </div>
          
          <div className="space-y-2">
            <Label htmlFor="cloud-name" className="text-foreground">Connection Name</Label>
            <Input
              id="cloud-name"
              value={connectionName}
              onChange={(e) => setConnectionName(e.target.value)}
              placeholder={`My ${cloudProvider.name}`}
              className="bg-card border-border text-foreground"
            />
          </div>
          
          {selectedProvider === "google-drive" && (
            <div className="space-y-2">
              <Label htmlFor="gd-project" className="text-foreground">Google Project ID</Label>
              <Input
                id="gd-project"
                value={projectId}
                onChange={(e) => setProjectId(e.target.value)}
                placeholder="my-project-12345"
                className="bg-card border-border text-foreground"
              />
            </div>
          )}
          
          {selectedProvider === "gcs" && (
            <>
              <div className="space-y-2">
                <Label htmlFor="gcs-project" className="text-foreground">Google Project ID</Label>
                <Input
                  id="gcs-project"
                  value={projectId}
                  onChange={(e) => setProjectId(e.target.value)}
                  placeholder="my-project-12345"
                  className="bg-card border-border text-foreground"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="gcs-bucket" className="text-foreground">Bucket Name</Label>
                <Input
                  id="gcs-bucket"
                  value={bucket}
                  onChange={(e) => setBucket(e.target.value)}
                  placeholder="my-bucket"
                  className="bg-card border-border text-foreground"
                />
              </div>
            </>
          )}
          
          {selectedProvider === "aws-s3" && (
            <>
              <div className="space-y-2">
                <Label htmlFor="s3-bucket" className="text-foreground">Bucket Name</Label>
                <Input
                  id="s3-bucket"
                  value={bucket}
                  onChange={(e) => setBucket(e.target.value)}
                  placeholder="my-bucket"
                  className="bg-card border-border text-foreground"
                />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-2">
                  <Label htmlFor="s3-region" className="text-foreground">Region</Label>
                  <Input
                    id="s3-region"
                    value={region}
                    onChange={(e) => setRegion(e.target.value)}
                    placeholder="us-east-1"
                    className="bg-card border-border text-foreground"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="s3-access" className="text-foreground">Access Key ID</Label>
                  <Input
                    id="s3-access"
                    value={accessKeyId}
                    onChange={(e) => setAccessKeyId(e.target.value)}
                    placeholder="AKIA..."
                    className="bg-card border-border text-foreground"
                  />
                </div>
              </div>
              <div className="space-y-2">
                <Label htmlFor="s3-secret" className="text-foreground">Secret Access Key</Label>
                <Input
                  id="s3-secret"
                  type="password"
                  value={secretAccessKey}
                  onChange={(e) => setSecretAccessKey(e.target.value)}
                  placeholder="Your secret key"
                  className="bg-card border-border text-foreground"
                />
              </div>
            </>
          )}
          
          <Button 
            onClick={() => handleConnect("cloud")}
            disabled={!connectionName || (selectedProvider === "aws-s3" ? !bucket || !accessKeyId : !projectId)}
            className="w-full bg-[#0052CC] hover:bg-[#003D99] text-white"
          >
            Connect {cloudProvider.name}
          </Button>
        </div>
      )
    }

    return null
  }

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="border-border bg-card sm:max-w-[600px] flex flex-col overflow-hidden">
        <DialogHeader>
          <DialogTitle className="text-foreground">Connect Data Source</DialogTitle>
          <DialogDescription className="text-muted-foreground">
            Choose a data source to connect to your workspace
          </DialogDescription>
        </DialogHeader>
        <div className="overflow-y-auto max-h-[60vh]">
        {selectedProvider ? (
          renderProviderForm()
        ) : (
          <Tabs defaultValue="upload" className="w-full">
            <TabsList className="grid w-full grid-cols-4 bg-muted">
              <TabsTrigger value="upload" className="data-[state=active]:bg-card">
                <FileUp className="h-4 w-4 mr-2" />
                Upload
              </TabsTrigger>
              <TabsTrigger value="databases" className="data-[state=active]:bg-card">
                <Database className="h-4 w-4 mr-2" />
                Databases
              </TabsTrigger>
              <TabsTrigger value="cloud" className="data-[state=active]:bg-card">
                <Cloud className="h-4 w-4 mr-2" />
                Cloud
              </TabsTrigger>
              <TabsTrigger value="api" className="data-[state=active]:bg-card">
                <Globe className="h-4 w-4 mr-2" />
                API
              </TabsTrigger>
            </TabsList>

            <TabsContent value="databases" className="mt-4">
              <div className="grid grid-cols-2 gap-3">
                {databaseProviders.map((provider) => (
                  <button
                    key={provider.id}
                    onClick={() => setSelectedProvider(provider.id)}
                    className="flex items-center gap-3 rounded-lg border border-border bg-card p-3 text-left transition-all hover:border-[#0052CC] hover:bg-muted"
                  >
                    <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-muted">
                      {provider.icon}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="font-medium text-foreground text-sm">{provider.name}</p>
                      <p className="text-xs text-muted-foreground truncate">{provider.description}</p>
                    </div>
                  </button>
                ))}
              </div>
            </TabsContent>

            <TabsContent value="cloud" className="mt-4">
              <div className="grid grid-cols-2 gap-3">
                {cloudProviders.map((provider) => (
                  <button
                    key={provider.id}
                    onClick={() => setSelectedProvider(provider.id)}
                    className="flex items-center gap-3 rounded-lg border border-border bg-card p-3 text-left transition-all hover:border-[#0052CC] hover:bg-muted"
                  >
                    <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-muted">
                      {provider.icon}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="font-medium text-foreground text-sm">{provider.name}</p>
                      <p className="text-xs text-muted-foreground truncate">{provider.description}</p>
                    </div>
                  </button>
                ))}
              </div>
            </TabsContent>

            <TabsContent value="api" className="mt-4 space-y-4">
              <div className="flex gap-2 mb-4">
                <Button
                  variant={apiType === "rest" ? "default" : "outline"}
                  onClick={() => setApiType("rest")}
                  className={apiType === "rest" ? "bg-[#0052CC]" : ""}
                >
                  <Globe className="h-4 w-4 mr-2" />
                  REST API
                </Button>
                <Button
                  variant={apiType === "graphql" ? "default" : "outline"}
                  onClick={() => setApiType("graphql")}
                  className={apiType === "graphql" ? "bg-[#E10098]" : ""}
                >
                  <Globe className="h-4 w-4 mr-2" />
                  GraphQL
                </Button>
              </div>

              <div className="space-y-2">
                <Label htmlFor="api-name" className="text-foreground">Connection Name</Label>
                <Input
                  id="api-name"
                  value={connectionName}
                  onChange={(e) => setConnectionName(e.target.value)}
                  placeholder={apiType === "graphql" ? "My GraphQL API" : "My REST API"}
                  className="bg-card border-border text-foreground"
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="api-endpoint" className="text-foreground">{apiType === "graphql" ? "GraphQL Endpoint" : "API Endpoint URL"}</Label>
                <Input
                  id="api-endpoint"
                  value={endpoint}
                  onChange={(e) => setEndpoint(e.target.value)}
                  placeholder={apiType === "graphql" ? "https://api.example.com/graphql" : "https://api.example.com/v1/data"}
                  className="bg-card border-border text-foreground"
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="api-token" className="text-foreground">Auth Token (Optional)</Label>
                <Input
                  id="api-token"
                  type="password"
                  value={authToken}
                  onChange={(e) => setAuthToken(e.target.value)}
                  placeholder="Bearer token or API key"
                  className="bg-card border-border text-foreground"
                />
              </div>
              
              <Button 
                onClick={() => handleConnect("api")}
                disabled={!connectionName || !endpoint}
                className="w-full bg-[#0052CC] hover:bg-[#003D99] text-white"
              >
                Connect {apiType === "graphql" ? "GraphQL" : "REST API"}
              </Button>
            </TabsContent>

            <TabsContent value="upload" className="mt-4 space-y-4">
              <div className="space-y-2">
                <Label htmlFor="upload-name" className="text-foreground">Connection Name</Label>
                <Input
                  id="upload-name"
                  value={connectionName}
                  onChange={(e) => setConnectionName(e.target.value)}
                  placeholder="My Uploaded Data"
                  className="bg-card border-border text-foreground"
                />
              </div>
              
              <div
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                className={cn(
                  "flex flex-col items-center justify-center rounded-xl border-2 border-dashed p-8 transition-colors",
                  dragActive ? "border-[#0052CC] bg-[#0052CC]/5" : "border-border bg-muted/50"
                )}
              >
                <FileUp className="mb-3 h-10 w-10 text-muted-foreground" />
                <p className="mb-1 text-sm font-medium text-foreground">Drag and drop your files here</p>
                <p className="mb-4 text-xs text-muted-foreground">CSV, Excel, JSON, up to 50 MB</p>
                <label htmlFor="file-upload">
                  <input
                    id="file-upload"
                    type="file"
                    multiple
                    accept=".csv,.xlsx,.xls,.json"
                    onChange={handleFileInput}
                    className="hidden"
                  />
                  <Button variant="outline" size="sm" asChild className="bg-transparent">
                    <span>Browse Files</span>
                  </Button>
                </label>
              </div>
              
              {uploadedFiles.length > 0 && (
                <div className="space-y-2">
                  <Label className="text-foreground">Uploaded Files ({uploadedFiles.length})</Label>
                  <div className="max-h-32 space-y-1 overflow-y-auto rounded-lg border border-border bg-muted/50 p-2">
                    {uploadedFiles.map((file, i) => (
                      <div key={i} className="flex items-center gap-2 text-sm">
                        <Check className="h-4 w-4 text-green-500" />
                        <span className="text-foreground">{file.name}</span>
                        <span className="text-xs text-muted-foreground">({(file.size / 1024).toFixed(1)} KB)</span>
                        <button
                          onClick={() => setUploadedFiles(prev => prev.filter((_, idx) => idx !== i))}
                          className="ml-auto text-muted-foreground hover:text-foreground"
                        >
                          <X className="h-3 w-3" />
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              <Button 
                onClick={() => handleConnect("upload")}
                disabled={uploadedFiles.length === 0}
                className="w-full bg-[#0052CC] hover:bg-[#003D99] text-white"
              >
                Upload & Connect
              </Button>
            </TabsContent>
          </Tabs>
        )}
        </div>
      </DialogContent>
    </Dialog>
  )
}

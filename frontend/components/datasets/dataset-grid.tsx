"use client"
import { toast } from "sonner"

import { useState, useMemo, useEffect } from "react"
import { DatasetFilters } from "./dataset-filters"
import { DatasetCard } from "./dataset-card"
import { DatasetSchemaModal } from "./dataset-schema-modal"
import { ConnectModal } from "./connect-modal"
import { api } from "@/lib/api"
import verticals from "@/lib/verticals.json"
import type { Dataset, DataSource, Vertical, Complexity, RowCount } from "@/lib/types"
import { Button } from "@/components/ui/button"
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
} from "@/components/ui/dialog"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { Plus, Search, FolderOpen, ChevronDown, ChevronRight, SlidersHorizontal } from "lucide-react"
import { cn } from "@/lib/utils"
import { Sparkles } from "lucide-react" // Import Sparkles component

interface DataFolder {
  id: string
  name: string
  isOpen: boolean
}

export function DatasetGrid() {
  const [selectedSources, setSelectedSources] = useState<DataSource[]>([])
  const [selectedVerticals, setSelectedVerticals] = useState<Vertical[]>([])
  const [selectedComplexity, setSelectedComplexity] = useState<Complexity[]>([])
  const [selectedRowCount, setSelectedRowCount] = useState<RowCount[]>([])
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null)
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [isConnectModalOpen, setIsConnectModalOpen] = useState(false)
  const [isGenerateModalOpen, setIsGenerateModalOpen] = useState(false)
  const [searchQuery, setSearchQuery] = useState("")
  const [showMobileFilters, setShowMobileFilters] = useState(false)
  const [folders, setFolders] = useState<DataFolder[]>([])
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const loadData = async () => {
      const t0 = performance.now()
      try {
        // Progressive loading - show data as each API completes
        const filesPromise = api.getUploadedFiles()
        const modelsPromise = api.getFineTunedModels()
        const connectionsPromise = api.getConnections().catch(() => ({ connections: [] }))
        
        // Show files as soon as they arrive (fastest path)
        const [filesData, modelsData, connectionsData] = await Promise.all([filesPromise, modelsPromise, connectionsPromise])
        const files = (filesData.files || []).filter((f: any) => !f.is_merged && !f.filename?.includes("_merged_all"))
        const models = modelsData.models || []
        const connections = connectionsData.connections || []
        const fileDatasets: Dataset[] = files.map((f: any) => {
          const model = models.find((m: any) => m.source_file_id === f.file_id)
          const cols = f.columns ? f.columns.split(",") : []
  return {
    id: f.file_id,
    name: f.filename?.replace(/^[a-f0-9-]{36}_/, "") || f.filename,
    description: model ? `Trained: ${model.name}` : "Uploaded file",
    source: (f.source || "upload") as DataSource,
    vertical: (f.vertical || "") as Vertical,
    complexity: cols.length > 25 ? "advanced" : cols.length > 10 ? "medium" : "simple" as Complexity,
    rowCount: (f.row_count || 0) > 10000 ? "large" : (f.row_count || 0) > 1000 ? "medium" : "small" as RowCount,
    rows: f.row_count || 0,
    columns: cols.length,
    schema: cols.map((col: string) => ({ name: col.trim(), type: "string", nullable: true, description: "" })),
    sampleData: [],
    syncStatus: "outdated",
  }
        })
        const connDatasets: Dataset[] = []
        // Show connections immediately with cached/basic info, load table details in background
        for (let ci = 0; ci < connections.length; ci++) {
          const c = connections[ci]
          let totalRows = c.cached_rows || 0
          let totalCols = c.cached_cols || 0
          let schemaDetails: any[] = c.cached_schema ? [{ name: c.cached_schema, type: "string" as const, description: "" }] : []
          connDatasets.push({
            id: c.id,
            name: c.name,
            description: `${c.sub_type || c.type} connection`,
            source: (c.sub_type === "postgresql" ? "postgresql" : c.sub_type === "mongodb" ? "mongodb" : c.sub_type || "api") as DataSource,
            vertical: "" as Vertical,
            complexity: "medium" as Complexity,
            rowCount: totalRows > 10000 ? "large" as RowCount : totalRows > 1000 ? "medium" as RowCount : "small" as RowCount,
            rows: totalRows,
            columns: totalCols,
            schema: schemaDetails,
            sampleData: [],
            syncStatus: "synced",
          })
        }
        setDatasets([...fileDatasets, ...connDatasets])
        setLoading(false)
        
        
        // Background: fetch table details for connections and update
        if (connections.length > 0) {
          Promise.allSettled(
            connections.map((c: any) => api.listTables(c.id).catch(() => ({ table_details: [] })))
          ).then(tableResults => {
            setDatasets(prev => {
              const updated = [...prev]
              for (let ci = 0; ci < connections.length; ci++) {
                const result = tableResults[ci]
                if (result.status === "fulfilled") {
                  const details = result.value.table_details || []
                  const totalRows = details.reduce((sum: number, t: any) => sum + (t.rows || 0), 0)
                  const totalCols = details.length > 0 ? details[0].columns || 0 : 0
                  const schemaDetails = details.map((t: any) => ({ name: t.name, type: "string" as const, description: `${t.rows} rows, ${t.columns} cols` }))
                  const idx = updated.findIndex(d => d.id === connections[ci].id)
                  if (idx >= 0) {
                    updated[idx] = { ...updated[idx], rows: totalRows, columns: totalCols, schema: schemaDetails,
                      rowCount: totalRows > 10000 ? "large" : totalRows > 1000 ? "medium" : "small" as any }
                  }
                }
              }
              return updated
            })
          })
        }
      } catch (e) {
        console.error("Load error:", e)
      }
    }
    loadData().finally(() => setLoading(false))
  }, [])

  // Generate modal form state
  const [generateName, setGenerateName] = useState("")
  const [generateDescription, setGenerateDescription] = useState("")
  const [generateRows, setGenerateRows] = useState("1000")
  const [generateColumns, setGenerateColumns] = useState("10")
  const [generateVertical, setGenerateVertical] = useState("")
  const [generatePrompt, setGeneratePrompt] = useState("")
  const [usePythonScript, setUsePythonScript] = useState(false)
  const [pythonScript, setPythonScript] = useState("")
  const [isGenerating, setIsGenerating] = useState(false)
  const [openFolders, setOpenFolders] = useState<Record<string, boolean>>({})

  // Group datasets by source for display
  // Get unique sources from datasets
  const availableSources = useMemo(() => {
    const uniqueSources = [...new Set(datasets.map(d => d.source))]
    return uniqueSources as DataSource[]
  }, [datasets])

  const availableVerticals = useMemo(() => {
    const unique = [...new Set(datasets.map(d => d.vertical))].filter(v => v && v.trim() !== "")
    return unique as string[]
  }, [datasets])

  const availableComplexity = useMemo(() => {
    const unique = [...new Set(datasets.map(d => d.complexity))]
    return unique as Complexity[]
  }, [datasets])

  const availableRowCount = useMemo(() => {
    const unique = [...new Set(datasets.map(d => d.rowCount))]
    return unique as RowCount[]
  }, [datasets])
  const groupedDatasets = useMemo(() => {
    const filtered = datasets.filter((dataset) => {
      // Search filter
      if (searchQuery) {
        const query = searchQuery.toLowerCase()
        const matchesSearch = 
          dataset.name.toLowerCase().includes(query) ||
          dataset.description.toLowerCase().includes(query) ||
          dataset.source.toLowerCase().includes(query) ||
          dataset.vertical.toLowerCase().includes(query)
        if (!matchesSearch) return false
      }
      
      if (selectedSources.length > 0 && !selectedSources.includes(dataset.source)) return false
      if (selectedVerticals.length > 0 && !selectedVerticals.includes(dataset.vertical)) return false
      if (selectedComplexity.length > 0 && !selectedComplexity.includes(dataset.complexity)) return false
      if (selectedRowCount.length > 0 && !selectedRowCount.includes(dataset.rowCount)) return false
      return true
    })

    // Group by source
    const groups: Record<string, Dataset[]> = {}
    filtered.forEach((dataset) => {
      const source = dataset.source
      if (!groups[source]) {
        groups[source] = []
      }
      groups[source].push(dataset)
    })
    
    return { filtered, groups }
  }, [datasets, searchQuery, selectedSources, selectedVerticals, selectedComplexity, selectedRowCount])

  const activeFiltersCount =
    selectedSources.length + selectedVerticals.length + selectedComplexity.length + selectedRowCount.length

  const handleViewSchema = (dataset: Dataset) => {
    setSelectedDataset(dataset)
    setIsModalOpen(true)
  }

  const [deleteTarget, setDeleteTarget] = useState<Dataset | null>(null)
  
  const handleDelete = (dataset: Dataset) => {
    setDeleteTarget(dataset)
  }

  const confirmDelete = async () => {
    if (!deleteTarget) return
    try {
      const isConnection = deleteTarget.syncStatus === "synced" || deleteTarget.syncStatus === "pending"
      if (isConnection) {
        await api.deleteConnection(deleteTarget.id)
      } else {
        await api.deleteFile(deleteTarget.id)
      }
      toast.success("Dataset deleted successfully")
      setDatasets(prev => prev.filter(d => d.id !== deleteTarget.id))
      setDeleteTarget(null)
    } catch (error) {
      console.error("Delete failed:", error)
      toast.error("Failed to delete dataset")
      setDeleteTarget(null)
    }
  }

  const handleEdit = (dataset: Dataset) => {
    toast.info("Editing...")
  }
  const toggleFolder = (source: string) => {
    setOpenFolders(prev => ({
      ...prev,
      [source]: prev[source] === undefined ? false : !prev[source]
    }))
  }

  const isFolderOpen = (source: string) => {
    return openFolders[source] === undefined ? true : openFolders[source]
  }

  const handleGenerate = async () => {
    const hasValidInput = usePythonScript ? pythonScript.trim() : generatePrompt.trim()
    if (!generateName || !generateVertical || !hasValidInput) return
    setIsGenerating(true)
    try {
      const response = await api.generateDataset({
        name: generateName,
        description: generateDescription,
        rows: parseInt(generateRows) || 1000,
        columns: parseInt(generateColumns) || 10,
        vertical: generateVertical,
        prompt: generatePrompt,
        use_python: usePythonScript,
        python_code: pythonScript
      })
      if (response.status === "success") {
        toast.success(`Dataset "${response.filename}" created with ${response.rows} rows and ${response.columns} columns`, { duration: 4000 })
        setIsGenerateModalOpen(false)
        setGenerateName("")
        setGenerateDescription("")
        setGenerateRows("1000")
        setGenerateColumns("10")
        setGenerateVertical("")
        setGeneratePrompt("")
        setUsePythonScript(false)
        setPythonScript("")
        window.location.reload()
      } else {
        toast.error("Generation failed")
      }
    } catch (error) {
      console.error("Generate error:", error)
      toast.error("Generation failed")
    } finally {
      setIsGenerating(false)
    }
  }

  const sourceLabels: Record<string, string> = {
    "databricks": "Databricks",
    "supabase": "Supabase",
    "api": "API",
    "google-drive": "Google Drive",
    "postgresql": "PostgreSQL",
    "mongodb": "MongoDB",
    "snowflake": "Snowflake",
    "pinecone": "Pinecone",
    "gcs": "Google Cloud Storage",
    "aws-s3": "AWS S3",
    "upload": "Uploaded Files",
  }

  return (
    <>
      <div className="flex flex-col md:flex-row gap-6">
        {/* Mobile filter toggle */}
        <div className={cn("md:block", showMobileFilters ? "block" : "hidden")}>
        <DatasetFilters
          selectedSources={selectedSources}
          selectedVerticals={selectedVerticals}
          selectedComplexity={selectedComplexity}
          selectedRowCount={selectedRowCount}
          onSourceChange={setSelectedSources}
          onVerticalChange={setSelectedVerticals}
          onComplexityChange={setSelectedComplexity}
          onRowCountChange={setSelectedRowCount}
          availableSources={availableSources}
          availableVerticals={availableVerticals}
          availableComplexity={availableComplexity}
          availableRowCount={availableRowCount}
        />
        </div>
        <div className="flex-1">
          {/* Search and Actions Header */}
          <div className="mb-4 flex items-center gap-3">
            <Button
              variant="outline"
              size="icon"
              className="md:hidden bg-transparent shrink-0"
              onClick={() => setShowMobileFilters(!showMobileFilters)}
            >
              <SlidersHorizontal className="h-4 w-4" />
            </Button>            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search datasets by name, description, source..."
                className="bg-background pl-10"
              />
            </div>
            <Button 
              onClick={() => setIsGenerateModalOpen(true)}
              variant="outline"
              className="gap-2 bg-transparent"
            >
              <Plus className="h-4 w-4" />
              Generate
            </Button>
            <Button 
              onClick={() => setIsConnectModalOpen(true)}
              className="gap-2 bg-[#0052CC] text-white hover:bg-[#003D99]"
            >
              <Plus className="h-4 w-4" />
              Connect
            </Button>
          </div>

          {/* Results Header */}
          <div className="mb-4 flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">
                Showing <span className="font-medium text-foreground">{groupedDatasets.filtered.length}</span> of{" "}
                {datasets.length} datasets
              </p>
              {activeFiltersCount > 0 && (
                <p className="mt-1 text-xs text-muted-foreground">
                  {activeFiltersCount} filter{activeFiltersCount !== 1 ? "s" : ""} active
                </p>
              )}
            </div>
          </div>

          {/* Grouped Grid by Source */}
          {groupedDatasets.filtered.length > 0 ? (
            <div className="space-y-6">
              {Object.entries(groupedDatasets.groups).map(([source, datasets]) => (
                <div key={source} className="space-y-3">
                  <button
                    onClick={() => toggleFolder(source)}
                    className="flex items-center gap-2 text-sm font-medium text-foreground hover:text-[#0052CC] transition-colors"
                  >
                    {isFolderOpen(source) ? (
                      <ChevronDown className="h-4 w-4" />
                    ) : (
                      <ChevronRight className="h-4 w-4" />
                    )}
                    <FolderOpen className="h-4 w-4 text-[#0052CC]" />
                    <span>{sourceLabels[source] || source}</span>
                    <span className="ml-1 text-xs text-muted-foreground">({datasets.length})</span>
                  </button>
                  
                  {isFolderOpen(source) && (
                    <div className="grid gap-4 pl-6 md:grid-cols-2 xl:grid-cols-3">
                      {datasets.map((dataset) => (
                        <DatasetCard key={dataset.id} dataset={dataset} onViewSchema={handleViewSchema} onEdit={handleEdit} onDelete={handleDelete} />
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="flex h-64 items-center justify-center rounded-xl border border-border bg-card">
              <div className="text-center">
                <p className="text-muted-foreground">{loading ? "Loading datasets..." : "No datasets match your search"}</p>
                <p className="mt-1 text-sm text-muted-foreground">Try adjusting your search or filter criteria</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {selectedDataset && <DatasetSchemaModal dataset={selectedDataset} open={isModalOpen} onOpenChange={setIsModalOpen} />}
      <ConnectModal 
        open={isConnectModalOpen} 
        onOpenChange={setIsConnectModalOpen}
        onConnect={async (connection) => {
          if (connection.type === "upload" && connection.files) {
            try {
              // Get limits from API
              const limits = await api.getUploadLimits()
              const maxFileSize = (limits.max_file_size_mb || 50) * 1024 * 1024
              
              for (const file of connection.files) {
                if (file.size > maxFileSize) {
                  toast.error(`File "${file.name}" exceeds ${limits.max_file_size_mb || 50}MB limit`)
                  return
                }
              }
              for (const file of connection.files) {
                await api.upload(file, undefined)
              }
              toast.dismiss()
              toast.success("Files uploaded successfully!")
            } catch (error) {
              toast.error("File upload failed")
            }
          } else if (connection.type === "database") {
            try {
              const isRelationalDB = ["postgresql", "mysql", "supabase", "mongodb", "snowflake", "databricks"].includes(connection.subType || "")
              toast.loading("Connecting to database...")
              await api.createConnection({
                name: connection.name,
                type: "database",
                sub_type: connection.subType || "",
                ...(isRelationalDB ? {
                  host: connection.config.host || "",
                  port: parseInt(connection.config.port) || 5432,
                  database: connection.config.database || "",
                  username: connection.config.username || "",
                  password: connection.config.password || "",
                  ssl: connection.config.ssl || false
                } : {
                  api_key: connection.config.apiKey || "",
                  endpoint: connection.config.endpoint || ""
                })
              })
              toast.dismiss()
              toast.success("Database connected!")
            } catch (error) {
              toast.error("Database connection failed")
            }
          } else if (connection.type === "api") {
            try {
              toast.loading("Connecting to API...")
              await api.createConnection({
                name: connection.name,
                type: "api",
                sub_type: "rest",
                endpoint: connection.config.endpoint || "",
                auth_token: connection.config.authToken || ""
              })
              toast.dismiss()
              toast.success("API connected!")
            } catch (error) {
              toast.error("API connection failed")
            }
          } else if (connection.type === "cloud") {
            try {
              toast.loading("Connecting to cloud storage...")
              await api.createConnection({
                name: connection.name,
                type: "cloud",
                sub_type: connection.subType || "",
                ...connection.config
              })
              toast.dismiss()
              toast.success("Connected successfully!")
            } catch (error) {
              toast.error("Cloud connection failed")
            }
          }
          // Reload datasets
          window.location.reload()
        }}
      />

      <Dialog open={isGenerateModalOpen} onOpenChange={setIsGenerateModalOpen}>
        <DialogContent className="border-border bg-card sm:max-w-[500px]">
          <DialogHeader>
            <DialogTitle className="text-foreground">
              Generate Synthetic Data
            </DialogTitle>
            <DialogDescription className="text-muted-foreground">
              Create synthetic datasets for training and testing your models
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="gen-name" className="text-foreground">Dataset Name</Label>
              <Input
                id="gen-name"
                placeholder="e.g., Customer Data Sample"
                value={generateName}
                onChange={(e) => setGenerateName(e.target.value)}
                className="border-border bg-background text-foreground"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="gen-desc" className="text-foreground">Description (Optional)</Label>
              <Input
                id="gen-desc"
                placeholder="Brief description of the dataset"
                value={generateDescription}
                onChange={(e) => setGenerateDescription(e.target.value)}
                className="border-border bg-background text-foreground"
              />
            </div>

            <div className="grid grid-cols-3 gap-4">
              <div className="space-y-2">
                <Label className="text-foreground">Rows</Label>
                <Select value={generateRows} onValueChange={setGenerateRows}>
                  <SelectTrigger className="border-border bg-background text-foreground">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="border-border bg-popover">
                    <SelectItem value="100">100</SelectItem>
                    <SelectItem value="500">500</SelectItem>
                    <SelectItem value="1000">1,000</SelectItem>
                    <SelectItem value="5000">5,000</SelectItem>
                    <SelectItem value="10000">10,000</SelectItem>
                    <SelectItem value="50000">50,000</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label className="text-foreground">Columns</Label>
                <Select value={generateColumns} onValueChange={setGenerateColumns}>
                  <SelectTrigger className="border-border bg-background text-foreground">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="border-border bg-popover">
                    <SelectItem value="5">5</SelectItem>
                    <SelectItem value="10">10</SelectItem>
                    <SelectItem value="15">15</SelectItem>
                    <SelectItem value="20">20</SelectItem>
                    <SelectItem value="30">30</SelectItem>
                    <SelectItem value="50">50</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label className="text-foreground">Vertical</Label>
                <Select value={generateVertical} onValueChange={setGenerateVertical}>
                  <SelectTrigger className="border-border bg-background text-foreground">
                    <SelectValue placeholder="Select vertical..." />
                  </SelectTrigger>
                  <SelectContent className="border-border bg-popover max-h-[300px]">
                    {verticals.map((v) => (
                      <SelectItem key={v.value} value={v.value}>{v.label}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-foreground">
                  {usePythonScript ? "Python Script" : "Data Description"}
                </Label>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-muted-foreground">Use Python</span>
                  <Switch
                    checked={usePythonScript}
                    onCheckedChange={setUsePythonScript}
                  />
                </div>
              </div>
              {usePythonScript ? (
                <Textarea
                  placeholder={`# Python script to generate data
import pandas as pd
import numpy as np

def generate_data(rows, columns):
    data = {
        'id': range(1, rows + 1),
        'value': np.random.randn(rows),
        # Add more columns...
    }
    return pd.DataFrame(data)

# Return the dataframe
df = generate_data(${generateRows}, ${generateColumns})`}
                  value={pythonScript}
                  onChange={(e) => setPythonScript(e.target.value)}
                  className="border-border bg-background text-foreground font-mono text-sm resize-none min-h-[140px]"
                />
              ) : (
                <Textarea
                  placeholder="Describe the data you need. Include column names, data types, and any specific patterns or distributions...

Example: Generate customer data with columns: customer_id, name, email, signup_date, plan_type (free/pro/enterprise), monthly_spend, churn_risk_score (0-1)"
                  value={generatePrompt}
                  onChange={(e) => setGeneratePrompt(e.target.value)}
                  className="border-border bg-background text-foreground resize-none min-h-[120px]"
                />
              )}
            </div>

            <div className="rounded-lg border border-border bg-muted/30 p-3">
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">Estimated cost</span>
                <span className="font-medium text-foreground">
                  ~{(Number.parseInt(generateRows) * Number.parseInt(generateColumns) * 0.0001).toFixed(2)} credits
                </span>
              </div>
            </div>
          </div>

          <DialogFooter>
            <Button 
              variant="outline" 
              onClick={() => setIsGenerateModalOpen(false)} 
              className="bg-transparent"
              disabled={isGenerating}
            >
              Cancel
            </Button>
            <Button
              onClick={handleGenerate}
              disabled={!generateName || !generateVertical || (usePythonScript ? !pythonScript.trim() : !generatePrompt.trim()) || isGenerating}
              className="bg-[#0052CC] text-white hover:bg-[#003D99] gap-2"
            >
              {isGenerating ? (
                <>
                  <div className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                  Generating...
                </>
              ) : (
                "Generate Dataset"
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={!!deleteTarget} onOpenChange={() => setDeleteTarget(null)}>
        <DialogContent className="sm:max-w-md border-border bg-card">
          <DialogHeader>
            <DialogTitle className="text-foreground">Delete Dataset</DialogTitle>
            <DialogDescription className="text-muted-foreground">
              Are you sure you want to delete "{deleteTarget?.name}"? This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="gap-3">
            <Button variant="outline" onClick={() => setDeleteTarget(null)}>Cancel</Button>
            <Button variant="destructive" onClick={confirmDelete}>Delete</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}

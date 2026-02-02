"use client"

import { useState, useMemo } from "react"
import { DatasetFilters } from "./dataset-filters"
import { DatasetCard } from "./dataset-card"
import { DatasetSchemaModal } from "./dataset-schema-modal"
import { ConnectModal } from "./connect-modal"
import { mockDatasets } from "@/lib/mock-data"
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
import { Plus, Search, FolderOpen, ChevronDown, ChevronRight } from "lucide-react"
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
  const [folders, setFolders] = useState<DataFolder[]>([])

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
  const groupedDatasets = useMemo(() => {
    const filtered = mockDatasets.filter((dataset) => {
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
  }, [searchQuery, selectedSources, selectedVerticals, selectedComplexity, selectedRowCount])

  const activeFiltersCount =
    selectedSources.length + selectedVerticals.length + selectedComplexity.length + selectedRowCount.length

  const handleViewSchema = (dataset: Dataset) => {
    setSelectedDataset(dataset)
    setIsModalOpen(true)
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

  const handleGenerate = () => {
    const hasValidInput = usePythonScript ? pythonScript.trim() : generatePrompt.trim()
    if (!generateName || !generateVertical || !hasValidInput) return
    setIsGenerating(true)
    // Simulate generation
    setTimeout(() => {
      setIsGenerating(false)
      setIsGenerateModalOpen(false)
      setGenerateName("")
      setGenerateDescription("")
      setGenerateRows("1000")
      setGenerateColumns("10")
      setGenerateVertical("")
      setGeneratePrompt("")
      setUsePythonScript(false)
      setPythonScript("")
    }, 2000)
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
      <div className="flex gap-6">
        <DatasetFilters
          selectedSources={selectedSources}
          selectedVerticals={selectedVerticals}
          selectedComplexity={selectedComplexity}
          selectedRowCount={selectedRowCount}
          onSourceChange={setSelectedSources}
          onVerticalChange={setSelectedVerticals}
          onComplexityChange={setSelectedComplexity}
          onRowCountChange={setSelectedRowCount}
        />

        <div className="flex-1">
          {/* Search and Actions Header */}
          <div className="mb-4 flex items-center gap-3">
            <div className="relative flex-1">
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
                {mockDatasets.length} datasets
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
                        <DatasetCard key={dataset.id} dataset={dataset} onViewSchema={handleViewSchema} />
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="flex h-64 items-center justify-center rounded-xl border border-border bg-card">
              <div className="text-center">
                <p className="text-muted-foreground">No datasets match your search</p>
                <p className="mt-1 text-sm text-muted-foreground">Try adjusting your search or filter criteria</p>
              </div>
            </div>
          )}
        </div>
      </div>

      <DatasetSchemaModal dataset={selectedDataset} open={isModalOpen} onOpenChange={setIsModalOpen} />
      <ConnectModal 
        open={isConnectModalOpen} 
        onOpenChange={setIsConnectModalOpen}
        onConnect={(connection) => {
          console.log("New connection:", connection)
        }}
      />

      {/* Generate Synthetic Data Modal */}
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
                    <SelectValue placeholder="Select" />
                  </SelectTrigger>
                  <SelectContent className="border-border bg-popover">
                    <SelectItem value="finance">Finance</SelectItem>
                    <SelectItem value="healthcare">Healthcare</SelectItem>
                    <SelectItem value="retail">Retail</SelectItem>
                    <SelectItem value="marketing">Marketing</SelectItem>
                    <SelectItem value="hr">HR / People</SelectItem>
                    <SelectItem value="operations">Operations</SelectItem>
                    <SelectItem value="other">Other</SelectItem>
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
    </>
  )
}

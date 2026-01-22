"use client"

import { useState, useMemo } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Checkbox } from "@/components/ui/checkbox"
import { SourceBadge } from "@/components/datasets/source-badge"
import type { Dataset, SyncMode } from "@/lib/types"
import {
  ArrowRight,
  Table,
  Columns,
  Sparkles,
  Database,
  X,
  Zap,
  Clock,
  RefreshCw,
  ChevronDown,
  ChevronRight,
  Box,
  Search,
} from "lucide-react"
import { mockDatasets } from "@/lib/mock-data"

interface ConfigStepProps {
  selectedDatasets: Dataset[]
  modelName: string
  modelDescription: string
  syncMode: SyncMode
  baseModel: string
  onDatasetToggle: (dataset: Dataset) => void
  onModelNameChange: (name: string) => void
  onModelDescriptionChange: (description: string) => void
  onSyncModeChange: (mode: SyncMode) => void
  onBaseModelChange: (model: string) => void
  onStartTraining: () => void
}

export function ConfigStep({
  selectedDatasets,
  modelName,
  modelDescription,
  syncMode,
  baseModel,
  onDatasetToggle,
  onModelNameChange,
  onModelDescriptionChange,
  onSyncModeChange,
  onBaseModelChange,
  onStartTraining,
}: ConfigStepProps) {
  const [expandedSources, setExpandedSources] = useState<Record<string, boolean>>({
    databricks: true,
    supabase: true,
    api: true,
    "google-drive": true,
  })
  const [searchQuery, setSearchQuery] = useState("")

  const toggleSource = (source: string) => {
    setExpandedSources((prev) => ({ ...prev, [source]: !prev[source] }))
  }

  // Filter datasets by search query
  const filteredDatasets = useMemo(() => {
    if (!searchQuery.trim()) return mockDatasets
    const query = searchQuery.toLowerCase()
    return mockDatasets.filter(
      (ds) =>
        ds.name.toLowerCase().includes(query) ||
        ds.description.toLowerCase().includes(query) ||
        ds.vertical.toLowerCase().includes(query) ||
        ds.source.toLowerCase().includes(query) ||
        ds.schema.some((col) => col.name.toLowerCase().includes(query))
    )
  }, [searchQuery])

  // Group filtered datasets by source
  const datasetsBySource = filteredDatasets.reduce(
    (acc, ds) => {
      if (!acc[ds.source]) acc[ds.source] = []
      acc[ds.source].push(ds)
      return acc
    },
    {} as Record<string, Dataset[]>,
  )

  const sourceOrder = ["databricks", "supabase", "api", "google-drive"]

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <Card className="border-border bg-card flex flex-col h-[700px]">
        <CardHeader className="shrink-0">
          <CardTitle className="flex items-center gap-2 text-base text-foreground">
            <Database className="h-4 w-4 text-[#2684FF]" />
            Connect Data Sources
          </CardTitle>
          <p className="text-sm text-muted-foreground">Select datasets from your connected sources</p>
        </CardHeader>
        <CardContent className="flex-1 overflow-hidden flex flex-col space-y-4">
          {/* Search */}
          <div className="shrink-0 relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search datasets by name, schema, vertical..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9 border-border bg-background text-foreground placeholder:text-muted-foreground"
            />
          </div>

          {/* Selected datasets summary */}
          {selectedDatasets.length > 0 && (
            <div className="shrink-0 rounded-lg border border-[#0052CC]/30 bg-[#0052CC]/10 p-3">
              <p className="text-xs font-medium text-[#2684FF] mb-2">
                {selectedDatasets.length} Dataset{selectedDatasets.length > 1 ? "s" : ""} Selected
              </p>
              <div className="flex flex-wrap gap-2">
                {selectedDatasets.map((ds) => (
                  <div key={ds.id} className="flex items-center gap-2 rounded bg-background/50 px-2 py-1">
                    <SourceBadge source={ds.source} size="sm" />
                    <span className="text-xs text-foreground">{ds.name}</span>
                    <button onClick={() => onDatasetToggle(ds)} className="text-muted-foreground hover:text-foreground">
                      <X className="h-3 w-3" />
                    </button>
                  </div>
                ))}
              </div>
              <div className="mt-2 pt-2 border-t border-border flex gap-4 text-xs text-muted-foreground">
                <span>
                  <span className="font-mono text-foreground">
                    {selectedDatasets.reduce((acc, ds) => acc + ds.rows, 0).toLocaleString()}
                  </span>{" "}
                  total rows
                </span>
                <span>
                  <span className="font-mono text-foreground">
                    {selectedDatasets.reduce((acc, ds) => acc + ds.columns, 0)}
                  </span>{" "}
                  total columns
                </span>
              </div>
            </div>
          )}

          <div className="flex-1 overflow-y-auto space-y-4 pr-2 min-h-0">
            {sourceOrder.map((source) => {
              const datasets = datasetsBySource[source]
              if (!datasets) return null
              const isExpanded = expandedSources[source]
              const selectedCount = selectedDatasets.filter((d) => d.source === source).length

              return (
                <div key={source} className="rounded-lg border border-border overflow-hidden">
                  {/* Source header - clickable to collapse */}
                  <button
                    onClick={() => toggleSource(source)}
                    className="w-full flex items-center justify-between gap-2 px-3 py-2.5 bg-muted/50 hover:bg-muted transition-colors"
                  >
                    <div className="flex items-center gap-2">
                      <SourceBadge source={source as any} />
                      <span className="text-xs text-muted-foreground">({datasets.length} available)</span>
                      {selectedCount > 0 && (
                        <span className="text-xs text-[#2684FF] bg-[#0052CC]/20 px-1.5 py-0.5 rounded">
                          {selectedCount} selected
                        </span>
                      )}
                    </div>
                    {isExpanded ? (
                      <ChevronDown className="h-4 w-4 text-muted-foreground" />
                    ) : (
                      <ChevronRight className="h-4 w-4 text-muted-foreground" />
                    )}
                  </button>

                  {isExpanded && (
                    <div className="p-3 space-y-3 bg-background/50">
                      {datasets.map((ds) => {
                        const isSelected = selectedDatasets.some((s) => s.id === ds.id)
                        return (
                          <div
                            key={ds.id}
                            className={`rounded-lg border transition-all ${
                              isSelected
                                ? "border-[#0052CC] ring-1 ring-[#0052CC]/50"
                                : "border-border hover:border-border/80"
                            }`}
                          >
                            <div className="p-4">
                              {/* Header with checkbox */}
                              <div className="flex items-start gap-3">
                                <Checkbox
                                  checked={isSelected}
                                  onCheckedChange={() => onDatasetToggle(ds)}
                                  className="mt-1 border-border data-[state=checked]:bg-[#0052CC] data-[state=checked]:border-[#0052CC]"
                                />
                                <div className="flex-1 min-w-0">
                                  <div className="flex items-start justify-between">
                                    <div>
                                      <h4 className="font-medium text-foreground">{ds.name}</h4>
                                      <p className="mt-0.5 text-xs text-muted-foreground line-clamp-2">
                                        {ds.description}
                                      </p>
                                    </div>
                                  </div>

                                  {/* Tags */}
                                  <div className="mt-2 flex flex-wrap gap-1.5">
                                    <span className="rounded bg-muted px-1.5 py-0.5 text-xs text-muted-foreground capitalize">
                                      {ds.vertical}
                                    </span>
                                    <span className="rounded bg-muted px-1.5 py-0.5 text-xs text-muted-foreground capitalize">
                                      {ds.complexity}
                                    </span>
                                    {ds.syncStatus && (
                                      <span
                                        className={`rounded px-1.5 py-0.5 text-xs ${
                                          ds.syncStatus === "synced"
                                            ? "bg-emerald-500/10 text-emerald-500"
                                            : ds.syncStatus === "pending"
                                              ? "bg-yellow-500/10 text-yellow-500"
                                              : "bg-orange-500/10 text-orange-500"
                                        }`}
                                      >
                                        {ds.syncStatus === "synced"
                                          ? "Synced"
                                          : ds.syncStatus === "pending"
                                            ? "Pending"
                                            : "Outdated"}
                                      </span>
                                    )}
                                  </div>

                                  {/* Stats */}
                                  <div className="mt-3 flex items-center gap-4 text-xs text-muted-foreground">
                                    <span className="flex items-center gap-1">
                                      <Table className="h-3 w-3" />
                                      <span className="font-mono text-foreground">{ds.rows.toLocaleString()}</span> rows
                                    </span>
                                    <span className="flex items-center gap-1">
                                      <Columns className="h-3 w-3" />
                                      <span className="font-mono text-foreground">{ds.columns}</span> cols
                                    </span>
                                  </div>

                                  {/* Schema Preview */}
                                  <div className="mt-3 rounded bg-muted/50 p-2">
                                    <p className="mb-1.5 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
                                      Schema
                                    </p>
                                    <div className="flex flex-wrap gap-1">
                                      {ds.schema.slice(0, 4).map((col) => (
                                        <span
                                          key={col.name}
                                          className="rounded bg-background px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground"
                                        >
                                          {col.name}
                                        </span>
                                      ))}
                                      {ds.schema.length > 4 && (
                                        <span className="rounded bg-background px-1.5 py-0.5 text-[10px] text-muted-foreground">
                                          +{ds.columns - 4} more
                                        </span>
                                      )}
                                    </div>
                                  </div>
                                </div>
                              </div>
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </CardContent>
      </Card>

      {/* Right side - Model Configuration */}
      <Card className="border-border bg-card">
        <CardHeader>
          <CardTitle className="text-base text-foreground">Model Configuration</CardTitle>
          <p className="text-sm text-muted-foreground">Configure your AI model</p>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Model Name */}
          <div className="space-y-2">
            <Label htmlFor="model-name" className="text-sm text-foreground">
              Model Name
            </Label>
            <Input
              id="model-name"
              value={modelName}
              onChange={(e) => onModelNameChange(e.target.value)}
              placeholder="e.g., Customer Intelligence Model"
              className="border-border bg-background text-foreground placeholder:text-muted-foreground"
            />
          </div>

          {/* Model Description */}
          <div className="space-y-2">
            <Label htmlFor="model-desc" className="text-sm text-foreground">
              Description
            </Label>
            <Textarea
              id="model-desc"
              value={modelDescription}
              onChange={(e) => onModelDescriptionChange(e.target.value)}
              placeholder="Describe what this model will do..."
              className="border-border bg-background text-foreground placeholder:text-muted-foreground min-h-[80px]"
            />
          </div>

          <div className="space-y-2">
            <Label className="text-sm text-foreground">Base Model</Label>
            <p className="text-xs text-muted-foreground">SchemaLabs foundation model for tabular AI</p>
            <div className="mt-2">
              <div className="flex items-center gap-3 rounded-lg border border-[#0052CC] bg-[#0052CC]/10 p-3">
                <Box className="h-5 w-5 text-[#2684FF]" />
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-sm font-medium text-foreground">schema-v0</span>
                    <span className="rounded bg-[#0052CC]/20 px-1.5 py-0.5 text-[10px] font-medium text-[#2684FF]">
                      SELECTED
                    </span>
                  </div>
                  <p className="mt-0.5 text-xs text-muted-foreground">
                    Native tabular data understanding.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Sync Mode - Real-time system */}
          <div className="space-y-2">
            <Label className="text-sm text-foreground">Data Sync Mode</Label>
            <p className="text-xs text-muted-foreground">How should the model stay updated with data changes?</p>
            <div className="grid grid-cols-3 gap-2">
              {[
                { mode: "real-time" as SyncMode, label: "Real-time", icon: Zap, desc: "Auto-track changes" },
                { mode: "scheduled" as SyncMode, label: "Scheduled", icon: Clock, desc: "Periodic sync" },
                { mode: "manual" as SyncMode, label: "Manual", icon: RefreshCw, desc: "On-demand" },
              ].map((option) => (
                <button
                  key={option.mode}
                  type="button"
                  onClick={() => onSyncModeChange(option.mode)}
                  className={`flex flex-col items-center justify-center gap-1 rounded-lg border px-3 py-3 transition-all ${
                    syncMode === option.mode
                      ? "border-[#0052CC] bg-[#0052CC]/10 text-[#2684FF]"
                      : "border-border bg-muted/50 text-muted-foreground hover:border-border/80"
                  }`}
                >
                  <option.icon className="h-4 w-4" />
                  <span className="text-sm font-medium">{option.label}</span>
                  <span className="text-[10px] opacity-60">{option.desc}</span>
                </button>
              ))}
            </div>
            {syncMode === "real-time" && (
              <p className="text-xs text-[#2684FF] bg-[#0052CC]/10 p-2 rounded">
                Model will automatically track and prompt for updates when connected data sources change.
              </p>
            )}
          </div>

          {/* Start Button */}
          <Button
            onClick={onStartTraining}
            disabled={selectedDatasets.length === 0 || !modelName}
            className="w-full gap-2 bg-[#0052CC] text-white hover:bg-[#003D99] disabled:opacity-50"
            size="lg"
          >
            <Sparkles className="h-4 w-4" />
            Build Model
            <ArrowRight className="h-4 w-4" />
          </Button>
        </CardContent>
      </Card>
    </div>
  )
}

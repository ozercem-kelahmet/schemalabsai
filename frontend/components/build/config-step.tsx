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

          {/* Selected datasets compact summary bar */}
          {selectedDatasets.length > 0 && (
            <div className="shrink-0 flex items-center gap-2 rounded-lg border border-[#0052CC]/30 bg-[#0052CC]/10 px-3 py-2">
              <span className="text-xs font-medium text-[#2684FF] whitespace-nowrap">
                {selectedDatasets.length} selected
              </span>
              <div className="h-3 w-px bg-[#0052CC]/30" />
              <span className="text-[11px] text-muted-foreground whitespace-nowrap">
                {selectedDatasets.reduce((acc, ds) => acc + ds.rows, 0).toLocaleString()} rows
              </span>
              <span className="text-[11px] text-muted-foreground whitespace-nowrap">
                {selectedDatasets.reduce((acc, ds) => acc + ds.columns, 0)} cols
              </span>
              <div className="flex-1" />
              <button
                onClick={() => selectedDatasets.forEach((ds) => onDatasetToggle(ds))}
                className="text-[11px] text-[#2684FF] hover:underline whitespace-nowrap"
              >
                Clear all
              </button>
            </div>
          )}

          <div className="flex-1 overflow-y-auto space-y-3 pr-2 min-h-0">
            {sourceOrder.map((source) => {
              const datasets = datasetsBySource[source]
              if (!datasets) return null
              const isExpanded = expandedSources[source]
              const selectedCount = selectedDatasets.filter((d) => d.source === source).length
              const allSourceSelected = datasets.every((ds) => selectedDatasets.some((s) => s.id === ds.id))

              return (
                <div key={source} className="rounded-lg border border-border overflow-hidden">
                  {/* Source header */}
                  <div className="flex items-center gap-2 px-3 py-2 bg-muted/50">
                    <Checkbox
                      checked={allSourceSelected}
                      onCheckedChange={() => {
                        if (allSourceSelected) {
                          datasets.forEach((ds) => {
                            if (selectedDatasets.some((s) => s.id === ds.id)) {
                              onDatasetToggle(ds)
                            }
                          })
                        } else {
                          datasets.forEach((ds) => {
                            if (!selectedDatasets.some((s) => s.id === ds.id)) {
                              onDatasetToggle(ds)
                            }
                          })
                        }
                      }}
                      className="border-border data-[state=checked]:bg-[#0052CC] data-[state=checked]:border-[#0052CC]"
                    />
                    <button
                      onClick={() => toggleSource(source)}
                      className="flex-1 flex items-center justify-between"
                    >
                      <div className="flex items-center gap-2">
                        <SourceBadge source={source as any} />
                        <span className="text-xs text-muted-foreground">({datasets.length})</span>
                        {selectedCount > 0 && (
                          <span className="text-[10px] text-[#2684FF] bg-[#0052CC]/20 px-1.5 py-0.5 rounded">
                            {selectedCount} selected
                          </span>
                        )}
                      </div>
                      {isExpanded ? (
                        <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
                      ) : (
                        <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
                      )}
                    </button>
                  </div>

                  {isExpanded && (
                    <div className="divide-y divide-border">
                      {datasets.map((ds) => {
                        const isSelected = selectedDatasets.some((s) => s.id === ds.id)
                        return (
                          <div
                            key={ds.id}
                            onClick={() => onDatasetToggle(ds)}
                            className={`flex items-center gap-3 px-3 py-2.5 cursor-pointer transition-colors ${
                              isSelected
                                ? "bg-[#0052CC]/5 dark:bg-[#2684FF]/5"
                                : "hover:bg-muted/50"
                            }`}
                          >
                            <Checkbox
                              checked={isSelected}
                              className="pointer-events-none shrink-0 border-border data-[state=checked]:bg-[#0052CC] data-[state=checked]:border-[#0052CC]"
                            />
                            <div className="flex-1 min-w-0">
                              <p className="text-sm font-medium text-foreground truncate">{ds.name}</p>
                              <div className="flex items-center gap-3 mt-0.5 text-[11px] text-muted-foreground">
                                <span className="capitalize">{ds.vertical}</span>
                                <span>{ds.rows.toLocaleString()} rows</span>
                                <span>{ds.columns} cols</span>
                              </div>
                            </div>
                            <div className="flex items-center gap-1.5 shrink-0">
                              {ds.schema.slice(0, 2).map((col) => (
                                <span
                                  key={col.name}
                                  className="rounded bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground hidden lg:inline-block"
                                >
                                  {col.name}
                                </span>
                              ))}
                              {ds.schema.length > 2 && (
                                <span className="text-[10px] text-muted-foreground hidden lg:inline-block">
                                  +{ds.columns - 2}
                                </span>
                              )}
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

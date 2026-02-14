"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Checkbox } from "@/components/ui/checkbox"
import { SourceBadge } from "@/components/datasets/source-badge"
import { api } from "@/lib/api"
import type { Dataset, SyncMode, DataSource, Complexity, RowCount, Vertical } from "@/lib/types"
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
  FileSpreadsheet,
  Loader2,
} from "lucide-react"

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
  scheduleCron?: string
  onScheduleChange?: (cron: string, desc: string) => void
  onConnectionIDsChange?: (ids: string) => void
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
  scheduleCron,
  onScheduleChange,
  onConnectionIDsChange,
  onStartTraining,
}: ConfigStepProps) {
  const [allDatasets, setAllDatasets] = useState<Dataset[]>([])
  const [loading, setLoading] = useState(true)
  const [expandedSources, setExpandedSources] = useState<Record<string, boolean>>({})
  const [searchQuery, setSearchQuery] = useState("")

  useEffect(() => {
    loadData()
  }, [])

  const [scheduleTime, setScheduleTime] = useState("02:00")
  const [scheduleDay, setScheduleDay] = useState("Monday")
  const [scheduleMonthDay, setScheduleMonthDay] = useState("1st")
  const [scheduleType, setScheduleType] = useState<"recurring" | "range">("recurring")
  const [startDate, setStartDate] = useState(() => new Date().toISOString().split('T')[0])
  const [startTime, setStartTime] = useState("02:00")
  const [endDate, setEndDate] = useState(() => { const d = new Date(); d.setMonth(d.getMonth() + 1); return d.toISOString().split('T')[0] })
  const [endTime, setEndTime] = useState("02:00")
  const [intervalValue, setIntervalValue] = useState(24)
  const [intervalUnit, setIntervalUnit] = useState("hours")

  const buildScheduleDesc = () => {
    setTimeout(() => {
      const desc = scheduleType === "range" && startDate && endDate
        ? `Every ${intervalValue} ${intervalUnit} from ${startDate} ${startTime} to ${endDate} ${endTime}`
        : `Every ${intervalValue} ${intervalUnit} starting ${startDate} ${startTime}`
      const cron = `${intervalValue}${intervalUnit.charAt(0)}`
      onScheduleChange?.(cron, desc)
    }, 0)
  }
  const [allConnections, setAllConnections] = useState<any[]>([])
  const [selectedConnIDs, setSelectedConnIDs] = useState<string[]>([])

  const loadData = async () => {
    try {
      const [filesData, modelsData, connectionsData] = await Promise.all([
        api.getUploadedFiles(),
        api.getFineTunedModels(),
        api.getConnections().catch(() => ({ connections: [] }))
      ])
      
      const files = (filesData.files || []).filter((f: any) => !f.is_merged && !f.filename?.includes("_merged_all"))
      const models = modelsData.models || []
      const connections = connectionsData.connections || []
      setAllConnections(connections)
      
      const fileDatasets: Dataset[] = files.map((f: any) => {
        const model = models.find((m: any) => m.source_file_id === f.file_id)
        const cols = f.columns ? f.columns.split(",") : []
        return {
          id: f.file_id,
          name: f.filename?.replace(/^[a-f0-9-]{36}_/, "") || f.filename,
          description: model ? `Trained: ${model.name}` : "Uploaded file",
          source: (f.source || "upload") as DataSource,
          vertical: (f.vertical || "") as Vertical,
          complexity: (cols.length > 25 ? "advanced" : cols.length > 10 ? "medium" : "simple") as Complexity,
          rowCount: ((f.row_count || 0) > 10000 ? "large" : (f.row_count || 0) > 1000 ? "medium" : "small") as RowCount,
          rows: f.row_count || 0,
          columns: cols.length,
          schema: cols.map((col: string) => ({ name: col.trim(), type: "string" as const })),
          sampleData: [],
          syncStatus: "outdated" as const,
        }
      })
      
      const connDatasets: Dataset[] = []
      for (const c of connections) {
        let totalRows = 0
        let totalCols = 0
        let schemaItems: any[] = []
        try {
          const tablesResp = await api.listTables(c.id)
          const tables = tablesResp.table_details || tablesResp.tables || []
          totalRows = tables.reduce((sum: number, t: any) => sum + (t.rows || 0), 0)
          totalCols = tables.length > 0 ? (tables[0].columns || 0) : 0
          schemaItems = tables.map((t: any) => ({ name: t.name, type: "string" as const, description: `${t.rows} rows, ${t.columns} cols` }))
        } catch {}
        connDatasets.push({
          id: c.id,
          name: c.name,
          description: `${c.sub_type || c.type} connection`,
          source: (c.sub_type === "postgresql" ? "postgresql" : c.sub_type === "mongodb" ? "mongodb" : c.sub_type || "api") as DataSource,
          vertical: "" as Vertical,
          complexity: (totalCols > 25 ? "advanced" : totalCols > 10 ? "medium" : "simple") as Complexity,
          rowCount: (totalRows > 10000 ? "large" : totalRows > 1000 ? "medium" : "small") as RowCount,
          rows: totalRows,
          columns: totalCols,
          schema: schemaItems,
          sampleData: [],
          syncStatus: "synced" as const,
        })
      }
      
      const datasets = [...fileDatasets, ...connDatasets]
      setAllDatasets(datasets)
      
      const sources = [...new Set(datasets.map(d => d.source))]
      const expanded: Record<string, boolean> = {}
      sources.forEach(s => { expanded[s] = true })
      setExpandedSources(expanded)
      
    } catch (e) {
      console.error("Load error:", e)
    } finally {
      setLoading(false)
    }
  }

  // Filter datasets
  const getFilteredDatasets = () => {
    if (!searchQuery.trim()) return allDatasets
    const q = searchQuery.toLowerCase()
    return allDatasets.filter(ds => 
      ds.name.toLowerCase().includes(q) ||
      ds.description.toLowerCase().includes(q) ||
      ds.source.toLowerCase().includes(q)
    )
  }
  
  const filteredDatasets = getFilteredDatasets()
  
  // Group by source
  const getGroupedDatasets = () => {
    const groups: Record<string, Dataset[]> = {}
    filteredDatasets.forEach(ds => {
      if (!groups[ds.source]) groups[ds.source] = []
      groups[ds.source].push(ds)
    })
    return groups
  }
  
  const datasetsBySource = getGroupedDatasets()
  const sourceOrder = Object.keys(datasetsBySource)

  const toggleSource = (source: string) => {
    setExpandedSources(prev => ({ ...prev, [source]: !prev[source] }))
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    )
  }

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
              placeholder="Search datasets by name..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9 border-border bg-background text-foreground"
            />
          </div>

          {/* Selected datasets summary */}
          {selectedDatasets.length > 0 && (
            <div className="shrink-0 rounded-lg border border-[#0052CC]/30 bg-[#0052CC]/10 p-3">
              <p className="text-xs font-medium text-[#2684FF] mb-2">
                {selectedDatasets.length} Dataset{selectedDatasets.length > 1 ? "s" : ""} Selected
              </p>
              <div className="flex flex-wrap gap-2">
                {selectedDatasets.map(ds => (
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
                <span><span className="font-mono text-foreground">{selectedDatasets.reduce((a, d) => a + d.rows, 0).toLocaleString()}</span> total rows</span>
                <span><span className="font-mono text-foreground">{selectedDatasets.reduce((a, d) => a + d.columns, 0)}</span> total columns</span>
              </div>
            </div>
          )}

          {/* Dataset list */}
          <div className="flex-1 overflow-y-auto space-y-4 pr-2 min-h-0">
            {filteredDatasets.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <FileSpreadsheet className="h-12 w-12 text-muted-foreground/50 mb-4" />
                <p className="text-muted-foreground">No datasets found</p>
                <p className="text-sm text-muted-foreground/70 mt-1">{searchQuery ? "Try a different search term" : "Upload files to get started"}</p>
              </div>
            ) : (
              sourceOrder.map(source => {
                const sourceDatasets = datasetsBySource[source]
                if (!sourceDatasets?.length) return null
                const isExpanded = expandedSources[source] !== false
                const selectedCount = selectedDatasets.filter(d => d.source === source).length

                return (
                  <div key={source} className="rounded-lg border border-border overflow-hidden">
                    <button
                      onClick={() => toggleSource(source)}
                      className="w-full flex items-center justify-between gap-2 px-3 py-2.5 bg-muted/50 hover:bg-muted transition-colors"
                    >
                      <div className="flex items-center gap-2">
                        <SourceBadge source={source as DataSource} />
                        <span className="text-xs text-muted-foreground">({sourceDatasets.length} available)</span>
                        {selectedCount > 0 && (
                          <span className="text-xs text-[#2684FF] bg-[#0052CC]/20 px-1.5 py-0.5 rounded">{selectedCount} selected</span>
                        )}
                      </div>
                      {isExpanded ? <ChevronDown className="h-4 w-4 text-muted-foreground" /> : <ChevronRight className="h-4 w-4 text-muted-foreground" />}
                    </button>

                    {isExpanded && (
                      <div className="p-3 space-y-3 bg-background/50">
                        {sourceDatasets.map(ds => {
                          const isSelected = selectedDatasets.some(s => s.id === ds.id)
                          return (
                            <div
                              key={ds.id}
                              className={`rounded-lg border transition-all ${isSelected ? "border-[#0052CC] ring-1 ring-[#0052CC]/50" : "border-border hover:border-border/80"}`}
                            >
                              <div className="p-4">
                                <div className="flex items-start gap-3">
                                  <Checkbox
                                    checked={isSelected}
                                    onCheckedChange={() => onDatasetToggle(ds)}
                                    className="mt-1 border-border data-[state=checked]:bg-[#0052CC] data-[state=checked]:border-[#0052CC]"
                                  />
                                  <div className="flex-1 min-w-0">
                                    <h4 className="font-medium text-foreground">{ds.name}</h4>
                                    <p className="mt-0.5 text-xs text-muted-foreground line-clamp-2">{ds.description}</p>
                                    
                                    <div className="mt-2 flex flex-wrap gap-1.5">
                                      {ds.vertical && (
                                        <span className="rounded bg-muted px-1.5 py-0.5 text-xs text-muted-foreground capitalize">{ds.vertical}</span>
                                      )}
                                      <span className="rounded bg-muted px-1.5 py-0.5 text-xs text-muted-foreground capitalize">{ds.complexity}</span>
                                      <span className={`rounded px-1.5 py-0.5 text-xs ${
                                        ds.syncStatus === "synced" ? "bg-emerald-500/10 text-emerald-500" :
                                        ds.syncStatus === "pending" ? "bg-yellow-500/10 text-yellow-500" :
                                        "bg-orange-500/10 text-orange-500"
                                      }`}>
                                        {ds.syncStatus === "synced" ? "Synced" : ds.syncStatus === "pending" ? "Pending" : "Outdated"}
                                      </span>
                                    </div>

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

                                    {ds.schema.length > 0 && (
                                      <div className="mt-3 rounded bg-muted/50 p-2">
                                        <p className="mb-1.5 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">Schema</p>
                                        <div className="flex flex-wrap gap-1">
                                          {ds.schema.slice(0, 4).map((col, i) => (
                                            <span key={i} className="rounded bg-background px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">{col.name}</span>
                                          ))}
                                          {ds.schema.length > 4 && (
                                            <span className="rounded bg-background px-1.5 py-0.5 text-[10px] text-muted-foreground">+{ds.schema.length - 4} more</span>
                                          )}
                                        </div>
                                      </div>
                                    )}
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
              })
            )}
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
          <div className="space-y-2">
            <Label htmlFor="model-name" className="text-sm text-foreground">Model Name</Label>
            <Input
              id="model-name"
              value={modelName}
              onChange={(e) => onModelNameChange(e.target.value)}
              placeholder="e.g., Customer Intelligence Model"
              className="border-border bg-background text-foreground"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="model-desc" className="text-sm text-foreground">Description</Label>
            <Textarea
              id="model-desc"
              value={modelDescription}
              onChange={(e) => onModelDescriptionChange(e.target.value)}
              placeholder="Describe what this model will do..."
              className="border-border bg-background text-foreground min-h-[80px]"
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
                    <span className="rounded bg-[#0052CC]/20 px-1.5 py-0.5 text-[10px] font-medium text-[#2684FF]">SELECTED</span>
                  </div>
                  <p className="mt-0.5 text-xs text-muted-foreground">Native tabular data understanding.</p>
                </div>
              </div>
            </div>
          </div>

          <div className="space-y-2">
            <Label className="text-sm text-foreground">Data Sync Mode</Label>
            <p className="text-xs text-muted-foreground">How should the model stay updated with data changes?</p>
            <div className="grid grid-cols-3 gap-2">
              {[
                { mode: "real-time" as SyncMode, label: "Real-time", icon: Zap, desc: "Auto-sync" },
                { mode: "scheduled" as SyncMode, label: "Scheduled", icon: Clock, desc: "Timed sync" },
                { mode: "manual" as SyncMode, label: "Manual", icon: RefreshCw, desc: "On-demand" },
              ].map(option => (
                <button
                  key={option.mode}
                  type="button"
                  onClick={() => {
                    if (syncMode === option.mode && option.mode !== "manual") {
                      onSyncModeChange("manual")
                      onScheduleChange?.("", "")
                    } else {
                      onSyncModeChange(option.mode)
                      if (option.mode === "manual") onScheduleChange?.("", "")
                    }
                  }}
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

            {/* Schedule Configuration */}
            {syncMode === "scheduled" && (
              <div className="mt-3 space-y-3 rounded-lg border border-[#0052CC]/20 bg-[#0052CC]/5 p-4">
                <div className="flex items-center gap-2">
                  <Clock className="h-4 w-4 text-[#2684FF]" />
                  <Label className="text-xs font-medium text-foreground">Schedule Configuration</Label>
                </div>

                {/* Connection Selection */}
                <div className="space-y-1.5">
                  <Label className="text-[10px] text-muted-foreground uppercase tracking-wider">Data Source Connection</Label>
                  {allConnections.length > 0 ? (
                    <div className="space-y-1">
                      {allConnections.map((c: any) => {
                        const isSelected = selectedConnIDs.includes(c.id)
                        const typeLabel = (c.sub_type || c.type || "").toUpperCase()
                        const typeColor = c.sub_type === "postgresql" ? "text-[#336791]" : c.sub_type === "mongodb" ? "text-[#47A248]" : c.sub_type === "mysql" ? "text-[#4479A1]" : c.sub_type === "aws-s3" || c.sub_type === "aws_s3" ? "text-[#FF9900]" : c.sub_type === "pinecone" ? "text-[#7B61FF]" : "text-muted-foreground"
                        return (
                          <button key={c.id} type="button"
                            onClick={() => {
                              const newIDs = isSelected ? selectedConnIDs.filter((id: string) => id !== c.id) : [...selectedConnIDs, c.id]
                              setSelectedConnIDs(newIDs)
                              onConnectionIDsChange?.(newIDs.join(","))
                            }}
                            className={`flex w-full items-center gap-3 rounded-lg border px-3 py-2 text-left transition-all ${
                              isSelected ? "border-[#0052CC] bg-[#0052CC]/10" : "border-border hover:border-border/80 hover:bg-muted/30"
                            }`}>
                            <div className={`flex h-4 w-4 items-center justify-center rounded border ${isSelected ? "border-[#0052CC] bg-[#0052CC]" : "border-muted-foreground/30"}`}>
                              {isSelected && <span className="text-white text-[8px]">✓</span>}
                            </div>
                            <Database className={`h-3.5 w-3.5 ${typeColor}`} />
                            <div className="flex-1 min-w-0">
                              <div className="text-xs font-medium text-foreground truncate">{c.name}</div>
                              <div className="text-[10px] text-muted-foreground">{c.host ? `${c.host}:${c.port}` : c.endpoint || "Cloud"}</div>
                            </div>
                            <span className={`rounded-full px-1.5 py-0.5 text-[8px] font-semibold ${typeColor} bg-muted/50`}>{typeLabel}</span>
                            <span className={`h-1.5 w-1.5 rounded-full ${c.status === "active" ? "bg-emerald-500" : "bg-red-500"}`} />
                          </button>
                        )
                      })}
                    </div>
                  ) : (
                    <div className="rounded-md bg-amber-500/10 px-3 py-2 text-xs text-amber-600 dark:text-amber-400">
                      No connections found. Add connections from the Database page first.
                    </div>
                  )}
                </div>

                {/* Schedule Mode */}
                <div className="space-y-1.5">
                  <Label className="text-[10px] text-muted-foreground uppercase tracking-wider">Schedule Type</Label>
                  <div className="grid grid-cols-2 gap-2">
                    <button type="button" onClick={() => setScheduleType("recurring")}
                      className={`rounded-md border px-3 py-2 text-xs font-medium transition-all ${scheduleType === "recurring" ? "border-[#0052CC] bg-[#0052CC]/10 text-[#2684FF]" : "border-border text-muted-foreground hover:border-border/80"}`}>
                      🔄 Recurring
                    </button>
                    <button type="button" onClick={() => setScheduleType("range")}
                      className={`rounded-md border px-3 py-2 text-xs font-medium transition-all ${scheduleType === "range" ? "border-[#0052CC] bg-[#0052CC]/10 text-[#2684FF]" : "border-border text-muted-foreground hover:border-border/80"}`}>
                      📅 Date Range
                    </button>
                  </div>
                </div>

                {/* Recurring */}
                {scheduleType === "recurring" && (
                  <div className="space-y-3">
                    <div className="space-y-1.5">
                      <Label className="text-[10px] text-muted-foreground uppercase tracking-wider">Start Date & Time</Label>
                      <div className="flex gap-2">
                        <input type="date" value={startDate} onChange={(e) => { setStartDate(e.target.value); buildScheduleDesc() }}
                          className="rounded-md border border-border bg-background px-3 py-1.5 text-xs text-foreground flex-1" />
                        <input type="time" value={startTime} onChange={(e) => { setStartTime(e.target.value); buildScheduleDesc() }}
                          className="rounded-md border border-border bg-background px-3 py-1.5 text-xs text-foreground w-28" />
                      </div>
                    </div>
                    <div className="space-y-1.5">
                      <Label className="text-[10px] text-muted-foreground uppercase tracking-wider">Repeat Every</Label>
                      <div className="flex items-center gap-2">
                        <input type="number" min={1} max={999} value={intervalValue} onChange={(e) => { setIntervalValue(parseInt(e.target.value) || 1); buildScheduleDesc() }}
                          className="rounded-md border border-border bg-background px-3 py-1.5 text-xs text-foreground w-20" />
                        <select value={intervalUnit} onChange={(e) => { setIntervalUnit(e.target.value); buildScheduleDesc() }}
                          className="rounded-md border border-border bg-background px-3 py-1.5 text-xs text-foreground">
                          <option value="hours">Hour(s)</option>
                          <option value="days">Day(s)</option>
                          <option value="weeks">Week(s)</option>
                          <option value="months">Month(s)</option>
                        </select>
                      </div>
                    </div>
                  </div>
                )}

                {/* Date Range */}
                {scheduleType === "range" && (
                  <div className="space-y-3">
                    <div className="space-y-1.5">
                      <Label className="text-[10px] text-muted-foreground uppercase tracking-wider">Start</Label>
                      <div className="flex gap-2">
                        <input type="date" value={startDate} onChange={(e) => { setStartDate(e.target.value); buildScheduleDesc() }}
                          className="rounded-md border border-border bg-background px-3 py-1.5 text-xs text-foreground flex-1" />
                        <input type="time" value={startTime} onChange={(e) => { setStartTime(e.target.value); buildScheduleDesc() }}
                          className="rounded-md border border-border bg-background px-3 py-1.5 text-xs text-foreground w-28" />
                      </div>
                    </div>
                    <div className="space-y-1.5">
                      <Label className="text-[10px] text-muted-foreground uppercase tracking-wider">End</Label>
                      <div className="flex gap-2">
                        <input type="date" value={endDate} onChange={(e) => { setEndDate(e.target.value); buildScheduleDesc() }}
                          className="rounded-md border border-border bg-background px-3 py-1.5 text-xs text-foreground flex-1" />
                        <input type="time" value={endTime} onChange={(e) => { setEndTime(e.target.value); buildScheduleDesc() }}
                          className="rounded-md border border-border bg-background px-3 py-1.5 text-xs text-foreground w-28" />
                      </div>
                    </div>
                    <div className="space-y-1.5">
                      <Label className="text-[10px] text-muted-foreground uppercase tracking-wider">Retrain Every</Label>
                      <div className="flex items-center gap-2">
                        <input type="number" min={1} max={999} value={intervalValue} onChange={(e) => { setIntervalValue(parseInt(e.target.value) || 1); buildScheduleDesc() }}
                          className="rounded-md border border-border bg-background px-3 py-1.5 text-xs text-foreground w-20" />
                        <select value={intervalUnit} onChange={(e) => { setIntervalUnit(e.target.value); buildScheduleDesc() }}
                          className="rounded-md border border-border bg-background px-3 py-1.5 text-xs text-foreground">
                          <option value="hours">Hour(s)</option>
                          <option value="days">Day(s)</option>
                          <option value="weeks">Week(s)</option>
                          <option value="months">Month(s)</option>
                        </select>
                      </div>
                    </div>
                  </div>
                )}

                {/* Summary */}
                <div className="rounded-md bg-[#0052CC]/10 px-3 py-2 text-xs text-[#2684FF]">
                  <Clock className="inline h-3 w-3 mr-1" />
                  {scheduleType === "range" && startDate && endDate
                    ? `Retrain every ${intervalValue} ${intervalUnit} from ${startDate} ${startTime} to ${endDate} ${endTime}`
                    : startDate
                    ? `Retrain every ${intervalValue} ${intervalUnit} starting ${startDate} ${startTime}`
                    : "Configure schedule above"}
                  {selectedConnIDs.length > 0 && ` · ${selectedConnIDs.length} connection(s)`}
                </div>
              </div>
            )}

            {/* Real-time Sync - Connection Selection */}
            {syncMode === "real-time" && (
              <div className="mt-3 space-y-3 rounded-lg border border-purple-500/20 bg-purple-500/5 p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Zap className="h-4 w-4 text-purple-500" />
                    <Label className="text-xs font-medium text-foreground">Real-time Sync</Label>
                  </div>

                </div>
                <p className="text-xs text-muted-foreground">
                  Select connections to monitor. Model will auto-retrain when data changes are detected (checked every 60 seconds).
                </p>

                {allConnections.length > 0 ? (
                  <div className="space-y-1.5">
                    <Label className="text-[10px] text-muted-foreground uppercase tracking-wider">Available Connections</Label>
                    <div className="space-y-1">
                      {allConnections.map((c: any) => {
                        const isSelected = selectedConnIDs.includes(c.id)
                        const typeLabel = (c.sub_type || c.type || "").toUpperCase()
                        const typeColor = c.sub_type === "postgresql" ? "text-[#336791]" : c.sub_type === "mongodb" ? "text-[#47A248]" : c.sub_type === "mysql" ? "text-[#4479A1]" : c.sub_type === "aws-s3" || c.sub_type === "aws_s3" ? "text-[#FF9900]" : c.sub_type === "pinecone" ? "text-[#7B61FF]" : "text-muted-foreground"
                        return (
                          <button
                            key={c.id}
                            type="button"
                            onClick={() => {
                              const newIDs = isSelected ? selectedConnIDs.filter(id => id !== c.id) : [...selectedConnIDs, c.id]
                              setSelectedConnIDs(newIDs)
                              onConnectionIDsChange?.(newIDs.join(","))
                            }}
                            className={`flex w-full items-center gap-3 rounded-lg border px-3 py-2.5 text-left transition-all ${
                              isSelected ? "border-purple-500 bg-purple-500/10" : "border-border hover:border-border/80 hover:bg-muted/30"
                            }`}
                          >
                            <div className={`flex h-5 w-5 items-center justify-center rounded border ${isSelected ? "border-purple-500 bg-purple-500" : "border-muted-foreground/30"}`}>
                              {isSelected && <span className="text-white text-[10px]">✓</span>}
                            </div>
                            <Database className={`h-4 w-4 ${typeColor}`} />
                            <div className="flex-1 min-w-0">
                              <div className="text-xs font-medium text-foreground truncate">{c.name}</div>
                              <div className="text-[10px] text-muted-foreground">{c.host ? `${c.host}:${c.port}` : c.endpoint || "Cloud"} · {c.database || c.bucket || ""}</div>
                            </div>
                            <span className={`rounded-full px-2 py-0.5 text-[9px] font-semibold ${typeColor} bg-muted/50`}>{typeLabel}</span>
                            <span className={`h-2 w-2 rounded-full ${c.status === "active" ? "bg-emerald-500" : "bg-red-500"}`} />
                          </button>
                        )
                      })}
                    </div>
                    {selectedConnIDs.length > 0 && (
                      <div className="rounded-md bg-purple-500/10 px-3 py-2 text-xs text-purple-600 dark:text-purple-400">
                        <Zap className="inline h-3 w-3 mr-1" />
                        {selectedConnIDs.length} connection(s) selected for real-time monitoring
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="rounded-md bg-amber-500/10 px-3 py-2.5 text-xs text-amber-600 dark:text-amber-400">
                    <span className="font-medium">No connections found.</span> Go to the Database page and use + Connect to add database, cloud storage, or API connections first.
                  </div>
                )}
              </div>
            )}
          </div>

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

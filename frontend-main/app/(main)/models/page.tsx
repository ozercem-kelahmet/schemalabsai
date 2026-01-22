"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Checkbox } from "@/components/ui/checkbox"
import { SourceBadge } from "@/components/datasets/source-badge"
import { mockModels } from "@/lib/mock-data"
import {
  Search,
  Layers,
  Play,
  Trash2,
  Edit3,
  GitCompare,
  MoreHorizontal,
  TrendingUp,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Database,
  RefreshCw,
  Table,
  Columns,
  Check,
  X,
  Copy,
  Calendar,
  Hash,
  Activity,
  Zap,
  Globe,
  ExternalLink,
} from "lucide-react"
import Link from "next/link"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
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
import type { Model } from "@/lib/types"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  Legend,
} from "recharts"

export default function ModelsPage() {
  const [searchQuery, setSearchQuery] = useState("")
  const [selectedModels, setSelectedModels] = useState<string[]>([])
  const [isCompareMode, setIsCompareMode] = useState(false)
  const [models, setModels] = useState<Model[]>(mockModels)
  const [editingModelId, setEditingModelId] = useState<string | null>(null)
  const [editingName, setEditingName] = useState("")
  const [metricsModalOpen, setMetricsModalOpen] = useState(false)
  const [selectedModelForMetrics, setSelectedModelForMetrics] = useState<Model | null>(null)
  const [endpointModalOpen, setEndpointModalOpen] = useState(false)
  const [selectedModelForEndpoint, setSelectedModelForEndpoint] = useState<Model | null>(null)
  const [endpointForm, setEndpointForm] = useState({
    name: "",
    urlPath: "",
    description: "",
  })

  const filteredModels = models.filter((model) => {
    const matchesSearch =
      model.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      model.description.toLowerCase().includes(searchQuery.toLowerCase())
    return matchesSearch
  })

  const toggleModelSelection = (modelId: string) => {
    setSelectedModels((prev) => (prev.includes(modelId) ? prev.filter((id) => id !== modelId) : [...prev, modelId]))
  }

  const selectedModelData = models.filter((m) => selectedModels.includes(m.id))

  const startEditing = (model: Model) => {
    setEditingModelId(model.id)
    setEditingName(model.name)
  }

  const saveModelName = () => {
    if (editingModelId && editingName.trim()) {
      setModels((prev) => prev.map((m) => (m.id === editingModelId ? { ...m, name: editingName.trim() } : m)))
    }
    setEditingModelId(null)
    setEditingName("")
  }

  const cancelEditing = () => {
    setEditingModelId(null)
    setEditingName("")
  }

  const openMetricsModal = (model: Model) => {
    setSelectedModelForMetrics(model)
    setMetricsModalOpen(true)
  }

  const openEndpointModal = (model: Model) => {
    setSelectedModelForEndpoint(model)
    setEndpointForm({
      name: "",
      urlPath: `/v1/models/${model.modelId}/`,
      description: "",
    })
    setEndpointModalOpen(true)
  }

  const createEndpoint = () => {
    if (selectedModelForEndpoint && endpointForm.name && endpointForm.urlPath) {
      const newEndpoint = {
        id: `ep-${Date.now()}`,
        name: endpointForm.name,
        urlPath: endpointForm.urlPath,
        description: endpointForm.description,
        createdAt: new Date(),
        status: "active" as const,
      }
      setModels((prev) =>
        prev.map((m) =>
          m.id === selectedModelForEndpoint.id
            ? { ...m, endpoints: [...(m.endpoints || []), newEndpoint] }
            : m
        )
      )
      setEndpointModalOpen(false)
      setEndpointForm({ name: "", urlPath: "", description: "" })
    }
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

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[#0052CC]/10 dark:bg-[#0052CC]/20">
            <Layers className="h-5 w-5 text-[#0052CC] dark:text-[#2684FF]" />
          </div>
          <div>
            <h1 className="text-xl font-semibold text-foreground">Models</h1>
            <p className="text-sm text-muted-foreground">Manage, evaluate, and compare your AI models</p>
          </div>
        </div>
        <div className="flex gap-2">
          {selectedModels.length >= 2 && (
            <Button onClick={() => setIsCompareMode(true)} className="gap-2 bg-[#0052CC] text-white hover:bg-[#003D99]">
              <GitCompare className="h-4 w-4" />
              Compare ({selectedModels.length})
            </Button>
          )}
          <Link href="/build">
            <Button className="gap-2 bg-[#0052CC] text-white hover:bg-[#003D99]">Build New Model</Button>
          </Link>
        </div>
      </div>

      <div className="flex gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search models..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10 border-border bg-background text-foreground placeholder:text-muted-foreground"
          />
        </div>
      </div>

      {/* Compare Mode Panel */}
      {isCompareMode && selectedModels.length >= 2 && (
        <Card className="border-[#0052CC]/30 bg-[#0052CC]/5 dark:bg-[#0052CC]/10">
          <CardHeader className="pb-4">
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2 text-base text-foreground">
                <GitCompare className="h-4 w-4 text-[#0052CC] dark:text-[#2684FF]" />
                Model Comparison
              </CardTitle>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsCompareMode(false)}
                className="text-muted-foreground hover:text-foreground"
              >
                Close
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border">
                    <th className="py-2 pr-4 text-left text-sm font-medium text-muted-foreground">Metric</th>
                    {selectedModelData.map((model) => (
                      <th key={model.id} className="py-2 px-4 text-left text-sm font-medium text-foreground">
                        {model.name}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-border/50">
                    <td className="py-3 pr-4 text-sm text-muted-foreground">Base Model</td>
                    {selectedModelData.map((model) => (
                      <td key={model.id} className="py-3 px-4 font-mono text-sm text-[#0052CC] dark:text-[#2684FF]">
                        {model.baseModel}
                      </td>
                    ))}
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-3 pr-4 text-sm text-muted-foreground">Accuracy</td>
                    {selectedModelData.map((model) => (
                      <td key={model.id} className="py-3 px-4 font-mono text-sm text-emerald-500">
                        {(model.accuracy * 100).toFixed(1)}%
                      </td>
                    ))}
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-3 pr-4 text-sm text-muted-foreground">API Requests</td>
                    {selectedModelData.map((model) => (
                      <td key={model.id} className="py-3 px-4 font-mono text-sm text-foreground">
                        {formatNumber(model.apiRequests)}
                      </td>
                    ))}
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-3 pr-4 text-sm text-muted-foreground">Tokens Used</td>
                    {selectedModelData.map((model) => (
                      <td key={model.id} className="py-3 px-4 font-mono text-sm text-foreground">
                        {formatNumber(model.tokensUsed)}
                      </td>
                    ))}
                  </tr>
                  <tr>
                    <td className="py-3 pr-4 text-sm text-muted-foreground">Data Sources</td>
                    {selectedModelData.map((model) => (
                      <td key={model.id} className="py-3 px-4">
                        <div className="flex gap-1">
                          {model.datasets.map((ds) => (
                            <SourceBadge key={ds.datasetId} source={ds.source} size="sm" />
                          ))}
                        </div>
                      </td>
                    ))}
                  </tr>
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Models Grid */}
      {filteredModels.length > 0 ? (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {filteredModels.map((model) => (
            <Card
              key={model.id}
              className={`group border-border bg-card transition-all hover:border-[#0052CC]/30 hover:bg-accent/50 ${
                selectedModels.includes(model.id) ? "border-[#0052CC] ring-1 ring-[#0052CC]/50" : ""
              }`}
            >
              <CardContent className="p-5">
                {/* Header with checkbox */}
                <div className="flex items-start justify-between gap-3">
                  <div className="flex items-start gap-3">
                    <Checkbox
                      checked={selectedModels.includes(model.id)}
                      onCheckedChange={() => toggleModelSelection(model.id)}
                      className="mt-1 border-border data-[state=checked]:bg-[#0052CC] data-[state=checked]:border-[#0052CC]"
                    />
                    <div className="flex-1 min-w-0">
                      {editingModelId === model.id ? (
                        <div className="flex items-center gap-2">
                          <Input
                            value={editingName}
                            onChange={(e) => setEditingName(e.target.value)}
                            className="h-7 text-sm border-border bg-background"
                            autoFocus
                            onKeyDown={(e) => {
                              if (e.key === "Enter") saveModelName()
                              if (e.key === "Escape") cancelEditing()
                            }}
                          />
                          <Button size="icon" variant="ghost" className="h-6 w-6 shrink-0" onClick={saveModelName}>
                            <Check className="h-3 w-3 text-emerald-500" />
                          </Button>
                          <Button size="icon" variant="ghost" className="h-6 w-6 shrink-0" onClick={cancelEditing}>
                            <X className="h-3 w-3 text-red-500" />
                          </Button>
                        </div>
                      ) : (
                        <h3 className="font-medium text-foreground group-hover:text-[#0052CC] dark:group-hover:text-[#2684FF] transition-colors truncate">
                          {model.name}
                        </h3>
                      )}
                      <p className="mt-0.5 text-sm text-muted-foreground line-clamp-1">{model.description}</p>
                    </div>
                  </div>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8 shrink-0 text-muted-foreground hover:text-foreground"
                      >
                        <MoreHorizontal className="h-4 w-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end" className="border-border bg-popover">
                      <DropdownMenuItem
                        className="text-foreground focus:bg-accent focus:text-accent-foreground"
                        onClick={() => startEditing(model)}
                      >
                        <Edit3 className="mr-2 h-4 w-4" />
                        Rename Model
                      </DropdownMenuItem>
                      <DropdownMenuItem className="text-foreground focus:bg-accent focus:text-accent-foreground">
                        <Play className="mr-2 h-4 w-4" />
                        Open in Playground
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        className="text-foreground focus:bg-accent focus:text-accent-foreground"
                        onClick={() => openEndpointModal(model)}
                      >
                        <Globe className="mr-2 h-4 w-4" />
                        Create Endpoint
                      </DropdownMenuItem>
                      <DropdownMenuItem className="text-foreground focus:bg-accent focus:text-accent-foreground">
                        <RefreshCw className="mr-2 h-4 w-4" />
                        Sync Data Sources
                      </DropdownMenuItem>
                      <DropdownMenuSeparator className="bg-border" />
                      <DropdownMenuItem className="text-red-500 focus:bg-red-500/10 focus:text-red-500">
                        <Trash2 className="mr-2 h-4 w-4" />
                        Delete
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>

                {/* Model ID and Created Date */}
                <div className="mt-3 flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
                  <div className="flex items-center gap-1.5">
                    <Hash className="h-3 w-3" />
                    <span className="font-mono">{model.modelId}</span>
                    <button
                      onClick={() => copyToClipboard(model.modelId)}
                      className="hover:text-foreground transition-colors"
                    >
                      <Copy className="h-3 w-3" />
                    </button>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <Calendar className="h-3 w-3" />
                    <span>{formatDate(model.createdAt)}</span>
                  </div>
                </div>

                {/* Tags */}
                <div className="mt-3 flex flex-wrap gap-2">
                  <span className="rounded-md bg-[#0052CC]/10 px-2 py-1 font-mono text-xs text-[#0052CC] dark:text-[#2684FF]">
                    {model.baseModel}
                  </span>
                  {model.status === "completed" ? (
                    <span className="flex items-center gap-1.5 rounded-md bg-emerald-500/10 px-2 py-1 text-xs text-emerald-500">
                      <CheckCircle2 className="h-3 w-3" />
                      Active
                    </span>
                  ) : model.status === "training" ? (
                    <span className="flex items-center gap-1.5 rounded-md bg-[#0052CC]/10 px-2 py-1 text-xs text-[#0052CC] dark:text-[#2684FF]">
                      <div className="h-2 w-2 animate-pulse rounded-full bg-[#0052CC] dark:bg-[#2684FF]" />
                      Training
                    </span>
                  ) : model.status === "needs-update" ? (
                    <span className="flex items-center gap-1.5 rounded-md bg-amber-500/10 px-2 py-1 text-xs text-amber-500">
                      <AlertTriangle className="h-3 w-3" />
                      Needs Update
                    </span>
                  ) : (
                    <span className="flex items-center gap-1.5 rounded-md bg-red-500/10 px-2 py-1 text-xs text-red-500">
                      <XCircle className="h-3 w-3" />
                      Failed
                    </span>
                  )}
                  <span className="rounded-md bg-muted px-2 py-1 text-xs capitalize text-muted-foreground">
                    {model.syncMode}
                  </span>
                </div>

                {/* Usage Stats */}
                <div className="mt-4 grid grid-cols-3 gap-3 rounded-lg bg-muted/50 p-3">
                  <div className="text-center">
                    <div className="flex items-center justify-center gap-1 text-muted-foreground">
                      <TrendingUp className="h-3 w-3" />
                      <span className="text-[10px] uppercase tracking-wider">Accuracy</span>
                    </div>
                    <p className="mt-1 font-mono text-sm font-medium text-emerald-500">
                      {(model.accuracy * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div className="text-center border-x border-border">
                    <div className="flex items-center justify-center gap-1 text-muted-foreground">
                      <Activity className="h-3 w-3" />
                      <span className="text-[10px] uppercase tracking-wider">Requests</span>
                    </div>
                    <p className="mt-1 font-mono text-sm font-medium text-foreground">
                      {formatNumber(model.apiRequests)}
                    </p>
                  </div>
                  <div className="text-center">
                    <div className="flex items-center justify-center gap-1 text-muted-foreground">
                      <Zap className="h-3 w-3" />
                      <span className="text-[10px] uppercase tracking-wider">Tokens</span>
                    </div>
                    <p className="mt-1 font-mono text-sm font-medium text-foreground">
                      {formatNumber(model.tokensUsed)}
                    </p>
                  </div>
                </div>

                {/* Connected Datasets */}
                <div className="mt-4 rounded-lg bg-muted/50 p-3">
                  <p className="mb-2 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
                    Connected Data Sources ({model.datasets.length})
                  </p>
                  <div className="space-y-2">
                    {model.datasets.slice(0, 2).map((ds) => (
                      <div
                        key={ds.datasetId}
                        className="flex items-center justify-between rounded bg-background px-2 py-1.5"
                      >
                        <div className="flex items-center gap-2 min-w-0">
                          <SourceBadge source={ds.source} size="sm" />
                          <span className="text-xs text-foreground truncate">{ds.datasetName}</span>
                        </div>
                        <div className="flex items-center gap-2 text-[10px] text-muted-foreground shrink-0">
                          <span className="flex items-center gap-1">
                            <Table className="h-2.5 w-2.5" />
                            {ds.rows.toLocaleString()}
                          </span>
                          <span className="flex items-center gap-1">
                            <Columns className="h-2.5 w-2.5" />
                            {ds.columns}
                          </span>
                          {ds.syncStatus === "outdated" && <AlertTriangle className="h-3 w-3 text-amber-500" />}
                        </div>
                      </div>
                    ))}
                    {model.datasets.length > 2 && (
                      <p className="text-xs text-muted-foreground text-center">
                        +{model.datasets.length - 2} more sources
                      </p>
                    )}
                  </div>
                </div>

                {/* Endpoints */}
                {model.endpoints && model.endpoints.length > 0 && (
                  <div className="mt-3">
                    <p className="mb-1.5 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
                      Endpoints ({model.endpoints.length})
                    </p>
                    <div className="flex flex-wrap gap-1.5">
                      {model.endpoints.map((ep) => (
                        <span
                          key={ep.id}
                          className="inline-flex items-center gap-1 rounded bg-emerald-500/10 px-2 py-1 text-xs text-emerald-500"
                        >
                          <Globe className="h-3 w-3" />
                          {ep.name}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Pending Updates Alert */}
                {model.pendingUpdates && model.pendingUpdates.length > 0 && (
                  <div className="mt-4 rounded-lg border border-amber-500/30 bg-amber-500/10 p-3">
                    <p className="text-xs font-medium text-amber-500">Data Update Available</p>
                    <p className="mt-1 text-xs text-muted-foreground">{model.pendingUpdates[0].message}</p>
                    <Button
                      size="sm"
                      className="mt-2 h-7 gap-1.5 bg-amber-500/20 text-amber-500 hover:bg-amber-500/30 text-xs"
                    >
                      <RefreshCw className="h-3 w-3" />
                      Update Model
                    </Button>
                  </div>
                )}

                {/* Actions */}
                <div className="mt-4 flex gap-2">
                  <Link href={`/playground?model=${model.id}`} className="flex-1">
                    <Button
                      variant="outline"
                      size="sm"
                      className="w-full gap-2 border-border text-foreground hover:bg-accent hover:text-accent-foreground bg-transparent"
                    >
                      <Play className="h-3.5 w-3.5" />
                      Playground
                    </Button>
                  </Link>
                  <Button
                    variant="outline"
                    size="sm"
                    className="flex-1 gap-2 border-border text-foreground hover:bg-accent hover:text-accent-foreground bg-transparent"
                    onClick={() => openMetricsModal(model)}
                  >
                    <TrendingUp className="h-3.5 w-3.5" />
                    Metrics
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <Card className="border-border bg-card">
          <CardContent className="flex h-64 flex-col items-center justify-center text-center">
            <Layers className="h-12 w-12 text-muted-foreground" />
            <p className="mt-4 text-muted-foreground">No models found</p>
            <p className="mt-1 text-sm text-muted-foreground">
              {searchQuery ? "Try adjusting your search" : "Build your first model to get started"}
            </p>
            <Link href="/build" className="mt-4">
              <Button className="gap-2 bg-[#0052CC] text-white hover:bg-[#003D99]">Build New Model</Button>
            </Link>
          </CardContent>
        </Card>
      )}

      {/* Metrics Modal */}
      <Dialog open={metricsModalOpen} onOpenChange={setMetricsModalOpen}>
        <DialogContent className="max-w-2xl border-border bg-card">
          <DialogHeader>
            <DialogTitle className="text-foreground">
              {selectedModelForMetrics?.name} - Training Metrics
            </DialogTitle>
            <DialogDescription className="text-muted-foreground">
              Loss and accuracy curves from model training
            </DialogDescription>
          </DialogHeader>
          {selectedModelForMetrics?.trainingMetricsHistory && (
            <div className="space-y-6">
              {/* Summary Stats */}
              <div className="grid grid-cols-4 gap-4">
                <div className="rounded-lg bg-muted/50 p-3 text-center">
                  <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Final Accuracy</p>
                  <p className="mt-1 font-mono text-lg font-semibold text-emerald-500">
                    {(selectedModelForMetrics.accuracy * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="rounded-lg bg-muted/50 p-3 text-center">
                  <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Final Loss</p>
                  <p className="mt-1 font-mono text-lg font-semibold text-foreground">
                    {selectedModelForMetrics.trainingMetricsHistory[
                      selectedModelForMetrics.trainingMetricsHistory.length - 1
                    ]?.loss.toFixed(3)}
                  </p>
                </div>
                <div className="rounded-lg bg-muted/50 p-3 text-center">
                  <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Epochs</p>
                  <p className="mt-1 font-mono text-lg font-semibold text-foreground">
                    {selectedModelForMetrics.trainingMetricsHistory.length * 4}
                  </p>
                </div>
                <div className="rounded-lg bg-muted/50 p-3 text-center">
                  <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Data Points</p>
                  <p className="mt-1 font-mono text-lg font-semibold text-foreground">
                    {selectedModelForMetrics.datasets.reduce((acc, ds) => acc + ds.rows, 0).toLocaleString()}
                  </p>
                </div>
              </div>

              {/* Charts */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="mb-2 text-sm font-medium text-foreground">Loss Curve</p>
                  <div className="h-48 w-full rounded-lg bg-muted/30 p-2">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={selectedModelForMetrics.trainingMetricsHistory}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                        <XAxis
                          dataKey="epoch"
                          tick={{ fill: "rgb(156, 163, 175)", fontSize: 10 }}
                          axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
                        />
                        <YAxis
                          tick={{ fill: "rgb(156, 163, 175)", fontSize: 10 }}
                          axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
                        />
                        <Tooltip
                          cursor={false}
                          contentStyle={{
                            backgroundColor: "hsl(var(--card))",
                            border: "1px solid hsl(var(--border))",
                            borderRadius: "8px",
                          }}
                          labelStyle={{ color: "hsl(var(--foreground))" }}
                          itemStyle={{ color: "hsl(var(--foreground))" }}
                        />
                        <Line
                          type="monotone"
                          dataKey="loss"
                          stroke="#ef4444"
                          strokeWidth={2}
                          dot={{ fill: "#ef4444", r: 3 }}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
                <div>
                  <p className="mb-2 text-sm font-medium text-foreground">Accuracy Curve</p>
                  <div className="h-48 w-full rounded-lg bg-muted/30 p-2">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={selectedModelForMetrics.trainingMetricsHistory}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                        <XAxis
                          dataKey="epoch"
                          tick={{ fill: "rgb(156, 163, 175)", fontSize: 10 }}
                          axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
                        />
                        <YAxis
                          tick={{ fill: "rgb(156, 163, 175)", fontSize: 10 }}
                          axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
                          domain={[0, 1]}
                        />
                        <Tooltip
                          cursor={false}
                          contentStyle={{
                            backgroundColor: "hsl(var(--card))",
                            border: "1px solid hsl(var(--border))",
                            borderRadius: "8px",
                          }}
                          labelStyle={{ color: "hsl(var(--foreground))" }}
                          itemStyle={{ color: "hsl(var(--foreground))" }}
                          formatter={(value: number) => [(value * 100).toFixed(1) + "%", "Accuracy"]}
                        />
                        <Line
                          type="monotone"
                          dataKey="accuracy"
                          stroke="#10b981"
                          strokeWidth={2}
                          dot={{ fill: "#10b981", r: 3 }}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>

              {/* Model Info */}
              <div className="rounded-lg border border-border p-4">
                <p className="mb-2 text-sm font-medium text-foreground">Model Information</p>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Model ID</span>
                    <span className="font-mono text-foreground">{selectedModelForMetrics.modelId}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Base Model</span>
                    <span className="font-mono text-[#0052CC] dark:text-[#2684FF]">{selectedModelForMetrics.baseModel}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Created</span>
                    <span className="text-foreground">{formatDate(selectedModelForMetrics.createdAt)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Sync Mode</span>
                    <span className="capitalize text-foreground">{selectedModelForMetrics.syncMode}</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Create Endpoint Modal */}
      <Dialog open={endpointModalOpen} onOpenChange={setEndpointModalOpen}>
        <DialogContent className="border-border bg-card">
          <DialogHeader>
            <DialogTitle className="text-foreground">Create API Endpoint</DialogTitle>
            <DialogDescription className="text-muted-foreground">
              Create a new API endpoint for {selectedModelForEndpoint?.name}
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="endpoint-name" className="text-foreground">
                Endpoint Name
              </Label>
              <Input
                id="endpoint-name"
                placeholder="e.g., Prediction Endpoint"
                value={endpointForm.name}
                onChange={(e) => setEndpointForm({ ...endpointForm, name: e.target.value })}
                className="border-border bg-background text-foreground"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="url-path" className="text-foreground">
                URL Path
              </Label>
              <Input
                id="url-path"
                placeholder="/v1/models/your-model/predict"
                value={endpointForm.urlPath}
                onChange={(e) => setEndpointForm({ ...endpointForm, urlPath: e.target.value })}
                className="border-border bg-background text-foreground font-mono"
              />
              <p className="text-xs text-muted-foreground">
                Full URL: https://api.schemalabs.ai{endpointForm.urlPath}
              </p>
            </div>
            <div className="space-y-2">
              <Label htmlFor="description" className="text-foreground">
                Description
              </Label>
              <Textarea
                id="description"
                placeholder="Describe what this endpoint does..."
                value={endpointForm.description}
                onChange={(e) => setEndpointForm({ ...endpointForm, description: e.target.value })}
                className="border-border bg-background text-foreground resize-none"
                rows={3}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setEndpointModalOpen(false)} className="bg-transparent">
              Cancel
            </Button>
            <Button
              onClick={createEndpoint}
              disabled={!endpointForm.name || !endpointForm.urlPath}
              className="bg-[#0052CC] text-white hover:bg-[#003D99]"
            >
              Create Endpoint
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

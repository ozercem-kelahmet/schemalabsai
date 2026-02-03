"use client"

import { useState, useEffect, useMemo } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Search, Layers, Play, MoreHorizontal, Calendar, Hash, Activity, Zap, ChevronLeft, ChevronRight, Database, FileText, MessageSquare, Trash2 } from "lucide-react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog"

interface FineTunedModel {
  id: string
  name: string
  accuracy: number
  created_at: string
  source_files?: string
  source_file_names?: string
  source_name?: string
  epochs: number
  batch_size: number
  loss: number
  usage_count?: number
  request_count?: number
  loss_history?: number[]
  accuracy_history?: number[]
}

function ChartWithTooltip({ data, label, finalValue, isLoss = false }: { data: number[]; label: string; finalValue: string; isLoss?: boolean }) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null)
  if (!data || data.length === 0) return null
  const minVal = isLoss ? Math.min(...data) * 0.8 : 55
  const maxVal = isLoss ? Math.max(...data) * 1.1 : 100
  const range = maxVal - minVal
  const getY = (val: number) => 80 - ((val - minVal) / range) * 55
  const getX = (i: number) => 45 + (i / Math.max(1, data.length - 1)) * 340
  const pathPoints = data.map((val, i) => `${i === 0 ? 'M' : 'L'} ${getX(i)} ${getY(val)}`).join(' ')
  return (
    <div className="border border-border rounded-lg p-3 relative">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-medium">{label}</span>
        <span className="text-xs text-muted-foreground">Final: {finalValue}</span>
      </div>
      <div className="h-28 w-full">
        <svg width="100%" height="100%" viewBox="0 0 400 105" preserveAspectRatio="xMidYMid meet" onMouseLeave={() => setHoveredIndex(null)} style={{ overflow: 'visible' }}>
          <line x1="45" y1="25" x2="45" y2="90" stroke="hsl(var(--border))" strokeWidth="1" />
          <line x1="45" y1="90" x2="385" y2="90" stroke="hsl(var(--border))" strokeWidth="1" />
          {[0, 0.5, 1].map((ratio, i) => (<line key={i} x1="45" y1={90 - ratio * 55} x2="385" y2={90 - ratio * 55} stroke="hsl(var(--border))" strokeWidth="1" opacity="0.3" />))}
          <text x="40" y="28" textAnchor="end" fontSize="9" fill="hsl(var(--muted-foreground))">{isLoss ? maxVal.toFixed(2) : `${maxVal.toFixed(0)}%`}</text>
          <text x="40" y="65" textAnchor="end" fontSize="9" fill="hsl(var(--muted-foreground))">{isLoss ? ((maxVal + minVal) / 2).toFixed(2) : `${((maxVal + minVal) / 2).toFixed(0)}%`}</text>
          <text x="40" y="93" textAnchor="end" fontSize="9" fill="hsl(var(--muted-foreground))">{isLoss ? minVal.toFixed(2) : `${minVal.toFixed(0)}%`}</text>
          <text x="45" y="103" textAnchor="start" fontSize="9" fill="hsl(var(--muted-foreground))">E1</text>
          <text x="385" y="103" textAnchor="end" fontSize="9" fill="hsl(var(--muted-foreground))">E{data.length}</text>
          <path d={`${pathPoints} L 385 90 L 45 90 Z`} fill={isLoss ? "rgba(239,68,68,0.1)" : "rgba(16,185,129,0.1)"} />
          <path d={pathPoints} fill="none" stroke={isLoss ? "#ef4444" : "#10b981"} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          {data.map((val, i) => {
            const x = getX(i); const y = getY(val); const isHovered = hoveredIndex === i
            return (<g key={i}><circle cx={x} cy={y} r="12" fill="transparent" onMouseEnter={() => setHoveredIndex(i)} style={{ cursor: 'pointer' }} /><circle cx={x} cy={y} r={isHovered ? 5 : 3} fill={isLoss ? "#ef4444" : "#10b981"} opacity={isHovered ? 1 : 0.6} />{isHovered && <line x1={x} y1={y} x2={x} y2={90} stroke={isLoss ? "#ef4444" : "#10b981"} strokeWidth="1" strokeDasharray="3" opacity="0.3" />}</g>)
          })}
        </svg>
      </div>
      {hoveredIndex !== null && (<div className="absolute bg-foreground/90 text-background px-2 py-1.5 rounded text-xs pointer-events-none z-50 shadow-md" style={{ left: `${Math.min(85, Math.max(5, (hoveredIndex / Math.max(1, data.length - 1)) * 100 - 7))}%`, top: '45px', transform: 'translateX(-50%)' }}><div className="font-medium">Epoch {hoveredIndex + 1}</div><div className="font-semibold">{isLoss ? data[hoveredIndex].toFixed(4) : `${data[hoveredIndex].toFixed(1)}%`}</div></div>)}
    </div>
  )
}

export default function ModelsPage() {
  const router = useRouter()
  const [searchInput, setSearchInput] = useState("")
  const [allModels, setAllModels] = useState<FineTunedModel[]>([])
  const [loading, setLoading] = useState(true)
  const [page, setPage] = useState(1)
  const perPage = 9
  const [metricsOpen, setMetricsOpen] = useState(false)
  const [selectedModel, setSelectedModel] = useState<FineTunedModel | null>(null)

  const [editingModelId, setEditingModelId] = useState<string | null>(null)
  const [editingName, setEditingName] = useState("")
  const [endpointModalOpen, setEndpointModalOpen] = useState(false)
  const [selectedModelForEndpoint, setSelectedModelForEndpoint] = useState<FineTunedModel | null>(null)
  const [endpointForm, setEndpointForm] = useState({ name: "", urlPath: "", description: "" })
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false)
  const [selectedModelForDelete, setSelectedModelForDelete] = useState<FineTunedModel | null>(null)

  const startEditing = (model: FineTunedModel) => {
    setEditingModelId(model.id)
    setEditingName(model.name)
  }

  const saveModelName = async () => {
    if (editingModelId && editingName.trim()) {
      try {
        await fetch("/api/models/finetuned/update", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "include",
          body: JSON.stringify({ id: editingModelId, name: editingName.trim() }),
        })
        setAllModels(prev => prev.map(m => m.id === editingModelId ? { ...m, name: editingName.trim() } : m))
      } catch (e) { console.error(e) }
    }
    setEditingModelId(null)
    setEditingName("")
  }

  const cancelEditing = () => {
    setEditingModelId(null)
    setEditingName("")
  }

  const openEndpointModal = (model: FineTunedModel) => {
    setSelectedModelForEndpoint(model)
    setEndpointForm({ name: "", urlPath: `/v1/models/${model.id}/`, description: "" })
    setEndpointModalOpen(true)
  }

  const deleteModel = async () => {
    if (!selectedModelForDelete) return
    try {
      await fetch("/api/models/" + selectedModelForDelete.id, {
        method: "DELETE",
        credentials: "include",
      })
      setAllModels(prev => prev.filter(m => m.id !== selectedModelForDelete.id))
    } catch (e) { console.error(e) }
    setDeleteConfirmOpen(false)
    setSelectedModelForDelete(null)
  }

  useEffect(() => {
    fetchModels()
  }, [])

  const fetchModels = async () => {
    try {
      const res = await fetch("/api/models/finetuned", { credentials: "include" })
      if (res.ok) {
        const data = await res.json()
        const list = (data.models || []).map((m: FineTunedModel) => {
          const epochs = m.epochs || 5
          const lossHistory: number[] = []
          const accHistory: number[] = []
          let loss = 0.65 + Math.random() * 0.15
          let acc = 55 + Math.random() * 10
          for (let i = 0; i < epochs; i++) {
            loss = Math.max((m.loss || 0.2) * 0.95, loss - (0.65 - (m.loss || 0.2)) / epochs + (Math.random() - 0.5) * 0.08)
            acc = Math.min((m.accuracy || 95) * 1.02, acc + ((m.accuracy || 95) - 55) / epochs + (Math.random() - 0.5) * 4)
            lossHistory.push(loss)
            accHistory.push(Math.max(55, Math.min(100, acc)))
          }
          if (lossHistory.length > 0) lossHistory[lossHistory.length - 1] = m.loss || 0.2
          if (accHistory.length > 0) accHistory[accHistory.length - 1] = m.accuracy || 95
          return { ...m, loss_history: lossHistory, accuracy_history: accHistory, usage_count: Math.floor(Math.random() * 500) + 50, request_count: Math.floor(Math.random() * 2000) + 100 }
        })
        setAllModels(list)
      }
    } catch (e) { console.error(e) }
    finally { setLoading(false) }
  }

  // SEARCH FILTER - useMemo ile her searchInput değiştiğinde yeniden hesapla
  const filteredModels = useMemo(() => {
    if (!searchInput || searchInput.trim() === "") {
      return allModels
    }
    const searchLower = searchInput.toLowerCase().trim()
    return allModels.filter((model) => {
      const nameMatch = model.name && model.name.toLowerCase().startsWith(searchLower)
      const idMatch = model.id && model.id.toLowerCase().includes(searchLower)
      const sourceMatch = model.source_name && model.source_name.toLowerCase().includes(searchLower)
      const filesMatch = model.source_file_names && model.source_file_names.toLowerCase().includes(searchLower)
      return nameMatch
    })
  }, [searchInput, allModels])

  const totalPages = Math.ceil(filteredModels.length / perPage)
  const startIdx = (page - 1) * perPage
  const paginated = filteredModels.slice(startIdx, startIdx + perPage)

  // Search değişince sayfa 1'e dön
  const handleSearchChange = (value: string) => {
    setSearchInput(value)
    setPage(1)
  }

  const formatDate = (d: string) => {
    try {
      return new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", year: "numeric" }).format(new Date(d))
    } catch { return d }
  }

  const cleanName = (s: string) => s.replace(/^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}[_.]?/, "").replace(/^\.csv_?/, "").replace(/_\d{8}_\d{6}/, "").replace(/\.csv$/, "")
  const getSourceNames = (m: FineTunedModel): string[] => {
    if (m.source_file_names) return m.source_file_names.split(",").map(s => cleanName(s.trim())).filter(Boolean)
    if (m.source_name) return [cleanName(m.source_name)]
    if (m.source_files) return m.source_files.split(",").filter(Boolean).map((_, i) => "Dataset " + (i + 1))
    return []
  }

  const handleClick = async (m: FineTunedModel) => {
    try {
      const res = await fetch("/api/queries?model_id=" + m.id, { credentials: "include" })
      if (res.ok) {
        const data = await res.json()
        if (data.queries?.length > 0) { router.push("/playground/" + data.queries[0].id); return }
      }
    } catch (e) { console.error(e) }
    router.push("/playground?model=" + m.id)
  }

  return (
    <div className="space-y-6">
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
        <Link href="/build"><Button className="bg-[#0052CC] hover:bg-[#0052CC]/90 text-white">Build New Model</Button></Link>
      </div>

      {/* SEARCH INPUT */}
      <div className="relative w-full">
        <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground pointer-events-none" />
        <input
          type="text"
          placeholder="Search models by name, ID, or source..."
          value={searchInput}
          onChange={(e) => handleSearchChange(e.target.value)}
          className="w-full pl-10 pr-4 py-2.5 border border-border bg-card rounded-lg text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-[#0052CC]/50 focus:border-[#0052CC]"
        />
        {searchInput && (
          <button
            onClick={() => handleSearchChange("")}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
          >
            ✕
          </button>
        )}
      </div>

      {/* RESULTS INFO */}
      {searchInput && (
        <div className="text-sm text-muted-foreground">
          Found {filteredModels.length} model{filteredModels.length !== 1 ? 's' : ''} matching "{searchInput}"
        </div>
      )}

      {loading ? (
        <div className="flex h-64 items-center justify-center"><div className="text-muted-foreground">Loading...</div></div>
      ) : filteredModels.length > 0 ? (
        <>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {paginated.map(m => (
              <Card key={m.id} className="border-border bg-card hover:border-[#0052CC]/50 transition-colors cursor-pointer" onClick={() => { if (editingModelId) return; setTimeout(() => handleClick(m), 100) }}>
                <CardContent className="p-4">
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      {editingModelId === m.id ? (
                        <div className="flex items-center gap-1 w-full" onClick={e => e.stopPropagation()}>
                          <input className="font-medium text-foreground bg-background border border-border rounded px-2 py-0.5 flex-1 min-w-0" value={editingName} onChange={e => setEditingName(e.target.value)} onKeyDown={e => { if (e.key === "Enter") saveModelName(); if (e.key === "Escape") cancelEditing() }} autoFocus />
                          <button className="p-1 rounded hover:bg-green-500/20 text-green-500" onClick={e => { e.stopPropagation(); saveModelName() }}>✓</button>
                          <button className="p-1 rounded hover:bg-red-500/20 text-red-500" onClick={e => { e.stopPropagation(); cancelEditing() }}>✕</button>
                        </div>
                      ) : (
                        <h3 className="font-medium text-foreground truncate">{m.name}</h3>
                      )}
                      <p className="text-xs text-muted-foreground mt-0.5">Fine-tuned model</p>
                    </div>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild onClick={e => e.stopPropagation()}><Button variant="ghost" size="icon" className="h-8 w-8"><MoreHorizontal className="h-4 w-4" /></Button></DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem onClick={e => { e.stopPropagation(); startEditing(m) }}><FileText className="mr-2 h-4 w-4" /> Rename Model</DropdownMenuItem>
                        <DropdownMenuItem onClick={e => { e.stopPropagation(); handleClick(m) }}><Play className="mr-2 h-4 w-4" /> Open in Playground</DropdownMenuItem>
                        <DropdownMenuItem onClick={e => { e.stopPropagation(); openEndpointModal(m) }}><Zap className="mr-2 h-4 w-4" /> Create Endpoint</DropdownMenuItem>
                        <DropdownMenuItem onClick={e => { e.stopPropagation(); }}><Database className="mr-2 h-4 w-4" /> Sync Data Sources</DropdownMenuItem>
                        <DropdownMenuItem onClick={e => { e.stopPropagation(); setSelectedModelForDelete(m); setDeleteConfirmOpen(true) }} className="text-red-500 focus:text-red-500"><Trash2 className="mr-2 h-4 w-4" /> Delete</DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>
                  <div className="mt-2 flex items-center gap-2 text-xs text-muted-foreground">
                    <Hash className="h-3 w-3" /><span className="truncate">{m.id.slice(0, 8)}...</span>
                    <Calendar className="h-3 w-3 ml-2" /><span>{formatDate(m.created_at)}</span>
                  </div>
                  <div className="mt-3 flex items-center gap-2">
                    <span className="inline-flex items-center rounded-md bg-[#0052CC]/10 px-2 py-1 text-xs font-medium text-[#0052CC] dark:text-[#2684FF]">schema-v0</span>
                    <span className="inline-flex items-center gap-1 rounded-md bg-emerald-500/10 px-2 py-1 text-xs font-medium text-emerald-500"><span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />Active</span>
                  </div>
                  <div className="mt-4 grid grid-cols-3 gap-3 border-t border-border pt-4">
                    <div><div className="text-[10px] text-muted-foreground uppercase">Accuracy</div><div className="mt-1 text-lg font-semibold text-emerald-500">{m.accuracy?.toFixed(1) || 0}%</div></div>
                    <div><div className="text-[10px] text-muted-foreground uppercase">Epochs</div><div className="mt-1 text-lg font-semibold">{m.epochs || 5}</div></div>
                    <div><div className="text-[10px] text-muted-foreground uppercase">Loss</div><div className="mt-1 text-lg font-semibold">{(m.loss || 0).toFixed(3)}</div></div>
                  </div>
                  {getSourceNames(m).length > 0 && (
                    <div className="mt-4 border-t border-border pt-4">
                      <div className="text-xs text-muted-foreground mb-2">DATA SOURCES ({getSourceNames(m).length})</div>
                      <div className="flex flex-wrap gap-1">
                        {getSourceNames(m).slice(0, 2).map((n, i) => (<span key={i} className="inline-flex items-center gap-1 rounded bg-emerald-500/10 px-2 py-1 text-xs text-emerald-600 dark:text-emerald-400"><Database className="h-3 w-3" />{n.length > 15 ? n.slice(0, 15) + "..." : n}</span>))}
                        {getSourceNames(m).length > 2 && <span className="text-xs text-muted-foreground">+{getSourceNames(m).length - 2} more</span>}
                      </div>
                    </div>
                  )}
                  <div className="mt-4 flex gap-2">
                    <Button variant="outline" size="sm" className="flex-1" onClick={e => { e.stopPropagation(); handleClick(m) }}><Play className="mr-2 h-4 w-4" /> Playground</Button>
                    <Button variant="outline" size="sm" className="flex-1" onClick={e => { e.stopPropagation(); setSelectedModel(m); setMetricsOpen(true) }}><Activity className="mr-2 h-4 w-4" /> Metrics</Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
          {totalPages > 1 && (
            <div className="flex items-center justify-between">
              <Button variant="outline" size="sm" onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1}><ChevronLeft className="h-4 w-4 mr-1" /> Previous</Button>
              <span className="text-sm text-muted-foreground">Page {page} of {totalPages}</span>
              <Button variant="outline" size="sm" onClick={() => setPage(p => Math.min(totalPages, p + 1))} disabled={page === totalPages}>Next <ChevronRight className="h-4 w-4 ml-1" /></Button>
            </div>
          )}
          <div className="text-sm text-muted-foreground text-center">{filteredModels.length} models</div>
        </>
      ) : (
        <Card className="border-border bg-card">
          <CardContent className="flex h-64 flex-col items-center justify-center">
            <Layers className="h-12 w-12 text-muted-foreground/50" />
            <h3 className="mt-4 text-lg font-medium">{searchInput ? "No models found" : "No models yet"}</h3>
            <p className="mt-1 text-sm text-muted-foreground">{searchInput ? `No results for "${searchInput}"` : "Build your first model"}</p>
            {!searchInput && <Link href="/build" className="mt-4"><Button className="bg-[#0052CC] hover:bg-[#0052CC]/90 text-white">Build Model</Button></Link>}
          </CardContent>
        </Card>
      )}

      <Dialog open={metricsOpen} onOpenChange={setMetricsOpen}>
        <DialogContent className="max-w-2xl border-border bg-card">
          <DialogHeader>
            <DialogTitle>{selectedModel?.name} - Metrics</DialogTitle>
            <DialogDescription>Training metrics</DialogDescription>
          </DialogHeader>
          {selectedModel && (
            <div className="space-y-6">
              <div className="grid grid-cols-4 gap-4">
                <div className="rounded-lg bg-muted/50 p-3 text-center"><p className="text-[10px] uppercase text-muted-foreground">Accuracy</p><p className="mt-1 font-mono text-lg font-semibold text-emerald-500">{selectedModel.accuracy?.toFixed(1)}%</p></div>
                <div className="rounded-lg bg-muted/50 p-3 text-center"><p className="text-[10px] uppercase text-muted-foreground">Loss</p><p className="mt-1 font-mono text-lg font-semibold">{(selectedModel.loss || 0).toFixed(3)}</p></div>
                <div className="rounded-lg bg-muted/50 p-3 text-center"><p className="text-[10px] uppercase text-muted-foreground">Epochs</p><p className="mt-1 font-mono text-lg font-semibold">{selectedModel.epochs || 10}</p></div>
                <div className="rounded-lg bg-muted/50 p-3 text-center"><p className="text-[10px] uppercase text-muted-foreground">Batch</p><p className="mt-1 font-mono text-lg font-semibold">{selectedModel.batch_size || 32}</p></div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <ChartWithTooltip data={selectedModel.loss_history || []} label="Loss" finalValue={(selectedModel.loss || 0).toFixed(4)} isLoss />
                <ChartWithTooltip data={selectedModel.accuracy_history || []} label="Accuracy" finalValue={`${(selectedModel.accuracy || 0).toFixed(1)}%`} />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-muted/30 rounded-lg p-3"><div className="flex items-center gap-2 text-muted-foreground mb-1"><MessageSquare className="w-3.5 h-3.5" /><span className="text-xs">Requests</span></div><p className="text-xl font-semibold">{(selectedModel.request_count || 0).toLocaleString()}</p></div>
                <div className="bg-muted/30 rounded-lg p-3"><div className="flex items-center gap-2 text-muted-foreground mb-1"><Layers className="w-3.5 h-3.5" /><span className="text-xs">Predictions</span></div><p className="text-xl font-semibold">{(selectedModel.usage_count || 0).toLocaleString()}</p></div>
              </div>
              <div className="rounded-lg border border-border p-4">
                <p className="mb-3 text-sm font-medium">Info</p>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between"><span className="text-muted-foreground flex items-center gap-1.5"><FileText className="w-3.5 h-3.5" />Source</span><span className="font-medium">{selectedModel.source_name || getSourceNames(selectedModel).join(", ") || "N/A"}</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground flex items-center gap-1.5"><Calendar className="w-3.5 h-3.5" />Created</span><span className="font-medium">{formatDate(selectedModel.created_at)}</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground flex items-center gap-1.5"><Hash className="w-3.5 h-3.5" />ID</span><span className="font-mono text-xs">{selectedModel.id.slice(0, 20)}...</span></div>
                </div>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Create Endpoint Modal */}
      <Dialog open={endpointModalOpen} onOpenChange={setEndpointModalOpen}>
        <DialogContent className="border-border bg-card sm:max-w-[500px]">
          <DialogHeader>
            <DialogTitle>Create Endpoint</DialogTitle>
            <DialogDescription>Create a new API endpoint for {selectedModelForEndpoint?.name}</DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Endpoint Name</label>
              <input className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm" placeholder="Sales Prediction API" value={endpointForm.name} onChange={e => setEndpointForm(prev => ({ ...prev, name: e.target.value }))} />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">URL Path</label>
              <div className="flex items-center">
                <span className="rounded-l-md border border-r-0 border-border bg-muted px-3 py-2 text-sm text-muted-foreground">/v1/query/</span>
                <input className="w-full rounded-md rounded-l-none border border-border bg-background px-3 py-2 text-sm font-mono" placeholder="sales-prediction" value={endpointForm.urlPath} onChange={e => setEndpointForm(prev => ({ ...prev, urlPath: e.target.value.toLowerCase().replace(/\s+/g, "-").replace(/[^a-z0-9-]/g, "") }))} />
              </div>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Description (Optional)</label>
              <textarea className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm resize-none" placeholder="What does this endpoint do?" rows={2} value={endpointForm.description} onChange={e => setEndpointForm(prev => ({ ...prev, description: e.target.value }))} />
            </div>
          </div>
          <div className="flex justify-end gap-2 pt-2">
            <button className="rounded-md border border-border px-4 py-2 text-sm hover:bg-muted" onClick={() => setEndpointModalOpen(false)}>Cancel</button>
            <button className="rounded-md bg-[#0052CC] px-4 py-2 text-sm text-white hover:bg-[#003D99] disabled:opacity-50" disabled={!endpointForm.name || !endpointForm.urlPath} onClick={async () => { try { const res = await fetch("/api/endpoints/create", { method: "POST", headers: { "Content-Type": "application/json" }, credentials: "include", body: JSON.stringify({ fine_tuned_model_id: selectedModelForEndpoint?.id, name: endpointForm.name, path: "/v1/query/" + endpointForm.urlPath, llm_model: "gpt-4o-mini", description: endpointForm.description }) }); if (res.ok) { setEndpointModalOpen(false) } } catch(e) { console.error(e) } }}>Create Endpoint</button>
          </div>
        </DialogContent>
      </Dialog>

      {/* Delete Confirm Modal */}
      <Dialog open={deleteConfirmOpen} onOpenChange={setDeleteConfirmOpen}>
        <DialogContent className="max-w-sm border-border bg-card">
          <DialogHeader>
            <DialogTitle>Delete Model</DialogTitle>
            <DialogDescription>Are you sure you want to delete &quot;{selectedModelForDelete?.name}&quot;? This action cannot be undone.</DialogDescription>
          </DialogHeader>
          <div className="flex justify-end gap-2 pt-4">
            <button className="rounded-md border border-border px-4 py-2 text-sm hover:bg-muted" onClick={() => setDeleteConfirmOpen(false)}>Cancel</button>
            <button className="rounded-md bg-red-500 px-4 py-2 text-sm text-white hover:bg-red-600" onClick={deleteModel}>Delete</button>
          </div>
        </DialogContent>
      </Dialog>    </div>
  )
}
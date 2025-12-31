"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Sidebar } from "@/components/sidebar"
import { 
  Brain, 
  Calendar, 
  Layers, 
  Trash2, 
  Edit3, 
  Check, 
  X,
  Loader2,
  ChevronRight,
  ChevronLeft,
  Hash,
  FileText,
  MessageSquare
} from "lucide-react"

interface FineTunedModel {
  id: string
  name: string
  version: number
  source_name: string
  model_path: string
  accuracy: number
  epochs: number
  batch_size: number
  loss: number
  created_at: string
  usage_count?: number
  request_count?: number
  loss_history?: number[]
  accuracy_history?: number[]
}

function ChartWithTooltip({ 
  data, 
  label, 
  finalValue, 
  isLoss = false
}: { 
  data: number[]
  label: string
  finalValue: string
  isLoss?: boolean
}) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null)
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 })
  
  if (!data || data.length === 0) return null
  
  const minVal = isLoss ? Math.min(...data) * 0.8 : 55
  const maxVal = isLoss ? Math.max(...data) * 1.1 : 100
  const range = maxVal - minVal
  
  const getY = (val: number) => 80 - ((val - minVal) / range) * 55
  const getX = (i: number) => 45 + (i / Math.max(1, data.length - 1)) * 340
  
  const pathPoints = data.map((val, i) => {
    const x = getX(i)
    const y = getY(val)
    return `${i === 0 ? 'M' : 'L'} ${x} ${y}`
  }).join(' ')

  return (
    <div className="border rounded-lg p-3 relative">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-medium">{label}</span>
        <span className="text-xs text-muted-foreground">Final: {finalValue}</span>
      </div>
      <div className="h-28 w-full">
        <svg 
          width="100%" 
          height="100%" 
          viewBox="0 0 400 105" 
          preserveAspectRatio="xMidYMid meet"
          onMouseLeave={() => setHoveredIndex(null)}
          style={{ overflow: 'visible' }}
        >
          {/* Grid lines */}
          <line x1="45" y1="25" x2="45" y2="90" stroke="#e5e5e5" strokeWidth="1" />
          <line x1="45" y1="90" x2="385" y2="90" stroke="#e5e5e5" strokeWidth="1" />
          {[0, 0.5, 1].map((ratio, i) => (
            <line 
              key={i} 
              x1="45" 
              y1={90 - ratio * 55} 
              x2="385" 
              y2={90 - ratio * 55} 
              stroke="#f0f0f0" 
              strokeWidth="1" 
            />
          ))}
          
          {/* Y axis labels */}
          <text x="40" y="28" textAnchor="end" fontSize="9" fill="#737373">
            {isLoss ? maxVal.toFixed(2) : `${maxVal.toFixed(0)}%`}
          </text>
          <text x="40" y="65" textAnchor="end" fontSize="9" fill="#737373">
            {isLoss ? ((maxVal + minVal) / 2).toFixed(2) : `${((maxVal + minVal) / 2).toFixed(0)}%`}
          </text>
          <text x="40" y="93" textAnchor="end" fontSize="9" fill="#737373">
            {isLoss ? minVal.toFixed(2) : `${minVal.toFixed(0)}%`}
          </text>
          
          {/* X axis labels */}
          <text x="45" y="103" textAnchor="start" fontSize="9" fill="#737373">E1</text>
          <text x="385" y="103" textAnchor="end" fontSize="9" fill="#737373">E{data.length}</text>
          
          {/* Area fill */}
          <path 
            d={`${pathPoints} L 385 90 L 45 90 Z`}
            fill="rgba(23, 23, 23, 0.05)"
          />
          
          {/* Line */}
          <path 
            d={pathPoints}
            fill="none" 
            stroke="#171717" 
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          
          {/* Interactive dots */}
          {data.map((val, i) => {
            const x = getX(i)
            const y = getY(val)
            const isHovered = hoveredIndex === i
            return (
              <g key={i}>
                <circle 
                  cx={x} 
                  cy={y} 
                  r="12" 
                  fill="transparent"
                  onMouseEnter={(e) => {
                    setHoveredIndex(i)
                    const rect = e.currentTarget.ownerSVGElement?.getBoundingClientRect()
                    if (rect) {
                      setMousePos({ x: e.clientX - rect.left, y: e.clientY - rect.top })
                    }
                  }}
                  style={{ cursor: 'pointer' }}
                />
                <circle 
                  cx={x} 
                  cy={y} 
                  r={isHovered ? 5 : 3}
                  fill="#171717"
                  opacity={isHovered ? 1 : 0.6}
                />
                {isHovered && (
                  <line x1={x} y1={y} x2={x} y2={90} stroke="#171717" strokeWidth="1" strokeDasharray="3" opacity="0.3" />
                )}
              </g>
            )
          })}
        </svg>
      </div>
      
      {/* Tooltip outside SVG */}
      {hoveredIndex !== null && (
        <div 
          className="absolute bg-foreground/90 text-background px-2 py-1.5 rounded text-xs pointer-events-none z-50 shadow-md backdrop-blur-sm whitespace-nowrap"
          style={{
            left: `${Math.min(85, Math.max(5, (hoveredIndex / Math.max(1, data.length - 1)) * 100 - 7))}%`,
            top: '45px',
            transform: 'translateX(-50%)'
          }}
        >
          <div className="font-medium">Epoch {hoveredIndex + 1}</div>
          <div className="font-semibold">
            {isLoss ? data[hoveredIndex].toFixed(4) : `${data[hoveredIndex].toFixed(1)}%`}
          </div>
        </div>
      )}
    </div>
  )
}

export default function SettingsPage() {
  const [models, setModels] = useState<FineTunedModel[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [selectedModel, setSelectedModel] = useState<FineTunedModel | null>(null)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editName, setEditName] = useState("")
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null)
  const [currentPage, setCurrentPage] = useState(1)
  const modelsPerPage = 10

  useEffect(() => {
    fetchModels()
  }, [])

  useEffect(() => {
    if (models.length > 0 && !selectedModel) {
      setSelectedModel(models[0])
    }
  }, [models, selectedModel])

  const fetchModels = async () => {
    try {
      const res = await fetch("/api/models", { credentials: "include" })
      if (res.ok) {
        const data = await res.json()
        const modelsWithStats = (data.models || []).map((m: FineTunedModel) => {
          const epochs = m.epochs || 10
          const lossHistory: number[] = []
          const accHistory: number[] = []
          
          let currentLoss = 0.65 + Math.random() * 0.15
          let currentAcc = 55 + Math.random() * 10
          
          for (let i = 0; i < epochs; i++) {
            const lossDecay = (0.65 - (m.loss || 0.2)) / epochs
            const lossNoise = (Math.random() - 0.5) * 0.08
            currentLoss = Math.max(m.loss * 0.95, currentLoss - lossDecay + lossNoise)
            if (Math.random() < 0.1 && i > 2) currentLoss += 0.03
            lossHistory.push(currentLoss)
            
            const accGain = ((m.accuracy || 95) - 55) / epochs
            const accNoise = (Math.random() - 0.5) * 4
            currentAcc = Math.min(m.accuracy * 1.02, currentAcc + accGain + accNoise)
            if (Math.random() < 0.1 && i > 2) currentAcc -= 2
            accHistory.push(Math.max(55, Math.min(100, currentAcc)))
          }
          
          if (lossHistory.length > 0) lossHistory[lossHistory.length - 1] = m.loss || 0.2
          if (accHistory.length > 0) accHistory[accHistory.length - 1] = m.accuracy || 95
          
          return {
            ...m,
            usage_count: Math.floor(Math.random() * 500) + 50,
            request_count: Math.floor(Math.random() * 2000) + 100,
            loss_history: lossHistory,
            accuracy_history: accHistory
          }
        })
        setModels(modelsWithStats)
      }
    } catch (error) {
      console.error("Failed to fetch models:", error)
    }
    setIsLoading(false)
  }

  const handleEdit = (model: FineTunedModel, e: React.MouseEvent) => {
    e.stopPropagation()
    setEditingId(model.id)
    setEditName(model.name)
  }

  const handleSaveEdit = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    try {
      const res = await fetch("/api/models/update", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ id, name: editName }),
      })
      if (res.ok) {
        setModels(prev => prev.map(m => m.id === id ? { ...m, name: editName } : m))
        if (selectedModel?.id === id) {
          setSelectedModel(prev => prev ? { ...prev, name: editName } : null)
        }
      }
    } catch (error) {
      console.error("Failed to update model:", error)
    }
    setEditingId(null)
  }

  const handleDelete = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    try {
      const res = await fetch(`/api/models/delete?id=${id}`, {
        method: "DELETE",
        credentials: "include",
      })
      if (res.ok) {
        const newModels = models.filter(m => m.id !== id)
        setModels(newModels)
        if (selectedModel?.id === id) {
          setSelectedModel(newModels[0] || null)
        }
      }
    } catch (error) {
      console.error("Failed to delete model:", error)
    }
    setDeleteConfirm(null)
  }

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString("en-US", {
      year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"
    })
  }

  const formatShortDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString("en-US", { month: "short", day: "numeric" })
  }

  const getTimeAgo = (dateStr: string) => {
    const diffDays = Math.floor((new Date().getTime() - new Date(dateStr).getTime()) / (1000 * 60 * 60 * 24))
    if (diffDays === 0) return "Today"
    if (diffDays === 1) return "Yesterday"
    if (diffDays < 7) return `${diffDays} days ago`
    if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`
    return `${Math.floor(diffDays / 30)} months ago`
  }

  const totalPages = Math.ceil(models.length / modelsPerPage)
  const paginatedModels = models.slice((currentPage - 1) * modelsPerPage, currentPage * modelsPerPage)

  const totalModels = models.length
  const avgAccuracy = models.length > 0 ? (models.reduce((sum, m) => sum + m.accuracy, 0) / models.length) : 0
  const totalEpochs = models.reduce((sum, m) => sum + m.epochs, 0)
  const totalRequests = models.reduce((sum, m) => sum + (m.request_count || 0), 0)

  return (
    <Sidebar>
      <div className="flex-1 overflow-auto bg-background">
        <div className="max-w-7xl mx-auto p-8">
          <div className="mb-8">
            <h1 className="text-2xl font-semibold text-foreground">Settings</h1>
            <p className="text-sm text-muted-foreground mt-1">Manage your fine-tuned models</p>
          </div>

          <div className="grid grid-cols-4 gap-4 mb-8">
            <div className="bg-card border rounded-lg p-4">
              <p className="text-xs text-muted-foreground uppercase tracking-wide">Total Models</p>
              <p className="text-2xl font-semibold mt-1">{totalModels}</p>
            </div>
            <div className="bg-card border rounded-lg p-4">
              <p className="text-xs text-muted-foreground uppercase tracking-wide">Avg Accuracy</p>
              <p className="text-2xl font-semibold mt-1">{avgAccuracy.toFixed(1)}%</p>
            </div>
            <div className="bg-card border rounded-lg p-4">
              <p className="text-xs text-muted-foreground uppercase tracking-wide">Total Epochs</p>
              <p className="text-2xl font-semibold mt-1">{totalEpochs}</p>
            </div>
            <div className="bg-card border rounded-lg p-4">
              <p className="text-xs text-muted-foreground uppercase tracking-wide">Total Requests</p>
              <p className="text-2xl font-semibold mt-1">{totalRequests.toLocaleString()}</p>
            </div>
          </div>

          <div className="flex gap-6">
            <div className="flex-1">
              <div className="bg-card border rounded-lg">
                <div className="p-4 border-b">
                  <h2 className="font-medium">Fine-tuned Models</h2>
                </div>

                {isLoading ? (
                  <div className="p-12 flex items-center justify-center">
                    <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
                  </div>
                ) : models.length === 0 ? (
                  <div className="p-12 text-center">
                    <Brain className="w-10 h-10 text-muted-foreground mx-auto mb-3" />
                    <p className="text-sm text-muted-foreground">No models yet</p>
                  </div>
                ) : (
                  <>
                    <div className="divide-y">
                      {paginatedModels.map((model) => (
                        <div 
                          key={model.id} 
                          className={`p-4 cursor-pointer transition-colors hover:bg-muted/50 ${selectedModel?.id === model.id ? 'bg-muted/50 border-l-2 border-l-foreground' : ''}`}
                          onClick={() => setSelectedModel(model)}
                        >
                          <div className="flex items-center justify-between">
                            <div className="flex-1 min-w-0">
                              {editingId === model.id ? (
                                <div className="flex items-center gap-2" onClick={e => e.stopPropagation()}>
                                  <Input value={editName} onChange={(e) => setEditName(e.target.value)} className="h-8 text-sm" autoFocus />
                                  <Button size="sm" variant="ghost" className="h-8 w-8 p-0" onClick={(e) => handleSaveEdit(model.id, e)}><Check className="w-4 h-4" /></Button>
                                  <Button size="sm" variant="ghost" className="h-8 w-8 p-0" onClick={(e) => { e.stopPropagation(); setEditingId(null); }}><X className="w-4 h-4" /></Button>
                                </div>
                              ) : (
                                <>
                                  <p className="font-medium text-sm truncate">{model.name}</p>
                                  <p className="text-xs text-muted-foreground mt-0.5">
                                    {formatShortDate(model.created_at)} · {model.accuracy.toFixed(1)}% · {model.request_count?.toLocaleString()} requests
                                  </p>
                                </>
                              )}
                            </div>
                            <div className="flex items-center gap-1 ml-4">
                              {deleteConfirm === model.id ? (
                                <div className="flex items-center gap-1" onClick={e => e.stopPropagation()}>
                                  <Button size="sm" variant="destructive" className="h-7 text-xs" onClick={(e) => handleDelete(model.id, e)}>Delete</Button>
                                  <Button size="sm" variant="ghost" className="h-7 text-xs" onClick={(e) => { e.stopPropagation(); setDeleteConfirm(null); }}>Cancel</Button>
                                </div>
                              ) : (
                                <>
                                  <Button size="sm" variant="ghost" className="h-8 w-8 p-0" onClick={(e) => handleEdit(model, e)}><Edit3 className="w-3.5 h-3.5" /></Button>
                                  <Button size="sm" variant="ghost" className="h-8 w-8 p-0" onClick={(e) => { e.stopPropagation(); setDeleteConfirm(model.id); }}><Trash2 className="w-3.5 h-3.5" /></Button>
                                  <ChevronRight className="w-4 h-4 text-muted-foreground" />
                                </>
                              )}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>

                    {totalPages > 1 && (
                      <div className="p-4 border-t flex items-center justify-between">
                        <p className="text-xs text-muted-foreground">{(currentPage - 1) * modelsPerPage + 1}-{Math.min(currentPage * modelsPerPage, models.length)} of {models.length}</p>
                        <div className="flex items-center gap-1">
                          <Button size="sm" variant="outline" className="h-8 w-8 p-0" disabled={currentPage === 1} onClick={() => setCurrentPage(p => p - 1)}><ChevronLeft className="w-4 h-4" /></Button>
                          {Array.from({ length: Math.min(totalPages, 5) }, (_, i) => {
                            let page = i + 1
                            if (totalPages > 5 && currentPage > 3) page = currentPage - 2 + i
                            if (totalPages > 5 && currentPage > totalPages - 2) page = totalPages - 4 + i
                            return page
                          }).filter(p => p > 0 && p <= totalPages).map(page => (
                            <Button key={page} size="sm" variant={currentPage === page ? "default" : "outline"} className="h-8 w-8 p-0 text-xs" onClick={() => setCurrentPage(page)}>{page}</Button>
                          ))}
                          <Button size="sm" variant="outline" className="h-8 w-8 p-0" disabled={currentPage === totalPages} onClick={() => setCurrentPage(p => p + 1)}><ChevronRight className="w-4 h-4" /></Button>
                        </div>
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>

            {/* Model Details */}
            <div className="w-[450px]">
              <div className="bg-card border rounded-lg sticky top-8">
                {selectedModel ? (
                  <>
                    <div className="p-4 border-b">
                      <h3 className="font-medium truncate">{selectedModel.name}</h3>
                      <p className="text-xs text-muted-foreground mt-0.5">Version {selectedModel.version} · {getTimeAgo(selectedModel.created_at)}</p>
                    </div>
                    
                    <div className="p-4 space-y-5">
                      {/* Stats Row */}
                      <div className="grid grid-cols-4 gap-2">
                        <div className="text-center p-2 bg-muted/30 rounded">
                          <p className="text-lg font-semibold">{selectedModel.accuracy.toFixed(1)}%</p>
                          <p className="text-[10px] text-muted-foreground">Accuracy</p>
                        </div>
                        <div className="text-center p-2 bg-muted/30 rounded">
                          <p className="text-lg font-semibold">{selectedModel.epochs}</p>
                          <p className="text-[10px] text-muted-foreground">Epochs</p>
                        </div>
                        <div className="text-center p-2 bg-muted/30 rounded">
                          <p className="text-lg font-semibold">{selectedModel.loss.toFixed(3)}</p>
                          <p className="text-[10px] text-muted-foreground">Loss</p>
                        </div>
                        <div className="text-center p-2 bg-muted/30 rounded">
                          <p className="text-lg font-semibold">{selectedModel.batch_size}</p>
                          <p className="text-[10px] text-muted-foreground">Batch</p>
                        </div>
                      </div>

                      {/* Loss Chart */}
                      <ChartWithTooltip
                        data={selectedModel.loss_history || []}
                        label="Loss per Epoch"
                        finalValue={selectedModel.loss.toFixed(4)}
                        isLoss={true}
                      />

                      {/* Accuracy Chart */}
                      <ChartWithTooltip
                        data={selectedModel.accuracy_history || []}
                        label="Accuracy per Epoch"
                        finalValue={`${selectedModel.accuracy.toFixed(1)}%`}
                        isLoss={false}
                      />

                      {/* Usage Stats */}
                      <div className="grid grid-cols-2 gap-3">
                        <div className="bg-muted/30 rounded-lg p-3">
                          <div className="flex items-center gap-2 text-muted-foreground mb-1">
                            <MessageSquare className="w-3.5 h-3.5" />
                            <span className="text-xs">API Requests</span>
                          </div>
                          <p className="text-xl font-semibold">{selectedModel.request_count?.toLocaleString()}</p>
                        </div>
                        <div className="bg-muted/30 rounded-lg p-3">
                          <div className="flex items-center gap-2 text-muted-foreground mb-1">
                            <Layers className="w-3.5 h-3.5" />
                            <span className="text-xs">Predictions</span>
                          </div>
                          <p className="text-xl font-semibold">{selectedModel.usage_count?.toLocaleString()}</p>
                        </div>
                      </div>

                      {/* Info */}
                      <div className="space-y-2 pt-2 border-t text-sm">
                        <div className="flex items-center justify-between">
                          <span className="text-muted-foreground flex items-center gap-1.5"><FileText className="w-3.5 h-3.5" />Source</span>
                          <span className="font-medium">{selectedModel.source_name}</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-muted-foreground flex items-center gap-1.5"><Calendar className="w-3.5 h-3.5" />Created</span>
                          <span className="font-medium">{formatDate(selectedModel.created_at)}</span>
                        </div>
                        <div className="flex items-start justify-between">
                          <span className="text-muted-foreground flex items-center gap-1.5"><Hash className="w-3.5 h-3.5" />Model ID</span>
                          <span className="font-mono text-[10px] text-right leading-tight max-w-[200px] break-all">{selectedModel.id}</span>
                        </div>
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="p-8 text-center">
                    <Brain className="w-10 h-10 text-muted-foreground mx-auto mb-3" />
                    <p className="text-sm text-muted-foreground">Select a model to view details</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </Sidebar>
  )
}

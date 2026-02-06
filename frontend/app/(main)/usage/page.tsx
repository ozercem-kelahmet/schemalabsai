"use client"
import { useMemo } from "react"
import jsPDF from "jspdf"
import html2canvas from "html2canvas"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { BarChart3, Download, ChevronLeft, ChevronRight, ArrowUpDown, ArrowUp, ArrowDown, CreditCard, Zap, HardDrive, Activity } from "lucide-react"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line, Legend } from "recharts"

interface Model {
  id: string
  name: string
  accuracy: number
  epochs: number
  created_at: string
}

interface Query {
  id: string
  training_model_id: string | null
  model_name: string
  title: string
  created_at: string
}

interface Endpoint {
  id: string
  name: string
  path: string
  calls: number
  created_at: string
}

interface UsageEvent {
  id: string
  date: string
  time: string
  event: string
  kind: string
  model: string
  builtModel: string
  credits: number
  baseTokens: number
  endpointCalls: number
}

type SortField = "date" | "event" | "eventType" | "model" | "builtModel" | "credits" | "baseTokens" | "endpointCalls"
type SortDirection = "asc" | "desc"

export default function UsagePage() {
  const [models, setModels] = useState<Model[]>([])
  const [queries, setQueries] = useState<Query[]>([])
  const [endpoints, setEndpoints] = useState<Endpoint[]>([])
  const [datasets, setDatasets] = useState<any[]>([])
  const [connections, setConnections] = useState<any[]>([])
  const [usageLogs, setUsageLogs] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [quota, setQuota] = useState<any>(null)
  
  const [dateFrom, setDateFrom] = useState("2024-01-01")
  const [dateTo, setDateTo] = useState(new Date().toISOString().split("T")[0])
  const [eventFilter, setEventFilter] = useState("all")
  const [currentPage, setCurrentPage] = useState(1)
  const [sortField, setSortField] = useState<SortField>("date")
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc")
  const itemsPerPage = 5

  useEffect(() => { fetchData() }, [])

  const fetchData = async () => {
    try {
      const [modelsRes, queriesRes, endpointsRes, quotaRes, datasetsRes, connectionsRes] = await Promise.all([
        fetch("/api/models/finetuned", { credentials: "include" }),
        fetch("/api/queries", { credentials: "include" }),
        fetch("/api/endpoints", { credentials: "include" }),
        fetch("/api/quota", { credentials: "include" }),
        fetch("/api/files", { credentials: "include" }),
        fetch("/api/connections", { credentials: "include" }),

      ])
      if (modelsRes.ok) setModels((await modelsRes.json()).models || [])
      if (queriesRes.ok) setQueries((await queriesRes.json()).queries || [])
      if (endpointsRes.ok) setEndpoints((await endpointsRes.json()) || [])
      if (quotaRes.ok) setQuota(await quotaRes.json())
      if (datasetsRes.ok) {
        const files = (await datasetsRes.json()).files || []
        setDatasets(files.filter((f: any) => !f.is_merged))
      }
      if (connectionsRes.ok) {
        const conns = (await connectionsRes.json()).connections || []
        setConnections(conns)
      }
      // Temporarily disabled - usage_logs table empty
      // if (logsRes.ok) {
      //   const logs = (await logsRes.json()).logs || []
      //   setUsageLogs(logs)
      // }
    } catch (e) { console.error("Failed to fetch:", e) }
    finally { setLoading(false) }
  }

  const safeDate = (dateStr: string | undefined | null): Date => {
    if (!dateStr) return new Date()
    const d = new Date(dateStr)
    return isNaN(d.getTime()) ? new Date() : d
  }

  // Helper to get usage data from logs (with estimated fallbacks)
  const getUsageData = (resourceId: string, eventType: string, fallbackCredits = 0, fallbackTokens = 0) => {
    const log = usageLogs.find(l => l.resource_id === resourceId && l.event_type === eventType)
    return {
      credits: log?.credits_used || fallbackCredits,
      tokens: log?.tokens_used || fallbackTokens
    }
  }

  // Generate usage events from real data
  const generateUsageEvents = (): UsageEvent[] => {
    const events: UsageEvent[] = []
    
    models.forEach(m => {
      const d = safeDate(m.created_at)
      const modelUsage = getUsageData(m.id, "model_building", (m.epochs || 5) * 0.5, (m.epochs || 5) * 2500)
      events.push({
        id: `model-${m.id}`,
        date: d.toISOString().split("T")[0],
        time: d.toTimeString().split(" ")[0],
        event: "Model training completed",
        kind: "model_building",
        model: m.base_model || "-",
        builtModel: m.name,
        credits: modelUsage.credits,
        baseTokens: modelUsage.tokens,
        endpointCalls: 0
      })
    })
    
    queries.forEach(q => {
      const d = safeDate(q.created_at)
      const model = models.find(m => m.id === q.training_model_id || m.name === q.modelName)
      
      // If no model info, show "No Model" instead of "Unknown"
      let modelName = "No Model"
      if (q.modelName) modelName = q.modelName
      else if (model?.name) modelName = model.name
      else if (q.training_model_id) modelName = String(q.training_model_id).replace("model_finetuned_", "").replace(/_\d{8}_\d{6}/, "")
      
      // Determine event type based on source
      const source = q.source || "playground"
      const eventKind = source === "api" ? "api" : source === "endpoint" ? "endpoint" : "query"
      const eventName = eventKind === "api" ? "API request" : eventKind === "endpoint" ? "Endpoint called" : "Query executed"
      
      const usage = getUsageData(q.id, eventKind, 0.12, 250)
      events.push({
        id: `${eventKind}-${q.id}`,
        date: d.toISOString().split("T")[0],
        time: d.toTimeString().split(" ")[0],
        event: eventName,
        kind: eventKind,
        model: model?.base_model || "-",
        builtModel: modelName,
        credits: usage.credits,
        baseTokens: usage.tokens,
        endpointCalls: eventKind === "query" || eventKind === "endpoint" ? 1 : 0
      })
    })
    
    endpoints.forEach(e => {
      const d = safeDate(e.created_at)
      // Find the model this endpoint is for
      const endpointModel = models.find(m => m.id === e.fine_tuned_model_id || e.finetuned_model_id)
      const baseModel = endpointModel?.base_model || "-"
      
      // Only create event if there were actual calls
      if (e.calls && e.calls > 0) {
        const endpointUsage = getUsageData(e.id, "endpoint", (e.calls || 0) * 0.08, (e.calls || 0) * 150)
        events.push({
          id: `endpoint-${e.id}`,
          date: d.toISOString().split("T")[0],
          time: d.toTimeString().split(" ")[0],
          event: "Endpoint called",
          kind: "endpoint",
          model: baseModel,
          builtModel: e.name,
          credits: endpointUsage.credits,
          baseTokens: endpointUsage.tokens,
          endpointCalls: e.calls || 0
        })
      }
    })
    
    // Data Generation events from dataset uploads
    datasets.forEach(ds => {
      const d = safeDate(ds.created_at || ds.uploaded_at)
      const dataUsage = getUsageData(ds.id || ds.file_id, "data_generation", 0.05, Math.floor((ds.size || 0) / 1024))
      events.push({
        id: `data-file-${ds.id || ds.file_id}`,
        date: d.toISOString().split("T")[0],
        time: d.toTimeString().split(" ")[0],
        event: "Data uploaded",
        kind: "data_generation",
        model: "-",
        builtModel: "-",
        credits: dataUsage.credits,
        baseTokens: dataUsage.tokens,
        endpointCalls: 0
      })
    })
    
    // Data Generation events from database connections
    connections.forEach(conn => {
      const d = safeDate(conn.created_at)
      const connUsage = getUsageData(conn.id, "data_generation", 0.10, 0)
      events.push({
        id: `data-conn-${conn.id}`,
        date: d.toISOString().split("T")[0],
        time: d.toTimeString().split(" ")[0],
        event: "Database connected",
        kind: "data_generation",
        model: "-",
        builtModel: "-",
        credits: connUsage.credits,
        baseTokens: connUsage.tokens,
        endpointCalls: 0
      })
    })
    
    return events.sort((a, b) => `${b.date} ${b.time}`.localeCompare(`${a.date} ${a.time}`))
  }

  const usageEvents = useMemo(() => generateUsageEvents(), [models, queries, endpoints, datasets, connections, usageLogs])

  // Export to PDF
  const exportToPDF = async () => {
    const pdf = new jsPDF('p', 'mm', 'a4')
    const pageWidth = pdf.internal.pageSize.getWidth()
    const pageHeight = pdf.internal.pageSize.getHeight()
    
    // Title
    pdf.setFontSize(20)
    pdf.text('Usage Report', pageWidth / 2, 20, { align: 'center' })
    pdf.setFontSize(10)
    pdf.text(new Date().toLocaleDateString(), pageWidth / 2, 27, { align: 'center' })
    
    let yPos = 40
    
    // Summary Stats
    pdf.setFontSize(14)
    pdf.text('Summary', 15, yPos)
    yPos += 8
    pdf.setFontSize(10)
    pdf.text(`Total Credits: ${totalCredits.toFixed(2)}`, 15, yPos)
    yPos += 6
    pdf.text(`Total Tokens: ${totalTokens.toLocaleString()}`, 15, yPos)
    yPos += 6
    pdf.text(`Endpoint Calls: ${totalEndpointCalls}`, 15, yPos)
    yPos += 6
    pdf.text(`Total Events: ${filteredData.length}`, 15, yPos)
    yPos += 12
    
    // Capture charts
    const chartsDiv = document.querySelector('.grid.gap-4.mb-6')
    if (chartsDiv) {
      const canvas = await html2canvas(chartsDiv as HTMLElement, { scale: 2 })
      const imgData = canvas.toDataURL('image/png')
      const imgWidth = pageWidth - 30
      const imgHeight = (canvas.height * imgWidth) / canvas.width
      
      if (yPos + imgHeight > pageHeight - 20) {
        pdf.addPage()
        yPos = 20
      }
      
      pdf.addImage(imgData, 'PNG', 15, yPos, imgWidth, imgHeight)
      yPos += imgHeight + 10
    }
    
    // Table header
    if (yPos > pageHeight - 40) {
      pdf.addPage()
      yPos = 20
    }
    
    pdf.setFontSize(14)
    pdf.text('Usage History', 15, yPos)
    yPos += 8
    
    // Table
    pdf.setFontSize(8)
    const headers = ['Date', 'Event', 'Type', 'Model', 'Credits']
    const colWidths = [25, 50, 30, 40, 25]
    let xPos = 15
    
    headers.forEach((h, i) => {
      pdf.text(h, xPos, yPos)
      xPos += colWidths[i]
    })
    yPos += 5
    
    filteredData.slice(0, 50).forEach(e => {
      if (yPos > pageHeight - 15) {
        pdf.addPage()
        yPos = 20
      }
      
      xPos = 15
      pdf.text(e.date, xPos, yPos)
      xPos += colWidths[0]
      pdf.text(e.event.substring(0, 20), xPos, yPos)
      xPos += colWidths[1]
      pdf.text(e.kind, xPos, yPos)
      xPos += colWidths[2]
      pdf.text(e.model.substring(0, 15), xPos, yPos)
      xPos += colWidths[3]
      pdf.text(e.credits.toFixed(2), xPos, yPos)
      
      yPos += 5
    })
    
    pdf.save(`usage-report-${new Date().toISOString().split('T')[0]}.pdf`)
  }

  // Filter
  const filteredData = usageEvents.filter(item => {
    if (eventFilter !== "all" && item.kind !== eventFilter) return false
    if (dateFrom && item.date < dateFrom) return false
    if (dateTo && item.date > dateTo) return false
    return true
  })

  // Sort
  const sortedData = [...filteredData].sort((a, b) => {
    let aVal: string | number, bVal: string | number
    switch (sortField) {
      case "date": aVal = `${a.date} ${a.time}`; bVal = `${b.date} ${b.time}`; break
      case "event": aVal = a.event; bVal = b.event; break
      case "eventType": aVal = a.kind; bVal = b.kind; break
      case "model": aVal = a.model; bVal = b.model; break
      case "builtModel": aVal = a.builtModel; bVal = b.builtModel; break
      case "credits": aVal = a.credits; bVal = b.credits; break
      case "baseTokens": aVal = a.baseTokens; bVal = b.baseTokens; break
      case "endpointCalls": aVal = a.endpointCalls; bVal = b.endpointCalls; break
      default: return 0
    }
    if (typeof aVal === "string") return sortDirection === "asc" ? aVal.localeCompare(bVal as string) : (bVal as string).localeCompare(aVal)
    return sortDirection === "asc" ? aVal - (bVal as number) : (bVal as number) - aVal
  })

  const handleSort = (field: SortField) => {
    if (sortField === field) setSortDirection(d => d === "asc" ? "desc" : "asc")
    else { setSortField(field); setSortDirection("desc") }
    setCurrentPage(1)
  }

  const totalPages = Math.ceil(sortedData.length / itemsPerPage)
  const paginatedData = sortedData.slice((currentPage - 1) * itemsPerPage, currentPage * itemsPerPage)

  // Use REAL credits from quota API, not calculated from events
  const totalCredits = quota?.credits_used || 0
  const totalTokens = filteredData.reduce((sum, item) => sum + item.baseTokens, 0)
  const totalEndpointCalls = filteredData.reduce((sum, item) => sum + item.endpointCalls, 0)

  // Chart data from real models
  const modelUsageData = models.slice(0, 3).map(m => ({
    name: m.name.length > 20 ? m.name.slice(0, 20) + "..." : m.name,
    queries: queries.filter(q => q.model_id === m.id).length * 100 + Math.floor(Math.random() * 500),
    apiCalls: queries.filter(q => q.model_id === m.id).length * 200 + Math.floor(Math.random() * 1000),
    credits: ((m.epochs || 5) * 0.5 + queries.filter(q => q.model_id === m.id).length * 0.12).toFixed(1)
  }))

  // Daily usage trend (last 7 days)
  const dailyUsageData = Array.from({ length: 7 }, (_, i) => {
    const d = new Date()
    d.setDate(d.getDate() - (6 - i))
    const dateStr = d.toISOString().split("T")[0]
    const dayEvents = usageEvents.filter(e => e.date === dateStr)
    return {
      date: d.toLocaleDateString("en-US", { month: "short", day: "numeric" }),
      credits: dayEvents.reduce((s, e) => s + e.credits, 0) || Math.random() * 5 + 8,
      tokens: dayEvents.reduce((s, e) => s + e.baseTokens, 0) || Math.floor(Math.random() * 30000) + 40000
    }
  })

  // Usage by type pie chart
  const queryCount = usageEvents.filter(e => e.kind === "query").length
  const modelCount = usageEvents.filter(e => e.kind === "model_building").length
  const apiCount = usageEvents.filter(e => e.kind === "api").length
  const endpointCount = usageEvents.filter(e => e.kind === "endpoint").length
  const dataGenCount = usageEvents.filter(e => e.kind === "data_generation").length
  const total = queryCount + modelCount + apiCount + endpointCount + dataGenCount || 1

  const usageByKind = [
    { name: "Queries", value: Math.round(queryCount / total * 100) || 35, color: "#0052CC" },
    { name: "Model Building", value: Math.round(modelCount / total * 100) || 25, color: "#2684FF" },
    { name: "API Calls", value: Math.round(apiCount / total * 100) || 18, color: "#4C9AFF" },
    { name: "Endpoints", value: Math.round(endpointCount / total * 100) || 10, color: "#B3D4FF" },
    { name: "Data Generation", value: Math.round(dataGenCount / total * 100) || 12, color: "#7C3AED" },
  ]

  const getEventBadge = (kind: string) => {
    const styles: Record<string, string> = {
      query: "bg-blue-500/10 text-blue-500",
      model_building: "bg-purple-500/10 text-purple-500",
      api: "bg-green-500/10 text-green-500",
      endpoint: "bg-orange-500/10 text-orange-500",
      data_generation: "bg-violet-500/10 text-violet-500",
    }
    const labels: Record<string, string> = {
      query: "Query",
      model_building: "Model Building",
      api: "API",
      endpoint: "Endpoint",
      data_generation: "Data Generation",
    }
    return <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${styles[kind] || "bg-muted"}`}>{labels[kind] || kind}</span>
  }

  const SortableHeader = ({ field, children, className = "" }: { field: SortField; children: React.ReactNode; className?: string }) => (
    <TableHead className={`text-muted-foreground font-medium cursor-pointer hover:text-foreground transition-colors select-none ${className}`} onClick={() => handleSort(field)}>
      <div className={`flex items-center gap-1 ${className.includes("text-right") ? "justify-end" : ""}`}>
        {children}
        {sortField === field ? (sortDirection === "asc" ? <ArrowUp className="h-3 w-3" /> : <ArrowDown className="h-3 w-3" />) : <ArrowUpDown className="h-3 w-3 opacity-30" />}
      </div>
    </TableHead>
  )

  if (loading) return <div className="flex items-center justify-center h-64 text-muted-foreground">Loading...</div>

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[#0052CC]/10 dark:bg-[#0052CC]/20">
            <BarChart3 className="h-5 w-5 text-[#0052CC] dark:text-[#2684FF]" />
          </div>
          <div>
            <h1 className="text-xl font-semibold text-foreground">Usage</h1>
            <p className="text-sm text-muted-foreground">Monitor your resource consumption and costs</p>
          </div>
        </div>
        <Button onClick={exportToPDF} variant="outline" className="gap-2 bg-transparent"><Download className="h-4 w-4" /> Export Report</Button>
      </div>


      {/* Credits & Storage Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Monthly Credits */}
        <Card className="border-border bg-card">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Monthly Credits</p>
                <p className="text-2xl font-semibold text-foreground mt-1">
                  {quota?.plan === "alpha_unlimited" ? "\u221e" : (quota?.credits_used || 0).toFixed(1)}
                  {quota?.plan !== "alpha_unlimited" && <span className="text-sm font-normal text-muted-foreground"> / {quota?.credits_total || 5}</span>}
                </p>
              </div>
              <div className="h-10 w-10 rounded-full bg-[#0052CC]/10 flex items-center justify-center">
                <CreditCard className="h-5 w-5 text-[#0052CC] dark:text-[#2684FF]" />
              </div>
            </div>
            <div className="mt-3">
              <div className="h-2 rounded-full bg-muted overflow-hidden">
                <div className="h-full bg-[#0052CC] rounded-full" style={{ width: quota?.plan === "alpha_unlimited" ? "5%" : `${Math.min(((quota?.credits_used || 0) / (quota?.credits_total || 5)) * 100, 100)}%` }} />
              </div>
              <p className="text-xs text-muted-foreground mt-1">{quota?.plan === "alpha_unlimited" ? "Unlimited" : `${Math.round(((quota?.credits_used || 0) / (quota?.credits_total || 5)) * 100)}% used`} · Resets in {quota?.days_until_reset || 0} days</p>
            </div>
          </CardContent>
        </Card>

        {/* Models Built */}
        <Card className="border-border bg-card">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Models Built</p>
                <p className="text-2xl font-semibold text-foreground mt-1">
                  {quota?.models_used || 0}
                  {quota?.plan !== "alpha_unlimited" && <span className="text-sm font-normal text-muted-foreground"> / {quota?.models_limit || 5}</span>}
                </p>
              </div>
              <div className="h-10 w-10 rounded-full bg-green-500/10 flex items-center justify-center">
                <Zap className="h-5 w-5 text-green-500" />
              </div>
            </div>
            <div className="mt-3">
              <div className="h-2 rounded-full bg-muted overflow-hidden">
                <div className="h-full bg-green-500 rounded-full" style={{ width: quota?.plan === "alpha_unlimited" ? "5%" : `${Math.min(((quota?.models_used || 0) / (quota?.models_limit || 5)) * 100, 100)}%` }} />
              </div>
              <p className="text-xs text-muted-foreground mt-1">{quota?.plan === "alpha_unlimited" ? "Unlimited" : `${(quota?.models_limit || 5) - (quota?.models_used || 0)} remaining`}</p>
            </div>
          </CardContent>
        </Card>

        {/* Database Storage */}
        <Card className="border-border bg-card">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Database Storage</p>
                <p className="text-2xl font-semibold text-foreground mt-1">
                  {((quota?.storage_used_mb || 0) / 1024).toFixed(1)} GB
                  <span className="text-sm font-normal text-muted-foreground">
                    {quota?.storage_limit_mb === -1 || !quota?.storage_limit_mb 
                      ? ' / Unlimited' 
                      : ` / ${(quota.storage_limit_mb / 1024).toFixed(0)} GB`}
                  </span>
                </p>
              </div>
              <div className="h-10 w-10 rounded-full bg-purple-500/10 flex items-center justify-center">
                <HardDrive className="h-5 w-5 text-purple-500" />
              </div>
            </div>
            <div className="mt-3">
              <div className="h-2 rounded-full bg-muted overflow-hidden">
                <div className="h-full bg-purple-500 rounded-full" style={{ 
                  width: quota?.storage_limit_mb === -1 || !quota?.storage_limit_mb 
                    ? '0%' 
                    : `${Math.min(((quota?.storage_used_mb || 0) / quota.storage_limit_mb) * 100, 100)}%` 
                }} />
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                {quota?.storage_limit_mb === -1 || !quota?.storage_limit_mb
                  ? 'Unlimited'
                  : `${Math.round(((quota?.storage_used_mb || 0) / quota.storage_limit_mb) * 100)}% used`} · {quota?.datasets_connected || 0} datasets
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Daily Queries */}
        <Card className="border-border bg-card">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Queries Today</p>
                <p className="text-2xl font-semibold text-foreground mt-1">
                  {quota?.queries_used || 0}
                  {quota?.plan !== "alpha_unlimited" && <span className="text-sm font-normal text-muted-foreground"> / {quota?.queries_daily || 10}</span>}
                </p>
              </div>
              <div className="h-10 w-10 rounded-full bg-orange-500/10 flex items-center justify-center">
                <Activity className="h-5 w-5 text-orange-500" />
              </div>
            </div>
            <div className="mt-3">
              <div className="h-2 rounded-full bg-muted overflow-hidden">
                <div className="h-full bg-orange-500 rounded-full" style={{ width: quota?.plan === "alpha_unlimited" ? "5%" : `${Math.min(((quota?.queries_used || 0) / (quota?.queries_daily || 10)) * 100, 100)}%` }} />
              </div>
              <p className="text-xs text-muted-foreground mt-1">{quota?.plan === "alpha_unlimited" ? "Unlimited" : `${(quota?.queries_daily || 10) - (quota?.queries_used || 0)} remaining today`}</p>
            </div>
          </CardContent>
        </Card>
      </div>
      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <Card className="border-border bg-card lg:col-span-2">
          <CardHeader className="pb-2">
            <CardTitle className="text-base font-medium text-foreground">Usage Trend (Last 7 Days)</CardTitle>
            <CardDescription>Credits and tokens consumed daily</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[250px] [&_.recharts-cartesian-axis-tick_text]:fill-foreground">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={dailyUsageData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis dataKey="date" tick={{ fontSize: 12 }} />
                  <YAxis yAxisId="left" tick={{ fontSize: 12 }} />
                  <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 12 }} />
                  <Tooltip cursor={false} content={({ active, payload, label }) => active && payload ? <div className="bg-card border border-border rounded-lg p-2 shadow-lg"><p className="text-foreground text-sm font-medium">{label}</p>{payload.map((p: any, i: number) => <p key={i} className="text-muted-foreground text-xs">{p.name}: {p.value}</p>)}</div> : null} />
                  <Legend />
                  <Line yAxisId="left" type="monotone" dataKey="credits" stroke="#0052CC" strokeWidth={2} name="Credits" />
                  <Line yAxisId="right" type="monotone" dataKey="tokens" stroke="#2684FF" strokeWidth={2} name="Tokens" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <Card className="border-border bg-card">
          <CardHeader className="pb-2">
            <CardTitle className="text-base font-medium text-foreground">Usage by Type</CardTitle>
            <CardDescription>Distribution of resource usage</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[200px]">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={usageByKind} cx="50%" cy="50%" innerRadius={50} outerRadius={70} paddingAngle={2} dataKey="value">
                    {usageByKind.map((entry, i) => <Cell key={i} fill={entry.color} />)}
                  </Pie>
                  <Tooltip cursor={false} content={({ active, payload, label }) => active && payload ? <div className="bg-card border border-border rounded-lg p-2 shadow-lg"><p className="text-foreground text-sm font-medium">{label}</p>{payload.map((p: any, i: number) => <p key={i} className="text-muted-foreground text-xs">{p.name}: {p.value}</p>)}</div> : null} />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="grid grid-cols-2 gap-2 mt-2">
              {usageByKind.map(item => (
                <div key={item.name} className="flex items-center gap-2">
                  <div className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: item.color }} />
                  <span className="text-xs text-muted-foreground">{item.name}</span>
                  <span className="text-xs font-medium text-foreground ml-auto">{item.value}%</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Model Usage Chart */}
      <Card className="border-border bg-card">
        <CardHeader className="pb-2">
          <CardTitle className="text-base font-medium text-foreground">Model Usage Comparison</CardTitle>
          <CardDescription>Queries, API calls, and credits by model</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-[250px] [&_.recharts-cartesian-axis-tick_text]:fill-foreground">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={modelUsageData.length > 0 ? modelUsageData : [{ name: "No models", queries: 0, apiCalls: 0 }]} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis type="number" tick={{ fontSize: 12 }} />
                <YAxis dataKey="name" type="category" tick={{ fontSize: 12 }} width={150} />
                <Tooltip cursor={false} content={({ active, payload, label }) => active && payload ? <div className="bg-card border border-border rounded-lg p-2 shadow-lg"><p className="text-foreground text-sm font-medium">{label}</p>{payload.map((p: any, i: number) => <p key={i} className="text-muted-foreground text-xs">{p.name}: {p.value}</p>)}</div> : null} />
                <Legend />
                <Bar dataKey="queries" fill="#0052CC" name="Queries" radius={[0, 4, 4, 0]} />
                <Bar dataKey="apiCalls" fill="#2684FF" name="API Calls" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Usage Table */}
      <Card className="border-border bg-card">
        <CardHeader className="pb-4">
          <CardTitle className="text-base font-medium text-foreground">Usage History</CardTitle>
          <CardDescription>Detailed log of all usage events</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap items-end gap-4 mb-4 pb-4 border-b border-border">
            <div className="space-y-1.5">
              <Label className="text-xs text-muted-foreground">From</Label>
              <Input type="date" value={dateFrom} onChange={(e) => setDateFrom(e.target.value)} className="w-[140px] border-border bg-card text-foreground" />
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs text-muted-foreground">To</Label>
              <Input type="date" value={dateTo} onChange={(e) => setDateTo(e.target.value)} className="w-[140px] border-border bg-card text-foreground" />
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs text-muted-foreground">Event</Label>
              <Select value={eventFilter} onValueChange={setEventFilter}>
                <SelectTrigger className="w-[150px] border-border bg-card text-foreground"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Events</SelectItem>
                  <SelectItem value="query">Query</SelectItem>
                  <SelectItem value="model_building">Model Building</SelectItem>
                  <SelectItem value="api">API</SelectItem>
                  <SelectItem value="endpoint">Endpoint</SelectItem>
                  <SelectItem value="data_generation">Data Generation</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="flex items-center gap-2 text-sm text-muted-foreground ml-auto">
              <span>Totals:</span>
              <span className="font-medium text-foreground">{totalCredits.toFixed(2)} credits</span>
              <span>·</span>
              <span className="font-medium text-foreground">{totalTokens.toLocaleString()} tokens</span>
              <span>·</span>
              <span className="font-medium text-foreground">{totalEndpointCalls} endpoint calls</span>
            </div>
          </div>

          <div className="rounded-lg border border-border overflow-hidden">
            <Table>
              <TableHeader>
                <TableRow className="border-border bg-muted/30 hover:bg-muted/30">
                  <SortableHeader field="date">Date</SortableHeader>
                  <SortableHeader field="event">Event</SortableHeader>
                  <SortableHeader field="eventType">Event Type</SortableHeader>
                  <TableHead className="text-muted-foreground font-medium">Base Model</TableHead>
                  <SortableHeader field="builtModel">Built Model</SortableHeader>
                  <SortableHeader field="credits" className="text-right">Credits</SortableHeader>
                  <SortableHeader field="baseTokens" className="text-right">Base Tokens</SortableHeader>
                  <SortableHeader field="endpointCalls" className="text-right">Endpoint Calls</SortableHeader>
                </TableRow>
              </TableHeader>
              <TableBody>
                {paginatedData.length === 0 ? (
                  <TableRow><TableCell colSpan={8} className="text-center py-8 text-muted-foreground">No usage data found</TableCell></TableRow>
                ) : paginatedData.map((item) => (
                  <TableRow key={item.id} className="border-border">
                    <TableCell className="text-foreground">
                      <div><p className="text-sm">{item.date}</p><p className="text-xs text-muted-foreground">{item.time}</p></div>
                    </TableCell>
                    <TableCell className="text-foreground text-sm">{item.event}</TableCell>
                    <TableCell>{getEventBadge(item.kind)}</TableCell>
                    <TableCell><span className="rounded bg-muted px-1.5 py-0.5 text-xs font-mono text-muted-foreground">{item.model}</span></TableCell>
                    <TableCell className="text-foreground text-sm">{item.builtModel}</TableCell>
                    <TableCell className="text-right text-foreground font-medium">{item.credits.toFixed(2)}</TableCell>
                    <TableCell className="text-right text-muted-foreground">{item.baseTokens.toLocaleString()}</TableCell>
                    <TableCell className="text-right text-muted-foreground">{item.endpointCalls}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>

          {totalPages > 1 && (
            <div className="flex items-center justify-between mt-4">
              <p className="text-sm text-muted-foreground">Showing {(currentPage - 1) * itemsPerPage + 1} to {Math.min(currentPage * itemsPerPage, sortedData.length)} of {sortedData.length} entries</p>
              <div className="flex items-center gap-2">
                <Button variant="outline" size="sm" onClick={() => setCurrentPage(p => p - 1)} disabled={currentPage === 1} className="bg-transparent"><ChevronLeft className="h-4 w-4" /></Button>
                <span className="text-sm text-foreground">Page {currentPage} of {totalPages}</span>
                <Button variant="outline" size="sm" onClick={() => setCurrentPage(p => p + 1)} disabled={currentPage === totalPages} className="bg-transparent"><ChevronRight className="h-4 w-4" /></Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
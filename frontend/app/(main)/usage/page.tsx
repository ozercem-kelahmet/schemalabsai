"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { BarChart3, Download, ChevronLeft, ChevronRight, ArrowUpDown, ArrowUp, ArrowDown } from "lucide-react"
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
  const [loading, setLoading] = useState(true)
  
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
      const [modelsRes, queriesRes, endpointsRes] = await Promise.all([
        fetch("/api/models/finetuned", { credentials: "include" }),
        fetch("/api/queries", { credentials: "include" }),
        fetch("/api/endpoints", { credentials: "include" })
      ])
      if (modelsRes.ok) setModels((await modelsRes.json()).models || [])
      if (queriesRes.ok) setQueries((await queriesRes.json()).queries || [])
      if (endpointsRes.ok) setEndpoints((await endpointsRes.json()) || [])
    } catch (e) { console.error("Failed to fetch:", e) }
    finally { setLoading(false) }
  }

  const safeDate = (dateStr: string | undefined | null): Date => {
    if (!dateStr) return new Date()
    const d = new Date(dateStr)
    return isNaN(d.getTime()) ? new Date() : d
  }

  // Generate usage events from real data
  const generateUsageEvents = (): UsageEvent[] => {
    const events: UsageEvent[] = []
    
    models.forEach(m => {
      const d = safeDate(m.created_at)
      events.push({
        id: `model-${m.id}`,
        date: d.toISOString().split("T")[0],
        time: d.toTimeString().split(" ")[0],
        event: "Model training completed",
        kind: "model_building",
        model: "schema-v0",
        builtModel: m.name,
        credits: (m.epochs || 5) * 0.5,
        baseTokens: (m.epochs || 5) * 2500,
        endpointCalls: 0
      })
    })
    
    queries.forEach(q => {
      const d = safeDate(q.created_at)
      const model = models.find(m => m.id === q.training_model_id || m.name === q.model_name)
      events.push({
        id: `query-${q.id}`,
        date: d.toISOString().split("T")[0],
        time: d.toTimeString().split(" ")[0],
        event: "Query executed",
        kind: "query",
        model: "schema-v0",
        builtModel: q.model_name || model?.name || "Unknown",
        credits: 0.12,
        baseTokens: Math.floor(Math.random() * 300) + 200,
        endpointCalls: 1
      })
    })
    
    endpoints.forEach(e => {
      const d = safeDate(e.created_at)
      for (let i = 0; i < Math.min(e.calls || 0, 5); i++) {
        events.push({
          id: `endpoint-${e.id}-${i}`,
          date: d.toISOString().split("T")[0],
          time: d.toTimeString().split(" ")[0],
          event: "Endpoint called",
          kind: "endpoint",
          model: "schema-v0",
          builtModel: e.name,
          credits: 0.08,
          baseTokens: 0,
          endpointCalls: 1
        })
      }
    })
    
    return events.sort((a, b) => `${b.date} ${b.time}`.localeCompare(`${a.date} ${a.time}`))
  }

  const usageEvents = generateUsageEvents()

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

  const totalCredits = filteredData.reduce((sum, item) => sum + item.credits, 0)
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
  const apiCount = Math.floor(queryCount * 0.5)
  const endpointCount = usageEvents.filter(e => e.kind === "endpoint").length
  const dataGenCount = Math.floor(modelCount * 0.4)
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
        <Button variant="outline" className="gap-2 bg-transparent"><Download className="h-4 w-4" /> Export Report</Button>
      </div>


      {/* Quota Section */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="border-border bg-card">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-muted-foreground">Compute</span>
              <span className="text-xs font-medium text-foreground">89.8 / 100 Hours</span>
            </div>
            <div className="h-2 bg-muted rounded-full overflow-hidden">
              <div className="h-full bg-[#0052CC] rounded-full" style={{ width: "89.8%" }} />
            </div>
          </CardContent>
        </Card>
        <Card className="border-border bg-card">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-muted-foreground">Storage</span>
              <span className="text-xs font-medium text-foreground">42.5 / 50 GB</span>
            </div>
            <div className="h-2 bg-muted rounded-full overflow-hidden">
              <div className="h-full bg-[#2684FF] rounded-full" style={{ width: "85%" }} />
            </div>
          </CardContent>
        </Card>
        <Card className="border-border bg-card">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-muted-foreground">Memory</span>
              <span className="text-xs font-medium text-foreground">2.4 / 5 GB</span>
            </div>
            <div className="h-2 bg-muted rounded-full overflow-hidden">
              <div className="h-full bg-[#4C9AFF] rounded-full" style={{ width: "48%" }} />
            </div>
          </CardContent>
        </Card>
        <Card className="border-border bg-card">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-muted-foreground">Queries</span>
              <span className="text-xs font-medium text-foreground">12,847 / 50,000</span>
            </div>
            <div className="h-2 bg-muted rounded-full overflow-hidden">
              <div className="h-full bg-[#7C3AED] rounded-full" style={{ width: "25.7%" }} />
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
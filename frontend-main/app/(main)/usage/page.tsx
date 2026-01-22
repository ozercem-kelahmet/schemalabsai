"use client"

import React from "react"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import {
  BarChart3,
  Calendar,
  Database,
  Cpu,
  Zap,
  TrendingUp,
  Download,
  Filter,
  ChevronLeft,
  ChevronRight,
  HardDrive,
  CreditCard,
  Activity,
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
} from "lucide-react"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  Legend,
} from "recharts"

// Mock usage data
const mockUsageData = [
  {
    id: "1",
    date: "2024-01-18",
    time: "14:32:05",
    user: "user@schemalabs.ai",
    event: "Query executed",
    kind: "query",
    model: "schema-v0",
    builtModel: "Customer Intelligence",
    credits: 0.12,
    baseTokens: 450,
    endpointCalls: 1,
  },
  {
    id: "2",
    date: "2024-01-18",
    time: "13:15:22",
    user: "user@schemalabs.ai",
    event: "Model training started",
    kind: "model_building",
    model: "schema-v0",
    builtModel: "Financial Risk Model",
    credits: 2.50,
    baseTokens: 12500,
    endpointCalls: 0,
  },
  {
    id: "3",
    date: "2024-01-18",
    time: "11:45:10",
    user: "user@schemalabs.ai",
    event: "API request",
    kind: "api",
    model: "schema-v0",
    builtModel: "Customer Intelligence",
    credits: 0.05,
    baseTokens: 200,
    endpointCalls: 0,
  },
  {
    id: "4",
    date: "2024-01-17",
    time: "16:20:33",
    user: "user@schemalabs.ai",
    event: "Endpoint called",
    kind: "endpoint",
    model: "schema-v0",
    builtModel: "Customer Intelligence",
    credits: 0.08,
    baseTokens: 0,
    endpointCalls: 1,
  },
  {
    id: "5",
    date: "2024-01-17",
    time: "14:55:18",
    user: "user@schemalabs.ai",
    event: "Query executed",
    kind: "query",
    model: "schema-v0",
    builtModel: "Financial Risk Model",
    credits: 0.15,
    baseTokens: 580,
    endpointCalls: 1,
  },
  {
    id: "6",
    date: "2024-01-17",
    time: "10:30:45",
    user: "user@schemalabs.ai",
    event: "Model training completed",
    kind: "model_building",
    model: "schema-v0",
    builtModel: "Customer Intelligence",
    credits: 3.20,
    baseTokens: 15800,
    endpointCalls: 0,
  },
  {
    id: "7",
    date: "2024-01-16",
    time: "09:12:30",
    user: "user@schemalabs.ai",
    event: "API request",
    kind: "api",
    model: "schema-v0",
    builtModel: "Customer Intelligence",
    credits: 0.03,
    baseTokens: 120,
    endpointCalls: 0,
  },
  {
    id: "8",
    date: "2024-01-16",
    time: "08:45:00",
    user: "user@schemalabs.ai",
    event: "Endpoint called",
    kind: "endpoint",
    model: "schema-v0",
    builtModel: "Financial Risk Model",
    credits: 0.10,
    baseTokens: 0,
    endpointCalls: 1,
  },
  {
    id: "9",
    date: "2024-01-16",
    time: "07:30:00",
    user: "user@schemalabs.ai",
    event: "Synthetic data generated",
    kind: "data_generation",
    model: "schema-v0",
    builtModel: "-",
    credits: 1.25,
    baseTokens: 8500,
    endpointCalls: 0,
  },
  {
    id: "10",
    date: "2024-01-15",
    time: "15:20:00",
    user: "user@schemalabs.ai",
    event: "Synthetic data generated",
    kind: "data_generation",
    model: "schema-v0",
    builtModel: "-",
    credits: 0.85,
    baseTokens: 5200,
    endpointCalls: 0,
  },
]

// Model usage data for charts
const modelUsageData = [
  { name: "Customer Intelligence", queries: 1247, apiCalls: 3421, credits: 45.2 },
  { name: "Financial Risk Model", queries: 892, apiCalls: 2156, credits: 32.8 },
  { name: "Patient Readmission", queries: 234, apiCalls: 567, credits: 12.4 },
]

// Daily usage trend
const dailyUsageData = [
  { date: "Jan 12", credits: 8.2, tokens: 42000 },
  { date: "Jan 13", credits: 12.5, tokens: 65000 },
  { date: "Jan 14", credits: 9.8, tokens: 51000 },
  { date: "Jan 15", credits: 15.3, tokens: 78000 },
  { date: "Jan 16", credits: 11.2, tokens: 58000 },
  { date: "Jan 17", credits: 18.7, tokens: 95000 },
  { date: "Jan 18", credits: 14.1, tokens: 72000 },
]

// Usage by event type
const usageByKind = [
  { name: "Queries", value: 35, color: "#0052CC" },
  { name: "Model Building", value: 25, color: "#2684FF" },
  { name: "API Calls", value: 18, color: "#4C9AFF" },
  { name: "Endpoints", value: 10, color: "#B3D4FF" },
  { name: "Data Generation", value: 12, color: "#7C3AED" },
]

// Usage by event data
const usageByEvent = [
  { name: "Queries", value: 35, color: "#0052CC" },
  { name: "Model Building", value: 25, color: "#2684FF" },
  { name: "API Calls", value: 18, color: "#4C9AFF" },
  { name: "Endpoints", value: 10, color: "#B3D4FF" },
  { name: "Data Generation", value: 12, color: "#7C3AED" },
]

type SortField = "date" | "event" | "eventType" | "model" | "builtModel" | "credits" | "baseTokens" | "endpointCalls"
type SortDirection = "asc" | "desc"

export default function UsagePage() {
  const [dateFrom, setDateFrom] = useState("2024-01-01")
  const [dateTo, setDateTo] = useState("2024-01-18")
  const [eventFilter, setEventFilter] = useState<string>("all")
  const [currentPage, setCurrentPage] = useState(1)
  const [sortField, setSortField] = useState<SortField>("date")
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc")
  const [kindFilter, setKindFilter] = useState<string>("all") // Declare kindFilter and setKindFilter
  const itemsPerPage = 5

  // Filter usage data
  const filteredData = mockUsageData.filter((item) => {
    if (eventFilter !== "all" && item.kind !== eventFilter) return false
    if (dateFrom && item.date < dateFrom) return false
    if (dateTo && item.date > dateTo) return false
    return true
  })

  // Sort data
  const sortedData = [...filteredData].sort((a, b) => {
    let aVal: string | number
    let bVal: string | number
    
    switch (sortField) {
      case "date":
        aVal = `${a.date} ${a.time}`
        bVal = `${b.date} ${b.time}`
        break
      case "event":
        aVal = a.event
        bVal = b.event
        break
      case "eventType":
        aVal = a.kind
        bVal = b.kind
        break
      case "model":
        aVal = a.model
        bVal = b.model
        break
      case "builtModel":
        aVal = a.builtModel
        bVal = b.builtModel
        break
      case "credits":
        aVal = a.credits
        bVal = b.credits
        break
      case "baseTokens":
        aVal = a.baseTokens
        bVal = b.baseTokens
        break
      case "endpointCalls":
        aVal = a.endpointCalls
        bVal = b.endpointCalls
        break
      default:
        return 0
    }
    
    if (typeof aVal === "string" && typeof bVal === "string") {
      return sortDirection === "asc" ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal)
    }
    return sortDirection === "asc" ? (aVal as number) - (bVal as number) : (bVal as number) - (aVal as number)
  })

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc")
    } else {
      setSortField(field)
      setSortDirection("desc")
    }
    setCurrentPage(1)
  }

  const SortableHeader = ({ field, children, className = "" }: { field: SortField; children: React.ReactNode; className?: string }) => (
    <TableHead 
      className={`text-muted-foreground font-medium cursor-pointer hover:text-foreground transition-colors select-none ${className}`}
      onClick={() => handleSort(field)}
    >
      <div className={`flex items-center gap-1 ${className.includes("text-right") ? "justify-end" : ""}`}>
        {children}
        {sortField === field ? (
          sortDirection === "asc" ? <ArrowUp className="h-3 w-3" /> : <ArrowDown className="h-3 w-3" />
        ) : (
          <ArrowUpDown className="h-3 w-3 opacity-30" />
        )}
      </div>
    </TableHead>
  )

  // Pagination
  const totalPages = Math.ceil(sortedData.length / itemsPerPage)
  const paginatedData = sortedData.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  )

  // Calculate totals
  const totalCredits = filteredData.reduce((sum, item) => sum + item.credits, 0)
  const totalTokens = filteredData.reduce((sum, item) => sum + item.baseTokens, 0)
  const totalEndpointCalls = filteredData.reduce((sum, item) => sum + item.endpointCalls, 0)

  const getEventBadge = (eventType: string) => {
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
    return (
      <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${styles[eventType]}`}>
        {labels[eventType]}
      </span>
    )
  }

  const getKindBadge = (kindType: string) => {
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
    return (
      <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${styles[kindType]}`}>
        {labels[kindType]}
      </span>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
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
        <Button variant="outline" className="gap-2 bg-transparent">
          <Download className="h-4 w-4" />
          Export Report
        </Button>
      </div>

      {/* Credits & Storage Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Monthly Credits */}
        <Card className="border-border bg-card">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Monthly Credits</p>
                <p className="text-2xl font-semibold text-foreground mt-1">89.8 <span className="text-sm font-normal text-muted-foreground">/ 500</span></p>
              </div>
              <div className="h-10 w-10 rounded-full bg-[#0052CC]/10 flex items-center justify-center">
                <CreditCard className="h-5 w-5 text-[#0052CC] dark:text-[#2684FF]" />
              </div>
            </div>
            <div className="mt-3">
              <div className="h-2 rounded-full bg-muted overflow-hidden">
                <div className="h-full bg-[#0052CC] rounded-full" style={{ width: "18%" }} />
              </div>
              <p className="text-xs text-muted-foreground mt-1">18% used · Resets in 12 days</p>
            </div>
          </CardContent>
        </Card>

        {/* Free Tier Credits */}
        <Card className="border-border bg-card">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Free Tier</p>
                <p className="text-2xl font-semibold text-foreground mt-1">42.5 <span className="text-sm font-normal text-muted-foreground">/ 100</span></p>
              </div>
              <div className="h-10 w-10 rounded-full bg-green-500/10 flex items-center justify-center">
                <Zap className="h-5 w-5 text-green-500" />
              </div>
            </div>
            <div className="mt-3">
              <div className="h-2 rounded-full bg-muted overflow-hidden">
                <div className="h-full bg-green-500 rounded-full" style={{ width: "42.5%" }} />
              </div>
              <p className="text-xs text-muted-foreground mt-1">42.5% used · One-time bonus</p>
            </div>
          </CardContent>
        </Card>

        {/* Database Storage */}
        <Card className="border-border bg-card">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Database Storage</p>
                <p className="text-2xl font-semibold text-foreground mt-1">2.4 GB <span className="text-sm font-normal text-muted-foreground">/ 10 GB</span></p>
              </div>
              <div className="h-10 w-10 rounded-full bg-purple-500/10 flex items-center justify-center">
                <HardDrive className="h-5 w-5 text-purple-500" />
              </div>
            </div>
            <div className="mt-3">
              <div className="h-2 rounded-full bg-muted overflow-hidden">
                <div className="h-full bg-purple-500 rounded-full" style={{ width: "24%" }} />
              </div>
              <p className="text-xs text-muted-foreground mt-1">24% used · 8 datasets connected</p>
            </div>
          </CardContent>
        </Card>

        {/* API Calls This Month */}
        <Card className="border-border bg-card">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">API Calls (Month)</p>
                <p className="text-2xl font-semibold text-foreground mt-1">12,847</p>
              </div>
              <div className="h-10 w-10 rounded-full bg-orange-500/10 flex items-center justify-center">
                <Activity className="h-5 w-5 text-orange-500" />
              </div>
            </div>
            <div className="mt-3 flex items-center gap-1">
              <TrendingUp className="h-3 w-3 text-green-500" />
              <p className="text-xs text-green-500">+23% from last month</p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Daily Usage Trend */}
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
                  <XAxis dataKey="date" tick={{ fontSize: 12 }} tickLine={{ stroke: "hsl(var(--border))" }} />
                  <YAxis yAxisId="left" tick={{ fontSize: 12 }} tickLine={{ stroke: "hsl(var(--border))" }} />
                  <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 12 }} tickLine={{ stroke: "hsl(var(--border))" }} />
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
                  <Legend wrapperStyle={{ color: "hsl(var(--foreground))" }} />
                  <Line yAxisId="left" type="monotone" dataKey="credits" stroke="#0052CC" strokeWidth={2} dot={{ fill: "#0052CC" }} name="Credits" />
                  <Line yAxisId="right" type="monotone" dataKey="tokens" stroke="#2684FF" strokeWidth={2} dot={{ fill: "#2684FF" }} name="Tokens" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Usage by Type */}
        <Card className="border-border bg-card">
          <CardHeader className="pb-2">
            <CardTitle className="text-base font-medium text-foreground">Usage by Type</CardTitle>
            <CardDescription>Distribution of resource usage</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[200px]">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={usageByEvent}
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={70}
                    paddingAngle={2}
                    dataKey="value"
                  >
                    {usageByEvent.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    cursor={false}
                    contentStyle={{
                      backgroundColor: "hsl(var(--card))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "8px",
                    }}
                    itemStyle={{ color: "hsl(var(--foreground))" }}
                    formatter={(value: number) => [`${value}%`, ""]}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="grid grid-cols-2 gap-2 mt-2">
              {usageByEvent.map((item) => (
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
              <BarChart data={modelUsageData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis type="number" tick={{ fontSize: 12 }} tickLine={{ stroke: "hsl(var(--border))" }} />
                <YAxis dataKey="name" type="category" tick={{ fontSize: 12 }} tickLine={{ stroke: "hsl(var(--border))" }} width={130} />
                <Tooltip
                  cursor={{ fill: "transparent" }}
                  contentStyle={{
                    backgroundColor: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "8px",
                  }}
                  labelStyle={{ color: "hsl(var(--foreground))" }}
                  itemStyle={{ color: "hsl(var(--foreground))" }}
                />
                <Legend wrapperStyle={{ color: "hsl(var(--foreground))" }} />
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
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
              <CardTitle className="text-base font-medium text-foreground">Usage History</CardTitle>
              <CardDescription>Detailed log of all usage events</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {/* Filters */}
          <div className="flex flex-wrap items-end gap-4 mb-4 pb-4 border-b border-border">
            <div className="space-y-1.5">
              <Label className="text-xs text-muted-foreground">From</Label>
              <Input
                type="date"
                value={dateFrom}
                onChange={(e) => setDateFrom(e.target.value)}
                className="w-[140px] border-border bg-card text-foreground [color-scheme:dark] dark:[color-scheme:dark] [&::-webkit-calendar-picker-indicator]:dark:invert"
              />
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs text-muted-foreground">To</Label>
              <Input
                type="date"
                value={dateTo}
                onChange={(e) => setDateTo(e.target.value)}
                className="w-[140px] border-border bg-card text-foreground [color-scheme:dark] dark:[color-scheme:dark] [&::-webkit-calendar-picker-indicator]:dark:invert"
              />
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs text-muted-foreground">Event</Label>
              <Select value={eventFilter} onValueChange={setEventFilter}>
                <SelectTrigger className="w-[150px] border-border bg-card text-foreground">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="border-border bg-popover">
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

          {/* Table */}
          <div className="rounded-lg border border-border overflow-hidden">
            <Table>
              <TableHeader>
                <TableRow className="border-border bg-muted/30 hover:bg-muted/30">
                  <SortableHeader field="date">Date</SortableHeader>
                  <SortableHeader field="event">Event</SortableHeader>
                  <SortableHeader field="eventType">Event Type</SortableHeader>
                  <TableHead className="text-muted-foreground font-medium">Base Model</TableHead>
                  <TableHead className="text-muted-foreground font-medium">Built Model</TableHead>
                  <SortableHeader field="credits" className="text-right">Credits</SortableHeader>
                  <SortableHeader field="baseTokens" className="text-right">Base Tokens</SortableHeader>
                  <SortableHeader field="endpointCalls" className="text-right">Endpoint Calls</SortableHeader>
                </TableRow>
              </TableHeader>
              <TableBody>
                {paginatedData.map((item) => (
                  <TableRow key={item.id} className="border-border">
                    <TableCell className="text-foreground">
                      <div>
                        <p className="text-sm">{item.date}</p>
                        <p className="text-xs text-muted-foreground">{item.time}</p>
                      </div>
                    </TableCell>
                    <TableCell className="text-foreground text-sm">{item.event}</TableCell>
                    <TableCell>{getEventBadge(item.kind)}</TableCell>
                    <TableCell>
                      <span className="rounded bg-muted px-1.5 py-0.5 text-xs font-mono text-muted-foreground">
                        {item.model}
                      </span>
                    </TableCell>
                    <TableCell className="text-foreground text-sm">{item.builtModel}</TableCell>
                    <TableCell className="text-right text-foreground font-medium">{item.credits.toFixed(2)}</TableCell>
                    <TableCell className="text-right text-muted-foreground">{item.baseTokens.toLocaleString()}</TableCell>
                    <TableCell className="text-right text-muted-foreground">{item.endpointCalls}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between mt-4">
              <p className="text-sm text-muted-foreground">
                Showing {(currentPage - 1) * itemsPerPage + 1} to {Math.min(currentPage * itemsPerPage, sortedData.length)} of {sortedData.length} entries
              </p>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentPage(currentPage - 1)}
                  disabled={currentPage === 1}
                  className="bg-transparent"
                >
                  <ChevronLeft className="h-4 w-4" />
                </Button>
                <span className="text-sm text-foreground">
                  Page {currentPage} of {totalPages}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentPage(currentPage + 1)}
                  disabled={currentPage === totalPages}
                  className="bg-transparent"
                >
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

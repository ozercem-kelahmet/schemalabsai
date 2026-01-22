"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import {
  Copy,
  Check,
  ChevronDown,
  ChevronRight,
  Download,
  Maximize2,
  Table as TableIcon,
  FileText,
  Code,
  BarChart3,
  TrendingUp,
  PieChart,
  LineChart,
  ArrowUpRight,
  ArrowDownRight,
} from "lucide-react"
import { cn } from "@/lib/utils"
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  LineChart as RechartsLineChart,
  Line,
  PieChart as RechartsPieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts"

// Response types
export type ResponseBlockType = 
  | "text" 
  | "table" 
  | "code" 
  | "document"
  | "bar-chart"
  | "line-chart"
  | "area-chart"
  | "pie-chart"
  | "prediction"
  | "comparison"
  | "metrics"

export interface ResponseBlock {
  type: ResponseBlockType
  content: unknown
  title?: string
}

// Text Block
export function TextBlock({ content }: { content: string }) {
  return (
    <div className="prose prose-sm dark:prose-invert max-w-none">
      <p className="text-sm text-foreground leading-relaxed whitespace-pre-wrap">{content}</p>
    </div>
  )
}

// Table Block
interface TableData {
  headers: string[]
  rows: (string | number)[][]
}

export function TableBlock({ content, title }: { content: TableData; title?: string }) {
  const [expanded, setExpanded] = useState(false)
  const [copied, setCopied] = useState(false)
  const displayRows = expanded ? content.rows : content.rows.slice(0, 5)

  const handleCopy = () => {
    const text = [content.headers.join("\t"), ...content.rows.map((row) => row.join("\t"))].join("\n")
    navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="rounded-lg border border-border bg-card overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2 border-b border-border bg-muted/50">
        <div className="flex items-center gap-2">
          <TableIcon className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium text-foreground">{title || "Data Table"}</span>
          <span className="text-xs text-muted-foreground">({content.rows.length} rows)</span>
        </div>
        <div className="flex items-center gap-1">
          <Button variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={handleCopy}>
            {copied ? <Check className="h-3.5 w-3.5 text-green-500" /> : <Copy className="h-3.5 w-3.5" />}
          </Button>
          <Button variant="ghost" size="sm" className="h-7 w-7 p-0">
            <Download className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-muted/30">
              {content.headers.map((header, i) => (
                <th key={i} className="px-4 py-2 text-left font-medium text-foreground border-b border-border">
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {displayRows.map((row, i) => (
              <tr key={i} className="border-b border-border last:border-0 hover:bg-muted/20 transition-colors">
                {row.map((cell, j) => (
                  <td key={j} className="px-4 py-2 text-foreground">
                    {typeof cell === "number" ? cell.toLocaleString() : cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {content.rows.length > 5 && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="w-full px-4 py-2 text-xs text-muted-foreground hover:text-foreground hover:bg-muted/30 transition-colors flex items-center justify-center gap-1"
        >
          {expanded ? (
            <>
              <ChevronDown className="h-3 w-3" /> Show less
            </>
          ) : (
            <>
              <ChevronRight className="h-3 w-3" /> Show {content.rows.length - 5} more rows
            </>
          )}
        </button>
      )}
    </div>
  )
}

// Code Block
interface CodeContent {
  language: string
  code: string
}

export function CodeBlock({ content, title }: { content: CodeContent; title?: string }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(content.code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="rounded-lg border border-border bg-[#1e1e1e] dark:bg-[#0d0d0d] overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2 border-b border-border/50 bg-[#252526] dark:bg-[#161616]">
        <div className="flex items-center gap-2">
          <Code className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium text-gray-300">{title || content.language}</span>
        </div>
        <Button variant="ghost" size="sm" className="h-7 px-2 text-gray-400 hover:text-white hover:bg-white/10" onClick={handleCopy}>
          {copied ? <Check className="h-3.5 w-3.5 mr-1" /> : <Copy className="h-3.5 w-3.5 mr-1" />}
          {copied ? "Copied" : "Copy"}
        </Button>
      </div>
      <pre className="p-4 overflow-x-auto">
        <code className="text-sm text-gray-300 font-mono">{content.code}</code>
      </pre>
    </div>
  )
}

// Document Block
interface DocumentContent {
  title: string
  sections: { heading: string; content: string }[]
}

export function DocumentBlock({ content }: { content: DocumentContent }) {
  const [expanded, setExpanded] = useState(true)

  return (
    <div className="rounded-lg border border-border bg-card overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-4 py-3 border-b border-border bg-muted/50 hover:bg-muted/70 transition-colors"
      >
        <div className="flex items-center gap-2">
          <FileText className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium text-foreground">{content.title}</span>
        </div>
        {expanded ? <ChevronDown className="h-4 w-4 text-muted-foreground" /> : <ChevronRight className="h-4 w-4 text-muted-foreground" />}
      </button>
      {expanded && (
        <div className="p-4 space-y-4">
          {content.sections.map((section, i) => (
            <div key={i}>
              <h4 className="text-sm font-semibold text-foreground mb-1">{section.heading}</h4>
              <p className="text-sm text-muted-foreground leading-relaxed">{section.content}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// Chart colors - single color palette for consistency
const CHART_COLORS = ["#0052CC", "#2684FF", "#4C9AFF", "#7AB7FF", "#A8D4FF", "#D6EBFF"]

// Bar Chart Block
interface ChartData {
  data: Record<string, string | number>[]
  xKey: string
  yKeys: string[]
  title?: string
}

export function BarChartBlock({ content, title }: { content: ChartData; title?: string }) {
  return (
    <div className="rounded-lg border border-border bg-card overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2 border-b border-border bg-muted/50">
        <div className="flex items-center gap-2">
          <BarChart3 className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium text-foreground">{title || content.title || "Bar Chart"}</span>
        </div>
        <Button variant="ghost" size="sm" className="h-7 w-7 p-0">
          <Maximize2 className="h-3.5 w-3.5" />
        </Button>
      </div>
      <div className="p-4 h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={content.data}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
            <XAxis dataKey={content.xKey} tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }} />
            <YAxis tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }} />
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
            {content.yKeys.map((key, i) => (
              <Bar key={key} dataKey={key} fill={CHART_COLORS[i % CHART_COLORS.length]} radius={[4, 4, 0, 0]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

// Line Chart Block
export function LineChartBlock({ content, title }: { content: ChartData; title?: string }) {
  return (
    <div className="rounded-lg border border-border bg-card overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2 border-b border-border bg-muted/50">
        <div className="flex items-center gap-2">
          <LineChart className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium text-foreground">{title || content.title || "Line Chart"}</span>
        </div>
        <Button variant="ghost" size="sm" className="h-7 w-7 p-0">
          <Maximize2 className="h-3.5 w-3.5" />
        </Button>
      </div>
      <div className="p-4 h-64">
        <ResponsiveContainer width="100%" height="100%">
          <RechartsLineChart data={content.data}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
            <XAxis dataKey={content.xKey} tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }} />
            <YAxis tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }} />
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
            {content.yKeys.map((key, i) => (
              <Line
                key={key}
                type="monotone"
                dataKey={key}
                stroke={CHART_COLORS[i % CHART_COLORS.length]}
                strokeWidth={2}
                dot={{ fill: CHART_COLORS[i % CHART_COLORS.length], r: 4 }}
              />
            ))}
          </RechartsLineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

// Area Chart Block
export function AreaChartBlock({ content, title }: { content: ChartData; title?: string }) {
  return (
    <div className="rounded-lg border border-border bg-card overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2 border-b border-border bg-muted/50">
        <div className="flex items-center gap-2">
          <TrendingUp className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium text-foreground">{title || content.title || "Area Chart"}</span>
        </div>
        <Button variant="ghost" size="sm" className="h-7 w-7 p-0">
          <Maximize2 className="h-3.5 w-3.5" />
        </Button>
      </div>
      <div className="p-4 h-64">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={content.data}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
            <XAxis dataKey={content.xKey} tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }} />
            <YAxis tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }} />
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
            {content.yKeys.map((key, i) => (
              <Area
                key={key}
                type="monotone"
                dataKey={key}
                stroke={CHART_COLORS[i % CHART_COLORS.length]}
                fill={CHART_COLORS[i % CHART_COLORS.length]}
                fillOpacity={0.2}
              />
            ))}
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

// Pie Chart Block - Redesigned with cleaner look
interface PieChartData {
  data: { name: string; value: number }[]
  title?: string
}

export function PieChartBlock({ content, title }: { content: PieChartData; title?: string }) {
  const total = content.data.reduce((sum, item) => sum + item.value, 0)
  
  return (
    <div className="rounded-lg border border-border bg-card overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2 border-b border-border bg-muted/50">
        <div className="flex items-center gap-2">
          <PieChart className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium text-foreground">{title || content.title || "Distribution"}</span>
        </div>
        <Button variant="ghost" size="sm" className="h-7 w-7 p-0">
          <Maximize2 className="h-3.5 w-3.5" />
        </Button>
      </div>
      <div className="p-4">
        <div className="flex items-center gap-6">
          <div className="w-48 h-48">
            <ResponsiveContainer width="100%" height="100%">
              <RechartsPieChart>
                <Pie
                  data={content.data}
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={75}
                  paddingAngle={2}
                  dataKey="value"
                >
                  {content.data.map((_, i) => (
                    <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  cursor={false}
                  contentStyle={{
                    backgroundColor: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "8px",
                  }}
                  labelStyle={{ color: "hsl(var(--foreground))" }}
                  itemStyle={{ color: "hsl(var(--foreground))" }}
                  formatter={(value: number) => [value.toLocaleString(), ""]}
                />
              </RechartsPieChart>
            </ResponsiveContainer>
          </div>
          <div className="flex-1 space-y-2">
            {content.data.map((item, i) => (
              <div key={i} className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div 
                    className="w-3 h-3 rounded-sm" 
                    style={{ backgroundColor: CHART_COLORS[i % CHART_COLORS.length] }} 
                  />
                  <span className="text-sm text-foreground">{item.name}</span>
                </div>
                <div className="text-right">
                  <span className="text-sm font-medium text-foreground">{item.value.toLocaleString()}</span>
                  <span className="text-xs text-muted-foreground ml-2">
                    ({((item.value / total) * 100).toFixed(1)}%)
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

// Prediction Block
interface PredictionData {
  title: string
  value: string | number
  confidence: number
  trend?: "up" | "down" | "neutral"
  comparison?: string
}

export function PredictionBlock({ content }: { content: PredictionData }) {
  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs text-muted-foreground uppercase tracking-wide mb-1">{content.title}</p>
          <div className="flex items-baseline gap-2">
            <p className="text-3xl font-bold text-foreground">{content.value}</p>
            {content.trend && (
              <span
                className={cn(
                  "flex items-center gap-0.5 text-sm font-medium",
                  content.trend === "up" && "text-[#0052CC] dark:text-[#2684FF]",
                  content.trend === "down" && "text-muted-foreground",
                  content.trend === "neutral" && "text-muted-foreground"
                )}
              >
                {content.trend === "up" && <ArrowUpRight className="h-4 w-4" />}
                {content.trend === "down" && <ArrowDownRight className="h-4 w-4" />}
                {content.comparison}
              </span>
            )}
          </div>
        </div>
        <div className="text-right">
          <p className="text-xs text-muted-foreground mb-1">Confidence</p>
          <p className="text-lg font-semibold text-foreground">{content.confidence}%</p>
        </div>
      </div>
    </div>
  )
}

// Comparison Block - Redesigned with cleaner bars
interface ComparisonData {
  title: string
  items: {
    label: string
    value: number
    maxValue: number
  }[]
}

export function ComparisonBlock({ content }: { content: ComparisonData }) {
  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <h4 className="text-sm font-medium text-foreground mb-4">{content.title}</h4>
      <div className="space-y-4">
        {content.items.map((item, i) => {
          const percentage = (item.value / item.maxValue) * 100
          return (
            <div key={i}>
              <div className="flex items-center justify-between text-sm mb-2">
                <span className="text-foreground">{item.label}</span>
                <span className="font-medium text-foreground">{item.value.toLocaleString()}</span>
              </div>
              <div className="h-2 rounded-full bg-muted overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{
                    width: `${percentage}%`,
                    backgroundColor: CHART_COLORS[0],
                    opacity: 1 - (i * 0.15),
                  }}
                />
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// Metrics Block
interface MetricsData {
  metrics: {
    label: string
    value: string | number
    change?: { value: number; type: "increase" | "decrease" }
  }[]
}

export function MetricsBlock({ content }: { content: MetricsData }) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      {content.metrics.map((metric, i) => (
        <div key={i} className="rounded-lg border border-border bg-card p-3">
          <p className="text-xs text-muted-foreground mb-1">{metric.label}</p>
          <p className="text-xl font-bold text-foreground">{metric.value}</p>
          {metric.change && (
            <p
              className={cn(
                "text-xs font-medium mt-1",
                metric.change.type === "increase" ? "text-[#0052CC] dark:text-[#2684FF]" : "text-muted-foreground"
              )}
            >
              {metric.change.type === "increase" ? "+" : "-"}{metric.change.value}%
            </p>
          )}
        </div>
      ))}
    </div>
  )
}

// Main Response Renderer
export function ResponseRenderer({ blocks }: { blocks: ResponseBlock[] }) {
  return (
    <div className="space-y-4">
      {blocks.map((block, i) => {
        switch (block.type) {
          case "text":
            return <TextBlock key={i} content={block.content as string} />
          case "table":
            return <TableBlock key={i} content={block.content as TableData} title={block.title} />
          case "code":
            return <CodeBlock key={i} content={block.content as CodeContent} title={block.title} />
          case "document":
            return <DocumentBlock key={i} content={block.content as DocumentContent} />
          case "bar-chart":
            return <BarChartBlock key={i} content={block.content as ChartData} title={block.title} />
          case "line-chart":
            return <LineChartBlock key={i} content={block.content as ChartData} title={block.title} />
          case "area-chart":
            return <AreaChartBlock key={i} content={block.content as ChartData} title={block.title} />
          case "pie-chart":
            return <PieChartBlock key={i} content={block.content as PieChartData} title={block.title} />
          case "prediction":
            return <PredictionBlock key={i} content={block.content as PredictionData} />
          case "comparison":
            return <ComparisonBlock key={i} content={block.content as ComparisonData} />
          case "metrics":
            return <MetricsBlock key={i} content={block.content as MetricsData} />
          default:
            return null
        }
      })}
    </div>
  )
}

// Example data for demo responses
export const exampleTextResponse: ResponseBlock[] = [
  {
    type: "text",
    content: "Based on my analysis of your customer data, I've identified several key insights that can help improve your retention strategy.",
  },
  {
    type: "table",
    title: "Top Customers by Revenue",
    content: {
      headers: ["Customer ID", "Name", "Revenue", "Orders", "Status"],
      rows: [
        ["C001", "Acme Corp", "$125,430", 47, "Active"],
        ["C002", "TechStart Inc", "$98,200", 32, "Active"],
        ["C003", "Global Systems", "$87,650", 28, "At Risk"],
        ["C004", "DataFlow LLC", "$76,320", 24, "Active"],
        ["C005", "InnovateCo", "$65,890", 21, "Active"],
        ["C006", "CloudFirst", "$54,210", 18, "Churned"],
        ["C007", "NextGen Solutions", "$48,300", 15, "Active"],
      ],
    },
  },
  {
    type: "code",
    title: "SQL Query Used",
    content: {
      language: "sql",
      code: `SELECT 
  customer_id,
  customer_name,
  SUM(order_total) as revenue,
  COUNT(*) as orders,
  status
FROM customers c
JOIN orders o ON c.id = o.customer_id
WHERE o.created_at >= '2024-01-01'
GROUP BY customer_id
ORDER BY revenue DESC
LIMIT 10;`,
    },
  },
  {
    type: "document",
    content: {
      title: "Analysis Summary",
      sections: [
        {
          heading: "Key Findings",
          content: "The top 10 customers account for 45% of total revenue. Customer retention rate has improved by 12% compared to last quarter.",
        },
        {
          heading: "Recommendations",
          content: "Focus on the 'At Risk' segment with targeted retention campaigns. Consider loyalty programs for top-tier customers to maintain engagement.",
        },
      ],
    },
  },
]

export const exampleAnalyticsResponse: ResponseBlock[] = [
  {
    type: "text",
    content: "Here's a comprehensive analytics overview based on your data sources:",
  },
  {
    type: "metrics",
    content: {
      metrics: [
        { label: "Total Revenue", value: "$2.4M", change: { value: 12, type: "increase" } },
        { label: "Active Users", value: "45,230", change: { value: 8, type: "increase" } },
        { label: "Conversion Rate", value: "3.2%", change: { value: 5, type: "decrease" } },
        { label: "Avg. Order Value", value: "$156", change: { value: 3, type: "increase" } },
      ],
    },
  },
  {
    type: "bar-chart",
    title: "Revenue by Quarter",
    content: {
      data: [
        { quarter: "Q1", revenue: 520000, target: 500000 },
        { quarter: "Q2", revenue: 580000, target: 550000 },
        { quarter: "Q3", revenue: 620000, target: 600000 },
        { quarter: "Q4", revenue: 680000, target: 650000 },
      ],
      xKey: "quarter",
      yKeys: ["revenue", "target"],
    },
  },
  {
    type: "line-chart",
    title: "User Growth Trend",
    content: {
      data: [
        { month: "Jan", users: 32000, sessions: 128000 },
        { month: "Feb", users: 35000, sessions: 142000 },
        { month: "Mar", users: 38500, sessions: 156000 },
        { month: "Apr", users: 41200, sessions: 168000 },
        { month: "May", users: 43800, sessions: 178000 },
        { month: "Jun", users: 45230, sessions: 185000 },
      ],
      xKey: "month",
      yKeys: ["users", "sessions"],
    },
  },
  {
    type: "pie-chart",
    title: "Revenue by Segment",
    content: {
      data: [
        { name: "Enterprise", value: 980000 },
        { name: "Mid-Market", value: 650000 },
        { name: "SMB", value: 420000 },
        { name: "Startup", value: 350000 },
      ],
    },
  },
  {
    type: "area-chart",
    title: "Daily Active Users",
    content: {
      data: [
        { day: "Mon", desktop: 4200, mobile: 3800 },
        { day: "Tue", desktop: 4500, mobile: 4100 },
        { day: "Wed", desktop: 4800, mobile: 4400 },
        { day: "Thu", desktop: 5100, mobile: 4600 },
        { day: "Fri", desktop: 4900, mobile: 4500 },
        { day: "Sat", desktop: 3800, mobile: 5200 },
        { day: "Sun", desktop: 3500, mobile: 5500 },
      ],
      xKey: "day",
      yKeys: ["desktop", "mobile"],
    },
  },
  {
    type: "prediction",
    content: {
      title: "Predicted Q1 Revenue",
      value: "$2.8M",
      confidence: 87,
      trend: "up",
      comparison: "+16.7% YoY",
    },
  },
  {
    type: "comparison",
    content: {
      title: "Channel Performance",
      items: [
        { label: "Direct Traffic", value: 45230, maxValue: 50000 },
        { label: "Organic Search", value: 32100, maxValue: 50000 },
        { label: "Paid Ads", value: 28500, maxValue: 50000 },
        { label: "Social Media", value: 18200, maxValue: 50000 },
        { label: "Referrals", value: 12800, maxValue: 50000 },
      ],
    },
  },
]

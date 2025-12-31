"use client"

import { useState, useEffect, useRef } from "react"
import { useParams, useRouter } from "next/navigation"
import { Sidebar } from "@/components/sidebar"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Columns, Plus, Copy, Clock, Zap, X, Check, Loader2 } from "lucide-react"
import AiChat from "@/components/ai-chat"
import { useAuth } from "@/lib/auth"
import { useQueryStore } from "@/lib/query-store"
import { SchemaIcon } from "@/components/schema-icon"
import { SchemaProcessingAnimation } from "@/components/schema-processing-animation"
import { api } from "@/lib/api"

const availableModels = [
  { id: "claude-sonnet-4-5", name: "Claude Sonnet 4.5", provider: "Anthropic", speed: "Fast" },
  { id: "claude-opus-4", name: "Claude Opus 4", provider: "Anthropic", speed: "Medium" },
  { id: "claude-haiku-4-5", name: "Claude Haiku 4.5", provider: "Anthropic", speed: "Very Fast" },
  { id: "gpt-4o", name: "GPT-4o", provider: "OpenAI", speed: "Fast" },
  { id: "gpt-4o-mini", name: "GPT-4o Mini", provider: "OpenAI", speed: "Fast" },
  { id: "gpt-4.5-preview", name: "GPT-4.5 Preview", provider: "OpenAI", speed: "Medium" },
  { id: "gpt-5", name: "GPT-5", provider: "OpenAI", speed: "Fast" },
]

interface UploadedFile {
  file_id: string
  filename: string
  path: string
  size: number
}

interface Message {
  role: "user" | "assistant"
  content: string
  model?: string
  tokens?: number
  time?: string
}

interface ComparePane {
  id: string
  model: string
  messages: Message[]
  isLoading: boolean
}




// Advanced Chart System with 20+ chart types and interactive tooltips
interface ChartData {
  type: string
  labels: string[]
  values: number[]
  values2?: number[]
  values3?: number[]
  title: string
  xlabel?: string
  ylabel?: string
  series?: string[]
}

function parseCharts(text: string): { content: string; charts: ChartData[] } {
  const charts: ChartData[] = []
  const chartRegex = /\[CHART:(\w+)\]([\s\S]*?)\[\/CHART\]/g
  
  let match
  while ((match = chartRegex.exec(text)) !== null) {
    const type = match[1]
    const data = match[2]
    
    const labelsMatch = data.match(/labels:\s*(.+)/)
    const valuesMatch = data.match(/values:\s*(.+)/)
    const values2Match = data.match(/values2:\s*(.+)/)
    const values3Match = data.match(/values3:\s*(.+)/)
    const titleMatch = data.match(/title:\s*(.+)/)
    const xlabelMatch = data.match(/xlabel:\s*(.+)/)
    const ylabelMatch = data.match(/ylabel:\s*(.+)/)
    const seriesMatch = data.match(/series:\s*(.+)/)
    
    if (labelsMatch && valuesMatch) {
      charts.push({
        type,
        labels: labelsMatch[1].split(',').map(s => s.trim()),
        values: valuesMatch[1].split(',').map(s => parseFloat(s.trim())),
        values2: values2Match ? values2Match[1].split(',').map(s => parseFloat(s.trim())) : undefined,
        values3: values3Match ? values3Match[1].split(',').map(s => parseFloat(s.trim())) : undefined,
        title: titleMatch ? titleMatch[1].trim() : '',
        xlabel: xlabelMatch ? xlabelMatch[1].trim() : undefined,
        ylabel: ylabelMatch ? ylabelMatch[1].trim() : undefined,
        series: seriesMatch ? seriesMatch[1].split(',').map(s => s.trim()) : undefined
      })
    }
  }
  
  const cleanContent = text.replace(chartRegex, '').trim()
  return { content: cleanContent, charts }
}

// Tooltip component
function Tooltip({ x, y, content, visible }: { x: number; y: number; content: string; visible: boolean }) {
  if (!visible) return null
  return (
    <div className="absolute bg-gray-900 text-white text-xs px-2 py-1 rounded shadow-lg pointer-events-none z-50 whitespace-nowrap"
      style={{ left: x + 10, top: y - 25 }}>
      {content}
    </div>
  )
}

function AdvancedChart({ type, labels, values, values2, values3, title, xlabel, ylabel, series }: ChartData) {
  const [tooltip, setTooltip] = useState<{ x: number; y: number; content: string; visible: boolean }>({ x: 0, y: 0, content: '', visible: false })
  
  const maxValue = Math.max(...values, ...(values2 || []), ...(values3 || []))
  const minValue = Math.min(...values.filter(v => v > 0), ...(values2 || []).filter(v => v > 0))
  const total = values.reduce((a, b) => a + b, 0)
  const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1']
  
  const showTooltip = (e: React.MouseEvent, content: string) => {
    const rect = e.currentTarget.getBoundingClientRect()
    setTooltip({ x: e.clientX - rect.left, y: e.clientY - rect.top, content, visible: true })
  }
  const hideTooltip = () => setTooltip(prev => ({ ...prev, visible: false }))

  // 1. VERTICAL BAR CHART
  if (type === 'bar') {
    return (
      <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl shadow-sm relative">
        {title && <p className="text-sm font-semibold mb-3 text-gray-700">{title}</p>}
        <div className="flex items-end gap-2 h-40 px-2">
          {values.map((value, i) => (
            <div key={i} className="flex-1 flex flex-col items-center gap-1 group cursor-pointer"
              onMouseMove={(e) => showTooltip(e, `${labels[i]}: ${value.toLocaleString()}`)}
              onMouseLeave={hideTooltip}>
              <span className="text-[10px] font-medium text-gray-600 opacity-0 group-hover:opacity-100 transition-opacity">
                {value.toLocaleString()}
              </span>
              <div className="w-full rounded-t transition-all duration-300 group-hover:opacity-80 group-hover:scale-105"
                style={{ height: `${(value / maxValue) * 100}%`, backgroundColor: colors[i % colors.length], minHeight: '4px' }}
              />
              <span className="text-[10px] text-gray-500 truncate max-w-full">{labels[i]}</span>
            </div>
          ))}
        </div>
        {ylabel && <div className="absolute left-0 top-1/2 -translate-y-1/2 -rotate-90 text-[10px] text-gray-500">{ylabel}</div>}
        {xlabel && <div className="text-center text-[10px] text-gray-500 mt-2">{xlabel}</div>}
        <Tooltip {...tooltip} />
      </div>
    )
  }

  // 2. HORIZONTAL BAR CHART
  if (type === 'hbar') {
    return (
      <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl shadow-sm relative">
        {title && <p className="text-sm font-semibold mb-3 text-gray-700">{title}</p>}
        <div className="space-y-2">
          {labels.map((label, i) => (
            <div key={i} className="flex items-center gap-2 group cursor-pointer"
              onMouseMove={(e) => showTooltip(e, `${label}: ${values[i].toLocaleString()}`)}
              onMouseLeave={hideTooltip}>
              <span className="text-xs w-24 truncate text-gray-600 text-right">{label}</span>
              <div className="flex-1 bg-gray-200 rounded-full h-6 overflow-hidden">
                <div className="h-full rounded-full transition-all duration-500 flex items-center justify-end pr-2 group-hover:opacity-80"
                  style={{ width: `${(values[i] / maxValue) * 100}%`, backgroundColor: colors[i % colors.length] }}>
                  <span className="text-[10px] text-white font-medium opacity-0 group-hover:opacity-100">{values[i].toLocaleString()}</span>
                </div>
              </div>
              <span className="text-xs w-16 font-medium text-right">{values[i].toLocaleString()}</span>
            </div>
          ))}
        </div>
        <Tooltip {...tooltip} />
      </div>
    )
  }

  // 3. GROUPED BAR CHART
  if (type === 'grouped' && values2) {
    const groupMax = Math.max(...values, ...values2)
    const seriesNames = series || ['Series 1', 'Series 2']
    return (
      <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl shadow-sm relative">
        {title && <p className="text-sm font-semibold mb-3 text-gray-700">{title}</p>}
        <div className="flex items-end gap-4 h-36 px-2">
          {labels.map((label, i) => (
            <div key={i} className="flex-1 flex flex-col items-center gap-1">
              <div className="flex items-end gap-1 h-28 w-full justify-center">
                <div className="w-5 bg-blue-500 rounded-t transition-all cursor-pointer hover:opacity-80"
                  style={{ height: `${(values[i] / groupMax) * 100}%` }}
                  onMouseMove={(e) => showTooltip(e, `${seriesNames[0]}: ${values[i].toLocaleString()}`)}
                  onMouseLeave={hideTooltip} />
                <div className="w-5 bg-emerald-500 rounded-t transition-all cursor-pointer hover:opacity-80"
                  style={{ height: `${(values2[i] / groupMax) * 100}%` }}
                  onMouseMove={(e) => showTooltip(e, `${seriesNames[1]}: ${values2[i].toLocaleString()}`)}
                  onMouseLeave={hideTooltip} />
              </div>
              <span className="text-[10px] text-gray-500 truncate max-w-full text-center">{label}</span>
            </div>
          ))}
        </div>
        <div className="flex justify-center gap-4 mt-3 text-xs">
          <div className="flex items-center gap-1"><div className="w-3 h-3 bg-blue-500 rounded" />{seriesNames[0]}</div>
          <div className="flex items-center gap-1"><div className="w-3 h-3 bg-emerald-500 rounded" />{seriesNames[1]}</div>
        </div>
        <Tooltip {...tooltip} />
      </div>
    )
  }

  // 4. STACKED BAR CHART
  if (type === 'stacked' && values2) {
    const seriesNames = series || ['Series 1', 'Series 2']
    return (
      <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl shadow-sm relative">
        {title && <p className="text-sm font-semibold mb-3 text-gray-700">{title}</p>}
        <div className="flex items-end gap-2 h-36 px-2">
          {labels.map((label, i) => {
            const stackTotal = values[i] + values2[i]
            const maxStack = Math.max(...values.map((v, j) => v + values2[j]))
            return (
              <div key={i} className="flex-1 flex flex-col items-center gap-1">
                <div className="w-full flex flex-col-reverse" style={{ height: `${(stackTotal / maxStack) * 100}%`, minHeight: '8px' }}>
                  <div className="w-full bg-blue-500 rounded-b cursor-pointer hover:opacity-80 transition-opacity"
                    style={{ height: `${(values[i] / stackTotal) * 100}%` }}
                    onMouseMove={(e) => showTooltip(e, `${seriesNames[0]}: ${values[i].toLocaleString()}`)}
                    onMouseLeave={hideTooltip} />
                  <div className="w-full bg-emerald-500 rounded-t cursor-pointer hover:opacity-80 transition-opacity"
                    style={{ height: `${(values2[i] / stackTotal) * 100}%` }}
                    onMouseMove={(e) => showTooltip(e, `${seriesNames[1]}: ${values2[i].toLocaleString()}`)}
                    onMouseLeave={hideTooltip} />
                </div>
                <span className="text-[10px] text-gray-500">{label}</span>
              </div>
            )
          })}
        </div>
        <div className="flex justify-center gap-4 mt-3 text-xs">
          <div className="flex items-center gap-1"><div className="w-3 h-3 bg-blue-500 rounded" />{seriesNames[0]}</div>
          <div className="flex items-center gap-1"><div className="w-3 h-3 bg-emerald-500 rounded" />{seriesNames[1]}</div>
        </div>
        <Tooltip {...tooltip} />
      </div>
    )
  }

  // 5. LINE CHART
  if (type === 'line') {
    const width = 320, height = 180, padding = 45
    const chartWidth = width - padding * 2, chartHeight = height - padding * 2
    const range = maxValue - minValue || 1
    const points = values.map((v, i) => ({
      x: padding + (i / Math.max(values.length - 1, 1)) * chartWidth,
      y: height - padding - ((v - minValue) / range) * chartHeight,
      value: v, label: labels[i]
    }))
    const pathD = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ')
    
    return (
      <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl shadow-sm relative">
        {title && <p className="text-sm font-semibold mb-3 text-gray-700">{title}</p>}
        <svg width={width} height={height} className="overflow-visible">
          <defs>
            <linearGradient id="lineGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.3" />
              <stop offset="100%" stopColor="#3b82f6" stopOpacity="0" />
            </linearGradient>
          </defs>
          {/* Grid */}
          {[0, 0.25, 0.5, 0.75, 1].map((t, i) => (
            <line key={i} x1={padding} x2={width - padding} y1={height - padding - t * chartHeight} y2={height - padding - t * chartHeight} stroke="#e5e7eb" strokeDasharray="4" />
          ))}
          {/* Area fill */}
          <path d={`${pathD} L ${points[points.length-1].x} ${height - padding} L ${padding} ${height - padding} Z`} fill="url(#lineGrad)" />
          {/* Line */}
          <path d={pathD} fill="none" stroke="#3b82f6" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
          {/* Points */}
          {points.map((p, i) => (
            <g key={i}>
              <circle cx={p.x} cy={p.y} r="5" fill="#3b82f6" className="cursor-pointer hover:r-7 transition-all"
                onMouseMove={(e) => showTooltip(e, `${p.label}: ${p.value.toLocaleString()}`)}
                onMouseLeave={hideTooltip} />
              <text x={p.x} y={height - padding + 15} textAnchor="middle" className="text-[9px] fill-gray-500">{p.label}</text>
            </g>
          ))}
          {/* Y-axis */}
          <text x={padding - 8} y={padding} textAnchor="end" className="text-[9px] fill-gray-400">{maxValue.toLocaleString()}</text>
          <text x={padding - 8} y={height - padding} textAnchor="end" className="text-[9px] fill-gray-400">{minValue.toLocaleString()}</text>
          {ylabel && <text x={12} y={height/2} textAnchor="middle" transform={`rotate(-90,12,${height/2})`} className="text-[10px] fill-gray-600">{ylabel}</text>}
          {xlabel && <text x={width/2} y={height - 5} textAnchor="middle" className="text-[10px] fill-gray-600">{xlabel}</text>}
        </svg>
        <Tooltip {...tooltip} />
      </div>
    )
  }

  // 6. MULTI-LINE CHART
  if (type === 'multiline' && values2) {
    const width = 320, height = 180, padding = 45
    const chartWidth = width - padding * 2, chartHeight = height - padding * 2
    const allMax = Math.max(...values, ...values2)
    const allMin = Math.min(...values, ...values2)
    const range = allMax - allMin || 1
    const seriesNames = series || ['Series 1', 'Series 2']
    
    const points1 = values.map((v, i) => ({ x: padding + (i / Math.max(values.length - 1, 1)) * chartWidth, y: height - padding - ((v - allMin) / range) * chartHeight, value: v }))
    const points2 = values2.map((v, i) => ({ x: padding + (i / Math.max(values2.length - 1, 1)) * chartWidth, y: height - padding - ((v - allMin) / range) * chartHeight, value: v }))
    
    return (
      <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl shadow-sm relative">
        {title && <p className="text-sm font-semibold mb-3 text-gray-700">{title}</p>}
        <svg width={width} height={height}>
          {/* Grid */}
          {[0, 0.5, 1].map((t, i) => <line key={i} x1={padding} x2={width-padding} y1={height-padding-t*chartHeight} y2={height-padding-t*chartHeight} stroke="#e5e7eb" strokeDasharray="4" />)}
          {/* Lines */}
          <path d={points1.map((p,i) => `${i===0?'M':'L'} ${p.x} ${p.y}`).join(' ')} fill="none" stroke="#3b82f6" strokeWidth="2" />
          <path d={points2.map((p,i) => `${i===0?'M':'L'} ${p.x} ${p.y}`).join(' ')} fill="none" stroke="#10b981" strokeWidth="2" />
          {/* Points */}
          {points1.map((p,i) => <circle key={`a${i}`} cx={p.x} cy={p.y} r="4" fill="#3b82f6" className="cursor-pointer" onMouseMove={(e)=>showTooltip(e,`${seriesNames[0]}: ${p.value}`)} onMouseLeave={hideTooltip} />)}
          {points2.map((p,i) => <circle key={`b${i}`} cx={p.x} cy={p.y} r="4" fill="#10b981" className="cursor-pointer" onMouseMove={(e)=>showTooltip(e,`${seriesNames[1]}: ${p.value}`)} onMouseLeave={hideTooltip} />)}
          {/* Labels */}
          {labels.map((l,i) => <text key={i} x={padding+(i/Math.max(labels.length-1,1))*chartWidth} y={height-padding+15} textAnchor="middle" className="text-[9px] fill-gray-500">{l}</text>)}
        </svg>
        <div className="flex justify-center gap-4 mt-2 text-xs">
          <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-blue-500"></span>{seriesNames[0]}</span>
          <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-emerald-500"></span>{seriesNames[1]}</span>
        </div>
        <Tooltip {...tooltip} />
      </div>
    )
  }

  // 7. AREA CHART
  if (type === 'area') {
    const width = 320, height = 160, padding = 45
    const chartWidth = width - padding * 2, chartHeight = height - padding * 2
    const range = maxValue - minValue || 1
    const points = values.map((v, i) => ({
      x: padding + (i / Math.max(values.length - 1, 1)) * chartWidth,
      y: height - padding - ((v - minValue) / range) * chartHeight, value: v
    }))
    
    return (
      <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl shadow-sm relative">
        {title && <p className="text-sm font-semibold mb-3 text-gray-700">{title}</p>}
        <svg width={width} height={height}>
          <defs>
            <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#10b981" stopOpacity="0.6" />
              <stop offset="100%" stopColor="#10b981" stopOpacity="0.1" />
            </linearGradient>
          </defs>
          <path d={`${points.map((p,i)=>`${i===0?'M':'L'} ${p.x} ${p.y}`).join(' ')} L ${points[points.length-1].x} ${height-padding} L ${padding} ${height-padding} Z`} fill="url(#areaGrad)" />
          <path d={points.map((p,i)=>`${i===0?'M':'L'} ${p.x} ${p.y}`).join(' ')} fill="none" stroke="#10b981" strokeWidth="2" />
          {points.map((p,i) => <circle key={i} cx={p.x} cy={p.y} r="4" fill="#10b981" className="cursor-pointer" onMouseMove={(e)=>showTooltip(e,`${labels[i]}: ${p.value}`)} onMouseLeave={hideTooltip} />)}
          {labels.map((l,i) => <text key={i} x={padding+(i/Math.max(labels.length-1,1))*chartWidth} y={height-padding+15} textAnchor="middle" className="text-[9px] fill-gray-500">{l}</text>)}
        </svg>
        <Tooltip {...tooltip} />
      </div>
    )
  }

  // 8. PIE CHART
  if (type === 'pie') {
    let currentAngle = 0
    const slices = values.map((v, i) => {
      const angle = (v / total) * 360
      const start = currentAngle
      currentAngle += angle
      return { value: v, label: labels[i], startAngle: start, angle, color: colors[i % colors.length], percent: ((v/total)*100).toFixed(1) }
    })
    
    return (
      <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl shadow-sm relative">
        {title && <p className="text-sm font-semibold mb-3 text-gray-700">{title}</p>}
        <div className="flex items-center justify-center gap-6">
          <svg width="150" height="150" viewBox="0 0 150 150">
            {slices.map((s, i) => {
              const x1 = 75 + 65 * Math.cos((s.startAngle - 90) * Math.PI / 180)
              const y1 = 75 + 65 * Math.sin((s.startAngle - 90) * Math.PI / 180)
              const x2 = 75 + 65 * Math.cos((s.startAngle + s.angle - 90) * Math.PI / 180)
              const y2 = 75 + 65 * Math.sin((s.startAngle + s.angle - 90) * Math.PI / 180)
              return (
                <path key={i} d={`M 75 75 L ${x1} ${y1} A 65 65 0 ${s.angle > 180 ? 1 : 0} 1 ${x2} ${y2} Z`}
                  fill={s.color} className="cursor-pointer hover:opacity-80 transition-all hover:scale-105 origin-center"
                  onMouseMove={(e) => showTooltip(e, `${s.label}: ${s.value.toLocaleString()} (${s.percent}%)`)}
                  onMouseLeave={hideTooltip} />
              )
            })}
          </svg>
          <div className="text-xs space-y-1.5">
            {slices.map((s, i) => (
              <div key={i} className="flex items-center gap-2 cursor-pointer hover:opacity-70"
                onMouseMove={(e) => showTooltip(e, `${s.value.toLocaleString()} (${s.percent}%)`)}
                onMouseLeave={hideTooltip}>
                <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: s.color }} />
                <span className="text-gray-600">{s.label}</span>
                <span className="text-gray-400">{s.percent}%</span>
              </div>
            ))}
          </div>
        </div>
        <Tooltip {...tooltip} />
      </div>
    )
  }

  // 9. DONUT CHART
  if (type === 'donut') {
    let currentAngle = 0
    const outerR = 65, innerR = 40
    
    return (
      <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl shadow-sm relative">
        {title && <p className="text-sm font-semibold mb-3 text-gray-700">{title}</p>}
        <div className="flex items-center justify-center gap-6">
          <svg width="150" height="150" viewBox="0 0 150 150">
            {values.map((v, i) => {
              const angle = (v / total) * 360
              const start = currentAngle
              currentAngle += angle
              const x1o = 75 + outerR * Math.cos((start - 90) * Math.PI / 180)
              const y1o = 75 + outerR * Math.sin((start - 90) * Math.PI / 180)
              const x2o = 75 + outerR * Math.cos((start + angle - 90) * Math.PI / 180)
              const y2o = 75 + outerR * Math.sin((start + angle - 90) * Math.PI / 180)
              const x1i = 75 + innerR * Math.cos((start + angle - 90) * Math.PI / 180)
              const y1i = 75 + innerR * Math.sin((start + angle - 90) * Math.PI / 180)
              const x2i = 75 + innerR * Math.cos((start - 90) * Math.PI / 180)
              const y2i = 75 + innerR * Math.sin((start - 90) * Math.PI / 180)
              return (
                <path key={i}
                  d={`M ${x1o} ${y1o} A ${outerR} ${outerR} 0 ${angle > 180 ? 1 : 0} 1 ${x2o} ${y2o} L ${x1i} ${y1i} A ${innerR} ${innerR} 0 ${angle > 180 ? 1 : 0} 0 ${x2i} ${y2i} Z`}
                  fill={colors[i % colors.length]} className="cursor-pointer hover:opacity-80"
                  onMouseMove={(e) => showTooltip(e, `${labels[i]}: ${v.toLocaleString()} (${((v/total)*100).toFixed(1)}%)`)}
                  onMouseLeave={hideTooltip} />
              )
            })}
            <text x="75" y="72" textAnchor="middle" className="text-xl font-bold fill-gray-700">{total.toLocaleString()}</text>
            <text x="75" y="88" textAnchor="middle" className="text-[10px] fill-gray-400">Total</text>
          </svg>
          <div className="text-xs space-y-1.5">
            {labels.map((l, i) => (
              <div key={i} className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: colors[i % colors.length] }} />
                <span>{l}: {values[i].toLocaleString()}</span>
              </div>
            ))}
          </div>
        </div>
        <Tooltip {...tooltip} />
      </div>
    )
  }

  // 10. GAUGE CHART
  if (type === 'gauge') {
    const pct = Math.min(100, Math.max(0, values[0]))
    const color = pct < 33 ? '#ef4444' : pct < 66 ? '#f59e0b' : '#10b981'
    const angle = (pct / 100) * 180 - 90
    
    return (
      <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl shadow-sm relative">
        {title && <p className="text-sm font-semibold mb-3 text-gray-700">{title}</p>}
        <svg width="180" height="110" viewBox="0 0 180 110" className="mx-auto">
          <path d="M 20 90 A 70 70 0 0 1 160 90" fill="none" stroke="#e5e7eb" strokeWidth="14" strokeLinecap="round" />
          <path d={`M 20 90 A 70 70 0 ${pct > 50 ? 1 : 0} 1 ${90 + 70 * Math.cos(angle * Math.PI / 180)} ${90 + 70 * Math.sin(angle * Math.PI / 180)}`}
            fill="none" stroke={color} strokeWidth="14" strokeLinecap="round" />
          <text x="90" y="85" textAnchor="middle" className="text-3xl font-bold" fill={color}>{pct.toFixed(1)}%</text>
          <text x="90" y="105" textAnchor="middle" className="text-[11px] fill-gray-500">{labels[0] || 'Score'}</text>
        </svg>
      </div>
    )
  }

  // 11. SCATTER PLOT
  if (type === 'scatter' && values2) {
    const width = 300, height = 200, padding = 40
    const xMax = Math.max(...values), xMin = Math.min(...values)
    const yMax = Math.max(...values2), yMin = Math.min(...values2)
    const xRange = xMax - xMin || 1, yRange = yMax - yMin || 1
    
    return (
      <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl shadow-sm relative">
        {title && <p className="text-sm font-semibold mb-3 text-gray-700">{title}</p>}
        <svg width={width} height={height}>
          {/* Grid */}
          {[0, 0.5, 1].map((t, i) => (
            <g key={i}>
              <line x1={padding} x2={width-padding} y1={height-padding-t*(height-2*padding)} y2={height-padding-t*(height-2*padding)} stroke="#e5e7eb" strokeDasharray="4" />
              <line x1={padding+t*(width-2*padding)} y1={padding} x2={padding+t*(width-2*padding)} y2={height-padding} stroke="#e5e7eb" strokeDasharray="4" />
            </g>
          ))}
          {/* Axes */}
          <line x1={padding} y1={height-padding} x2={width-padding} y2={height-padding} stroke="#9ca3af" />
          <line x1={padding} y1={padding} x2={padding} y2={height-padding} stroke="#9ca3af" />
          {/* Points */}
          {values.map((x, i) => {
            const px = padding + ((x - xMin) / xRange) * (width - 2*padding)
            const py = height - padding - ((values2[i] - yMin) / yRange) * (height - 2*padding)
            return (
              <circle key={i} cx={px} cy={py} r="6" fill={colors[i % colors.length]} fillOpacity="0.7"
                className="cursor-pointer hover:r-8 transition-all"
                onMouseMove={(e) => showTooltip(e, `${labels[i] || `Point ${i+1}`}: (${x}, ${values2[i]})`)}
                onMouseLeave={hideTooltip} />
            )
          })}
          {xlabel && <text x={width/2} y={height-5} textAnchor="middle" className="text-[10px] fill-gray-600">{xlabel}</text>}
          {ylabel && <text x={12} y={height/2} textAnchor="middle" transform={`rotate(-90,12,${height/2})`} className="text-[10px] fill-gray-600">{ylabel}</text>}
        </svg>
        <Tooltip {...tooltip} />
      </div>
    )
  }

  // 12. RADAR/SPIDER CHART
  if (type === 'radar') {
    const cx = 120, cy = 100, r = 70
    const n = labels.length
    const angleStep = (2 * Math.PI) / n
    const points = values.map((v, i) => {
      const angle = angleStep * i - Math.PI / 2
      const dist = (v / maxValue) * r
      return { x: cx + dist * Math.cos(angle), y: cy + dist * Math.sin(angle), value: v, label: labels[i] }
    })
    const pathD = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ') + ' Z'
    
    return (
      <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl shadow-sm relative">
        {title && <p className="text-sm font-semibold mb-3 text-gray-700">{title}</p>}
        <svg width="240" height="200">
          {/* Grid circles */}
          {[0.25, 0.5, 0.75, 1].map((t, i) => (
            <polygon key={i} points={Array.from({length: n}, (_, j) => {
              const angle = angleStep * j - Math.PI / 2
              return `${cx + t * r * Math.cos(angle)},${cy + t * r * Math.sin(angle)}`
            }).join(' ')} fill="none" stroke="#e5e7eb" />
          ))}
          {/* Axis lines */}
          {labels.map((_, i) => {
            const angle = angleStep * i - Math.PI / 2
            return <line key={i} x1={cx} y1={cy} x2={cx + r * Math.cos(angle)} y2={cy + r * Math.sin(angle)} stroke="#d1d5db" />
          })}
          {/* Data polygon */}
          <polygon points={points.map(p => `${p.x},${p.y}`).join(' ')} fill="#3b82f6" fillOpacity="0.3" stroke="#3b82f6" strokeWidth="2" />
          {/* Points & Labels */}
          {points.map((p, i) => (
            <g key={i}>
              <circle cx={p.x} cy={p.y} r="4" fill="#3b82f6" className="cursor-pointer"
                onMouseMove={(e) => showTooltip(e, `${p.label}: ${p.value}`)}
                onMouseLeave={hideTooltip} />
              <text x={cx + (r + 15) * Math.cos(angleStep * i - Math.PI/2)} y={cy + (r + 15) * Math.sin(angleStep * i - Math.PI/2)}
                textAnchor="middle" className="text-[9px] fill-gray-600">{p.label}</text>
            </g>
          ))}
        </svg>
        <Tooltip {...tooltip} />
      </div>
    )
  }

  // 13. HEATMAP
  if (type === 'heatmap' && values2) {
    const rows = labels.length
    const cols = values.length / rows
    const cellW = 40, cellH = 30
    
    return (
      <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl shadow-sm relative overflow-x-auto">
        {title && <p className="text-sm font-semibold mb-3 text-gray-700">{title}</p>}
        <div className="inline-grid gap-1" style={{ gridTemplateColumns: `repeat(${cols}, ${cellW}px)` }}>
          {values.map((v, i) => {
            const intensity = (v - minValue) / (maxValue - minValue || 1)
            const bg = `rgba(59, 130, 246, ${0.2 + intensity * 0.8})`
            return (
              <div key={i} className="flex items-center justify-center text-[10px] font-medium cursor-pointer hover:scale-105 transition-transform rounded"
                style={{ width: cellW, height: cellH, backgroundColor: bg, color: intensity > 0.5 ? 'white' : '#374151' }}
                onMouseMove={(e) => showTooltip(e, `Value: ${v.toLocaleString()}`)}
                onMouseLeave={hideTooltip}>
                {v.toFixed(1)}
              </div>
            )
          })}
        </div>
        <Tooltip {...tooltip} />
      </div>
    )
  }

  // 14. WATERFALL CHART
  if (type === 'waterfall') {
    let running = 0
    const bars = values.map((v, i) => {
      const prev = running
      running += v
      return { label: labels[i], value: v, start: v >= 0 ? prev : running, height: Math.abs(v), isPositive: v >= 0 }
    })
    const allValues = bars.flatMap(b => [b.start, b.start + b.height])
    const wfMax = Math.max(...allValues), wfMin = Math.min(...allValues)
    const range = wfMax - wfMin || 1
    
    return (
      <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl shadow-sm relative">
        {title && <p className="text-sm font-semibold mb-3 text-gray-700">{title}</p>}
        <div className="flex items-end gap-1 h-40 px-2">
          {bars.map((b, i) => (
            <div key={i} className="flex-1 flex flex-col items-center relative h-full">
              <div className="absolute w-full cursor-pointer"
                style={{
                  bottom: `${((b.start - wfMin) / range) * 100}%`,
                  height: `${(b.height / range) * 100}%`,
                  backgroundColor: b.isPositive ? '#10b981' : '#ef4444'
                }}
                onMouseMove={(e) => showTooltip(e, `${b.label}: ${b.value >= 0 ? '+' : ''}${b.value.toLocaleString()}`)}
                onMouseLeave={hideTooltip}>
                <span className="absolute -top-4 left-1/2 -translate-x-1/2 text-[9px] font-medium">{b.value >= 0 ? '+' : ''}{b.value}</span>
              </div>
              <span className="absolute bottom-0 translate-y-full text-[9px] text-gray-500 pt-1">{b.label}</span>
            </div>
          ))}
        </div>
        <Tooltip {...tooltip} />
      </div>
    )
  }

  // 15. FUNNEL CHART
  if (type === 'funnel') {
    const maxWidth = 200
    return (
      <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl shadow-sm relative">
        {title && <p className="text-sm font-semibold mb-3 text-gray-700">{title}</p>}
        <div className="flex flex-col items-center gap-1">
          {values.map((v, i) => {
            const width = (v / maxValue) * maxWidth
            const pct = ((v / values[0]) * 100).toFixed(1)
            return (
              <div key={i} className="flex items-center gap-3 cursor-pointer group"
                onMouseMove={(e) => showTooltip(e, `${labels[i]}: ${v.toLocaleString()} (${pct}%)`)}
                onMouseLeave={hideTooltip}>
                <span className="text-[10px] text-gray-500 w-20 text-right">{labels[i]}</span>
                <div className="h-8 rounded transition-all group-hover:opacity-80"
                  style={{ width, backgroundColor: colors[i % colors.length] }} />
                <span className="text-[10px] font-medium w-16">{v.toLocaleString()}</span>
              </div>
            )
          })}
        </div>
        <Tooltip {...tooltip} />
      </div>
    )
  }

  // 16. TREEMAP
  if (type === 'treemap') {
    const sorted = values.map((v, i) => ({ value: v, label: labels[i], color: colors[i % colors.length] })).sort((a, b) => b.value - a.value)
    return (
      <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl shadow-sm relative">
        {title && <p className="text-sm font-semibold mb-3 text-gray-700">{title}</p>}
        <div className="flex flex-wrap gap-1" style={{ width: 280, height: 160 }}>
          {sorted.map((item, i) => {
            const pct = (item.value / total) * 100
            return (
              <div key={i} className="flex items-center justify-center text-white text-[10px] font-medium cursor-pointer hover:opacity-80 rounded"
                style={{ flexBasis: `${pct}%`, flexGrow: 1, minWidth: 40, minHeight: 30, backgroundColor: item.color }}
                onMouseMove={(e) => showTooltip(e, `${item.label}: ${item.value.toLocaleString()} (${pct.toFixed(1)}%)`)}
                onMouseLeave={hideTooltip}>
                {pct > 8 && item.label}
              </div>
            )
          })}
        </div>
        <Tooltip {...tooltip} />
      </div>
    )
  }

  // 17. BOX PLOT (simplified)
  if (type === 'boxplot') {
    const sorted = [...values].sort((a, b) => a - b)
    const q1 = sorted[Math.floor(sorted.length * 0.25)]
    const median = sorted[Math.floor(sorted.length * 0.5)]
    const q3 = sorted[Math.floor(sorted.length * 0.75)]
    const min = sorted[0], max = sorted[sorted.length - 1]
    const range = max - min || 1
    const width = 200, height = 100, padding = 30
    
    return (
      <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl shadow-sm relative">
        {title && <p className="text-sm font-semibold mb-3 text-gray-700">{title}</p>}
        <svg width={width} height={height}>
          {/* Whiskers */}
          <line x1={padding + ((min - min) / range) * (width - 2*padding)} y1={50} x2={padding + ((q1 - min) / range) * (width - 2*padding)} y2={50} stroke="#6b7280" strokeWidth="2" />
          <line x1={padding + ((q3 - min) / range) * (width - 2*padding)} y1={50} x2={padding + ((max - min) / range) * (width - 2*padding)} y2={50} stroke="#6b7280" strokeWidth="2" />
          {/* Box */}
          <rect x={padding + ((q1 - min) / range) * (width - 2*padding)} y={30} width={((q3 - q1) / range) * (width - 2*padding)} height={40}
            fill="#3b82f6" fillOpacity="0.5" stroke="#3b82f6" strokeWidth="2" className="cursor-pointer"
            onMouseMove={(e) => showTooltip(e, `Q1: ${q1}, Median: ${median}, Q3: ${q3}`)}
            onMouseLeave={hideTooltip} />
          {/* Median line */}
          <line x1={padding + ((median - min) / range) * (width - 2*padding)} y1={30} x2={padding + ((median - min) / range) * (width - 2*padding)} y2={70} stroke="#1d4ed8" strokeWidth="3" />
          {/* Min/Max labels */}
          <text x={padding} y={80} className="text-[9px] fill-gray-500">{min}</text>
          <text x={width - padding} y={80} textAnchor="end" className="text-[9px] fill-gray-500">{max}</text>
        </svg>
        <div className="text-center text-[10px] text-gray-500">Min: {min} | Q1: {q1} | Median: {median} | Q3: {q3} | Max: {max}</div>
        <Tooltip {...tooltip} />
      </div>
    )
  }

  // 18. HISTOGRAM
  if (type === 'histogram') {
    return (
      <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl shadow-sm relative">
        {title && <p className="text-sm font-semibold mb-3 text-gray-700">{title}</p>}
        <div className="flex items-end gap-0.5 h-32 px-2">
          {values.map((v, i) => (
            <div key={i} className="flex-1 bg-blue-500 hover:bg-blue-600 transition-colors cursor-pointer rounded-t"
              style={{ height: `${(v / maxValue) * 100}%`, minHeight: v > 0 ? 2 : 0 }}
              onMouseMove={(e) => showTooltip(e, `${labels[i]}: ${v}`)}
              onMouseLeave={hideTooltip} />
          ))}
        </div>
        <div className="flex justify-between text-[9px] text-gray-500 mt-1 px-2">
          <span>{labels[0]}</span>
          <span>{labels[labels.length - 1]}</span>
        </div>
        {xlabel && <div className="text-center text-[10px] text-gray-500 mt-1">{xlabel}</div>}
        <Tooltip {...tooltip} />
      </div>
    )
  }

  // 19. BULLET CHART
  if (type === 'bullet' && values2) {
    const target = values2[0]
    const actual = values[0]
    const max = Math.max(target, actual) * 1.2
    
    return (
      <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl shadow-sm relative">
        {title && <p className="text-sm font-semibold mb-3 text-gray-700">{title}</p>}
        <div className="relative h-10 bg-gray-200 rounded overflow-hidden">
          {/* Ranges */}
          <div className="absolute inset-y-0 left-0 bg-gray-300" style={{ width: '60%' }} />
          <div className="absolute inset-y-0 left-0 bg-gray-400" style={{ width: '30%' }} />
          {/* Actual bar */}
          <div className="absolute top-2 bottom-2 left-0 bg-blue-600 rounded cursor-pointer"
            style={{ width: `${(actual / max) * 100}%` }}
            onMouseMove={(e) => showTooltip(e, `Actual: ${actual}`)}
            onMouseLeave={hideTooltip} />
          {/* Target line */}
          <div className="absolute top-1 bottom-1 w-1 bg-red-500" style={{ left: `${(target / max) * 100}%` }}
            onMouseMove={(e) => showTooltip(e, `Target: ${target}`)}
            onMouseLeave={hideTooltip} />
        </div>
        <div className="flex justify-between text-[10px] text-gray-500 mt-1">
          <span>Actual: {actual}</span>
          <span>Target: {target}</span>
        </div>
        <Tooltip {...tooltip} />
      </div>
    )
  }

  // 20. PROGRESS/METRIC CARDS
  if (type === 'metrics') {
    return (
      <div className="my-4 grid grid-cols-2 gap-3">
        {labels.map((label, i) => {
          const pct = values2 ? ((values[i] / values2[i]) * 100).toFixed(1) : null
          return (
            <div key={i} className="p-3 bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl shadow-sm cursor-pointer hover:shadow-md transition-shadow"
              onMouseMove={(e) => showTooltip(e, `${label}: ${values[i].toLocaleString()}${pct ? ` (${pct}% of target)` : ''}`)}
              onMouseLeave={hideTooltip}>
              <div className="text-[10px] text-gray-500 uppercase tracking-wide">{label}</div>
              <div className="text-xl font-bold text-gray-800">{values[i].toLocaleString()}</div>
              {pct && (
                <div className="mt-1 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                  <div className="h-full bg-blue-500 rounded-full" style={{ width: `${Math.min(100, parseFloat(pct))}%` }} />
                </div>
              )}
            </div>
          )
        })}
        <Tooltip {...tooltip} />
      </div>
    )
  }

  // Default fallback - simple bar
  return (
    <div className="my-4 p-4 bg-gray-100 rounded-lg">
      {title && <p className="text-sm font-semibold mb-2">{title}</p>}
      <div className="text-xs text-gray-500">Chart type "{type}" - {labels.join(', ')}: {values.join(', ')}</div>
    </div>
  )
}


function getUserInitials(name?: string): string {
  if (!name) return "U"
  return name.split(" ").map(n => n[0]).join("").toUpperCase().slice(0, 2)
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
}

function generateWelcomeMessage(files: UploadedFile[], queryName: string): string {
  if (files.length === 0) {
    return "Welcome to \"" + queryName + "\"! Please select data sources to begin analysis."
  }

  let msg = "**Welcome to \"" + queryName + "\"**\n\n"
  msg += "I've loaded " + files.length + " data source" + (files.length > 1 ? "s" : "") + " for analysis:\n\n"

  files.forEach((file) => {
    msg += "📄 **" + file.filename + "**\n"
    msg += "   • Size: " + formatFileSize(file.size) + "\n"
    msg += "   • Ready for analysis\n\n"
  })

  msg += "**What I can help you with:**\n"
  msg += "• Data exploration and statistical analysis\n"
  msg += "• Pattern recognition and trend identification\n"
  msg += "• Anomaly detection and data quality assessment\n"
  msg += "• Business insights and recommendations\n"
  msg += "• Custom queries and aggregations\n\n"
  msg += "Ask me anything about your data!"

  return msg
}

function renderFormattedText(text: string, keyPrefix: string): React.ReactNode[] {
  const parts: React.ReactNode[] = []
  const boldRegex = /\*\*(.+?)\*\*/g
  let lastIndex = 0
  let match
  let partIndex = 0

  while ((match = boldRegex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(<span key={keyPrefix + "-" + partIndex++}>{text.slice(lastIndex, match.index)}</span>)
    }
    parts.push(<strong key={keyPrefix + "-" + partIndex++} className="font-semibold">{match[1]}</strong>)
    lastIndex = match.index + match[0].length
  }
  
  if (lastIndex < text.length) {
    parts.push(<span key={keyPrefix + "-" + partIndex++}>{text.slice(lastIndex)}</span>)
  }
  
  return parts.length > 0 ? parts : [<span key={keyPrefix + "-0"}>{text}</span>]
}

function MessageBubble({ message, userName, compact = false }: { message: Message; userName?: string; compact?: boolean }) {
  const isUser = message.role === "user"
  const showFooter = !isUser && message.tokens && message.tokens > 0
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const renderContent = () => {
    const { content: cleanContent, charts } = parseCharts(message.content)
    const lines = cleanContent.split("\n")
    const result: React.ReactNode[] = []
    let tableLines: string[] = []
    let listItems: string[] = []
    let resultIndex = 0

    const renderTable = (tLines: string[]) => {
      const rows = tLines.filter(l => l.includes("|") && !l.match(/^[\s|\-:]+$/))
      if (rows.length < 1) return null
      const parseRow = (line: string) => line.split("|").map(c => c.trim()).filter(c => c)
      const headers = parseRow(rows[0])
      const dataRows = rows.slice(1).map(parseRow)
      return (
        <div key={"table-" + resultIndex++} className="overflow-x-auto my-3">
          <table className="min-w-full border border-gray-200 rounded-lg text-xs">
            <thead className="bg-gray-100">
              <tr>{headers.map((h, i) => <th key={i} className="px-3 py-2 text-left font-semibold border-b border-gray-200">{renderFormattedText(h, "th-" + i)}</th>)}</tr>
            </thead>
            <tbody>
              {dataRows.map((row, i) => (
                <tr key={i} className={i % 2 === 0 ? "bg-white" : "bg-gray-50"}>
                  {row.map((cell, j) => <td key={j} className="px-3 py-2 border-b border-gray-200">{renderFormattedText(cell, "td-" + i + "-" + j)}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )
    }

    const renderList = (items: string[]) => {
      return (
        <ul key={"list-" + resultIndex++} className="list-disc list-inside my-2 space-y-1">
          {items.map((item, i) => {
            const cleanItem = item.replace(/^[\s]*[•\-\*]\s*/, '')
            return <li key={i} className="text-sm">{renderFormattedText(cleanItem, "li-" + i)}</li>
          })}
        </ul>
      )
    }

    const flushList = () => {
      if (listItems.length > 0) {
        result.push(renderList(listItems))
        listItems = []
      }
    }

    const flushTable = () => {
      if (tableLines.length > 0) {
        const tbl = renderTable(tableLines)
        if (tbl) result.push(tbl)
        tableLines = []
      }
    }

    lines.forEach((line, i) => {
      const trimmedLine = line.trim()
      
      if (line.includes("|")) {
        flushList()
        tableLines.push(line)
        return
      }
      
      if (tableLines.length > 0) {
        flushTable()
      }

      if (/^[\s]*[•\-\*]\s+/.test(line)) {
        listItems.push(line)
        return
      }

      flushList()

      if (trimmedLine) {
        // Handle ### headers - convert to bold
        if (trimmedLine.startsWith("#") || (trimmedLine.startsWith("**") && (trimmedLine.endsWith("**") || trimmedLine.endsWith(":**")))) {
          // Remove ### and ** from headers
          const headerText = trimmedLine.replace(/^#+\s*/, '').replace(/^\*\*/, '').replace(/\*\*:?$/, '').trim()
          result.push(
            <p key={"h-" + resultIndex++} className="mb-2 mt-3 font-semibold text-base">
              {headerText}
            </p>
          )
        } else {
          result.push(
            <p key={"p-" + resultIndex++} className="mb-2 leading-relaxed">
              {renderFormattedText(line, "text-" + i)}
            </p>
          )
        }
      } else if (result.length > 0) {
        result.push(<div key={"space-" + resultIndex++} className="h-2" />)
      }
    })

    flushTable()
    flushList()
    
    // Add charts at the end
    charts.forEach((chart, i) => {
      result.push(<AdvancedChart key={"chart-" + i} {...chart} />)
    })

    return result
  }

  return (
    <div className={"flex w-full " + (isUser ? "justify-end" : "justify-start")}>
      <div className={"flex gap-2 sm:gap-3 " + (compact ? "max-w-full" : "max-w-[85%]") + " " + (isUser ? "flex-row-reverse" : "flex-row")}>
        <div
          className={"flex-shrink-0 rounded-full flex items-center justify-center " + 
            (compact ? "w-6 h-6 " : "w-7 h-7 sm:w-8 sm:h-8 ") +
            (isUser ? "bg-black text-white" : "bg-black text-white")}
        >
          {isUser ? (
            <span className={compact ? "text-[9px] font-medium" : "text-[10px] sm:text-xs font-medium"}>{getUserInitials(userName)}</span>
          ) : (
            <span className={compact ? "text-[9px] font-bold" : "text-[10px] sm:text-xs font-bold"}>SL</span>
          )}
        </div>

        <div className="min-w-0 flex-1">
          <div
            className={"rounded-2xl bg-card border border-border " + (compact ? "px-2 py-1.5" : "px-3 sm:px-4 py-2 sm:py-3")}
          >
            <div className="text-xs sm:text-sm">
              {renderContent()}
            </div>
            {showFooter && (
              <div className={"flex flex-wrap items-center mt-2 pt-2 border-t border-border/50 text-muted-foreground " + 
                (compact ? "gap-1.5 text-[9px]" : "gap-2 sm:gap-3 text-[10px] sm:text-xs")}>
                {!compact && (
                  <Badge variant="outline" className="text-[10px] sm:text-xs">
                    {message.model}
                  </Badge>
                )}
                <span className="flex items-center gap-1">
                  <Zap className={compact ? "h-2 w-2" : "h-2.5 w-2.5 sm:h-3 sm:w-3"} />
                  {message.tokens}
                </span>
                <span className="flex items-center gap-1">
                  <Clock className={compact ? "h-2 w-2" : "h-2.5 w-2.5 sm:h-3 sm:w-3"} />
                  {message.time}
                </span>
                <Button variant="ghost" size="sm" className={"ml-auto " + (compact ? "h-4 px-1" : "h-5 sm:h-6 px-1.5 sm:px-2")} onClick={handleCopy}>
                  {copied ? <Check className={compact ? "h-2 w-2 text-green-500" : "h-2.5 w-2.5 sm:h-3 sm:w-3 text-green-500"} /> : <Copy className={compact ? "h-2 w-2" : "h-2.5 w-2.5 sm:h-3 sm:w-3"} />}
                </Button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function TypingIndicator() {
  return (
    <div className="flex justify-start">
      <div className="inline-flex gap-2 sm:gap-3">
        <div className="flex-shrink-0 w-7 h-7 sm:w-8 sm:h-8 rounded-full bg-white border border-border flex items-center justify-center">
          <span className="text-[10px] sm:text-xs font-bold">SL</span>
        </div>
        <div className="bg-card border border-border rounded-2xl px-4 py-3">
          <div className="flex gap-1">
            <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" />
            <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce [animation-delay:0.2s]" />
            <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce [animation-delay:0.4s]" />
          </div>
        </div>
      </div>
    </div>
  )
}

function TrainingIndicator({ progress }: { progress: { epoch: number, epochs: number, batch: number, batches: number, accuracy: number, loss: number, eta: string, status: string } }) {
  const progressPercent = progress.epochs > 0 ? (progress.epoch / progress.epochs) * 100 : 0
  
  return (
    <div className="flex justify-start w-full">
      <div className="inline-flex gap-2 sm:gap-3 w-full max-w-2xl">
        <div className="flex-shrink-0 w-7 h-7 sm:w-8 sm:h-8 rounded-full bg-gradient-to-br from-primary/20 to-primary/5 border border-primary/30 flex items-center justify-center">
          <span className="text-[10px] sm:text-xs font-bold text-primary">SL</span>
        </div>
        <div className="flex-1 bg-card border border-border rounded-2xl p-5 space-y-5">
          {/* Header with spinner */}
          <div className="text-center">
            <Loader2 className="h-10 w-10 text-primary mx-auto mb-3 animate-spin" />
            <h3 className="text-base font-semibold text-foreground">Fine-tuning...</h3>
            <p className="text-sm text-muted-foreground mt-1">Training... Epoch {progress.epoch}/{progress.epochs}</p>
          </div>
          
          {/* Progress bar */}
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Progress</span>
              <span className="font-medium">{progressPercent.toFixed(0)}%</span>
            </div>
            <div className="h-2 bg-secondary rounded-full overflow-hidden">
              <div 
                className="h-full bg-primary transition-all duration-500 ease-out"
                style={{ width: `${progressPercent}%` }}
              />
            </div>
          </div>
          
          {/* Stats Grid */}
          <div className="grid grid-cols-4 gap-2 text-center">
            <div className="p-2 bg-secondary/50 rounded-lg">
              <div className="text-lg font-bold">{progress.epoch}/{progress.epochs}</div>
              <div className="text-xs text-muted-foreground">Epoch</div>
            </div>
            <div className="p-2 bg-secondary/50 rounded-lg">
              <div className="text-lg font-bold">{progress.loss.toFixed(3)}</div>
              <div className="text-xs text-muted-foreground">Loss</div>
            </div>
            <div className="p-2 bg-secondary/50 rounded-lg">
              <div className="text-lg font-bold text-green-600">{progress.accuracy.toFixed(1)}%</div>
              <div className="text-xs text-muted-foreground">Accuracy</div>
            </div>
            <div className="p-2 bg-secondary/50 rounded-lg">
              <div className="text-lg font-bold text-blue-500">{progress.eta || "..."}</div>
              <div className="text-xs text-muted-foreground">ETA</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default function PlaygroundQueryPage() {
  const params = useParams()
  const router = useRouter()
  const { user } = useAuth()
  const queryId = params.id as string

  const { getQuery, isLoaded } = useQueryStore()
  const currentQuery = getQuery(queryId)

  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([])
  const [selectedFiles, setSelectedFiles] = useState<UploadedFile[]>([])
  const [compareMode, setCompareMode] = useState(false)
  const [showCompareModal, setShowCompareModal] = useState(false)
  const [compareResults, setCompareResults] = useState<{model: string, modelName: string, content: string, tokens: number, time: string}[]>([])
  const [messages, setMessages] = useState<Message[]>([])
  const [hasInitializedChat, setHasInitializedChat] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [selectedModel, setSelectedModel] = useState("gpt-4o")
  const [trainingProgress, setTrainingProgress] = useState({ epoch: 0, epochs: 10, batch: 0, batches: 100, accuracy: 0, loss: 0, eta: "", status: "" })
  const pollRef = useRef<NodeJS.Timeout | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  
  const [panes, setPanes] = useState<ComparePane[]>([
    { id: "1", model: "gpt-4o", messages: [], isLoading: false },
    { id: "2", model: "gpt-4o-mini", messages: [], isLoading: false },
  ])

  useEffect(() => {
    loadFiles()
  }, [])

  // Training progress polling
  useEffect(() => {
    if (currentQuery?.isTraining) {
      const poll = async () => {
        try {
          const progress = await api.getTrainingProgress(currentQuery.id)
          if (progress.status === "training") {
            setTrainingProgress({
              epoch: progress.epoch || 0,
              epochs: progress.epochs || 10,
              batch: progress.batch || 0,
              batches: progress.batches || 100,
              accuracy: progress.accuracy || 0,
              loss: progress.loss || 0,
              eta: progress.eta || "",
              status: "training"
            })
          } else if (progress.status === "completed") {
            setTrainingProgress(prev => ({ ...prev, status: "completed", accuracy: progress.accuracy }))
            if (pollRef.current) {
              clearInterval(pollRef.current)
              pollRef.current = null
            }
          }
        } catch (e) {
          console.error("Training progress error:", e)
        }
      }
      poll()
      pollRef.current = setInterval(poll, 1000)
      return () => {
        if (pollRef.current) {
          clearInterval(pollRef.current)
          pollRef.current = null
        }
      }
    }
  }, [currentQuery?.isTraining])

  useEffect(() => {
    if (currentQuery?.model) {
      setSelectedModel(currentQuery.model)
    }
  }, [currentQuery])

  const loadFiles = async () => {
    try {
      const data = await api.getUploadedFiles()
      setUploadedFiles(data.files || [])
    } catch (error) {
      console.error("Failed to load files:", error)
    }
  }


  useEffect(() => {
    if (currentQuery && uploadedFiles.length > 0 && !hasInitializedChat) {
      let files = currentQuery.dataSources
        .map(id => uploadedFiles.find(f => f.file_id === id))
        .filter(Boolean) as UploadedFile[]
      
      // Fallback: if no matching files, use most recent uploaded file
      if (files.length === 0 && uploadedFiles.length > 0) {
        files = [uploadedFiles[uploadedFiles.length - 1]]
        console.log("Using fallback file:", files[0].filename)
      }
      
      setSelectedFiles(files)
      
      // Try to load messages from DB first
      if (files.length > 0) {
        fetch("/api/messages?query_id=" + queryId, { credentials: "include" })
          .then(res => res.json())
          .then(data => {
            if (data.messages && data.messages.length > 0) {
              const loadedMessages = data.messages.map((m: any) => {
                console.log("Loading message:", m.id, "role:", m.role)
                return {
                  id: m.id,
                  role: m.role,
                  content: m.content,
                  isLoading: false,
                  model: m.model,
                  tokens: m.tokens
                }
              })
              setMessages(loadedMessages)
            } else {
              const welcomeMsg = generateWelcomeMessage(files, currentQuery.name)
              setMessages([{ id: "welcome", role: "assistant", content: welcomeMsg, isLoading: false }])
            }
            setHasInitializedChat(true)
          })
          .catch(() => {
            const welcomeMsg = generateWelcomeMessage(files, currentQuery.name)
            setMessages([{ id: "welcome", role: "assistant", content: welcomeMsg, isLoading: false }])
            setHasInitializedChat(true)
          })
      } else {
        const welcomeMsg = generateWelcomeMessage(files, currentQuery.name)
        setMessages([{ id: "welcome", role: "assistant", content: welcomeMsg, isLoading: false }])
        setHasInitializedChat(true)
      }
    }
  }, [currentQuery, uploadedFiles, hasInitializedChat])

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages, isLoading])

  const primaryFile = selectedFiles[0]
  const fileNames = selectedFiles.map(f => f.filename)

  const buildDataContext = () => {
    if (selectedFiles.length === 0) return ""
    let context = ""
    selectedFiles.forEach(file => {
      context += "- File: " + file.filename + "\n"
      context += "- Size: " + formatFileSize(file.size) + "\n"
    })
    return context
  }

  const handleSingleSend = async (message: string) => {
    if (!message.trim() || isLoading) return

    const userMessage: Message = { role: "user", content: message }
    setMessages(prev => [...prev, userMessage])
    setIsLoading(true)

    const startTime = Date.now()
    let streamContent = ""

    // Add empty assistant message for streaming
    const assistantMessage: Message = {
      role: "assistant",
      content: "",
      model: availableModels.find(m => m.id === selectedModel)?.name || selectedModel,
    }
    setMessages(prev => [...prev, assistantMessage])

    try {
      const isClaudeModel = selectedModel.startsWith("claude")
      
      if (isClaudeModel) {
        const response = await api.chat({
          message: message,
          file_id: primaryFile?.file_id || "",
          query_id: queryId,
          filename: primaryFile?.filename || "Unknown",
          model: selectedModel,
          data_context: buildDataContext(),
        })
        const endTime = Date.now()
        const timeTaken = ((endTime - startTime) / 1000).toFixed(1)
        setMessages(prev => {
          const newMessages = [...prev]
          newMessages[newMessages.length - 1] = {
            ...newMessages[newMessages.length - 1],
            content: response.response,
            time: timeTaken + "s",
            tokens: response.tokens,
          }
          return newMessages
        })
        setIsLoading(false)
        return
      }
      
      await api.chatStream(
          {
            message: message,
            file_id: primaryFile?.file_id || "",
            query_id: queryId,
            filename: primaryFile?.filename || "Unknown",
            model: selectedModel,
            data_context: buildDataContext(),
          },
          (chunk) => {
            streamContent += chunk
            setMessages(prev => {
              const newMessages = [...prev]
              newMessages[newMessages.length - 1] = {
                ...newMessages[newMessages.length - 1],
                content: streamContent,
              }
              return newMessages
            })
          },
          () => {
            const endTime = Date.now()
            const timeTaken = ((endTime - startTime) / 1000).toFixed(1)
            setMessages(prev => {
              const newMessages = [...prev]
              newMessages[newMessages.length - 1] = {
                ...newMessages[newMessages.length - 1],
                time: timeTaken + "s",
                tokens: Math.round(streamContent.length / 4),
              }
              return newMessages
            })
            setIsLoading(false)
          }
        )
    } catch (error) {
      console.error("Chat error:", error)
      setMessages(prev => {
        const newMessages = [...prev]
        newMessages[newMessages.length - 1] = {
          role: "assistant",
          content: "I apologize, but I encountered an error processing your request. Please try again.",
          model: selectedModel,
        }
        return newMessages
      })
      setIsLoading(false)
    }
  }

  const handleCompareSend = async (message: string) => {
    if (!message.trim()) return

    setPanes(current =>
      current.map(pane => ({
        ...pane,
        messages: [...pane.messages, { role: "user", content: message }],
        isLoading: true,
      }))
    )

    const responses = await Promise.all(
      panes.map(async (pane) => {
        const startTime = Date.now()
        try {
          const response = await api.chat({
            message: message,
            file_id: primaryFile?.file_id || "",
          query_id: queryId,
            filename: primaryFile?.filename || "Unknown",
            model: pane.model,
            data_context: buildDataContext(),
          })
          const endTime = Date.now()
          return {
            paneId: pane.id,
            content: response.response,
            tokens: response.tokens,
            time: ((endTime - startTime) / 1000).toFixed(1) + "s",
            model: availableModels.find(m => m.id === pane.model)?.name || pane.model,
          }
        } catch (error) {
          return {
            paneId: pane.id,
            content: "Error processing request",
            tokens: 0,
            time: "0s",
            model: pane.model,
          }
        }
      })
    )

    setPanes(current =>
      current.map(pane => {
        const response = responses.find(r => r.paneId === pane.id)
        return {
          ...pane,
          messages: [
            ...pane.messages,
            {
              role: "assistant",
              content: response?.content || "Error",
              model: response?.model,
              tokens: response?.tokens,
              time: response?.time,
            },
          ],
          isLoading: false,
        }
      })
    )
    
    // Show modal to select which model to continue with after 15 seconds
    const validResponses = responses.filter(r => r.content && r.content !== "Error processing request")
    if (validResponses.length > 0) {
      setCompareResults(validResponses.map(r => ({
        model: panes.find(p => p.id === r.paneId)?.model || '',
        modelName: r.model,
        content: r.content,
        tokens: r.tokens || 0,
        time: r.time || '0s'
      })))
      setTimeout(() => {
        setShowCompareModal(true)
      }, 15000) // 15 seconds delay
    }
  }

  const handleSelectCompareModel = (result: {model: string, modelName: string, content: string, tokens: number, time: string}, userMessage: string) => {
    // Get the last user message from panes
    const lastUserMsg = panes[0]?.messages.filter(m => m.role === 'user').pop()?.content || userMessage
    
    // Add messages to main chat
    setMessages(prev => [
      ...prev,
      { role: "user", content: lastUserMsg },
      { 
        role: "assistant", 
        content: result.content, 
        model: result.modelName,
        tokens: result.tokens,
        time: result.time
      }
    ])
    
    // Switch to selected model and exit compare mode
    setSelectedModel(result.model)
    setCompareMode(false)
    setShowCompareModal(false)
    
    // Reset panes
    setPanes([
      { id: "1", model: "gpt-4o", messages: [], isLoading: false },
      { id: "2", model: "claude-sonnet-4", messages: [], isLoading: false },
    ])
  }

  const addPane = () => {
    if (panes.length >= 4) return
    setPanes([...panes, { id: String(panes.length + 1), model: "gpt-4.5-preview", messages: [], isLoading: false }])
  }

  const removePane = (id: string) => {
    const newPanes = panes.filter(p => p.id !== id)
    if (newPanes.length <= 1) {
      setCompareMode(false)
      setPanes([
        { id: "1", model: "gpt-4o", messages: [], isLoading: false },
        { id: "2", model: "gpt-4o-mini", messages: [], isLoading: false },
      ])
    } else {
      setPanes(newPanes)
    }
  }

  if (!isLoaded) {
    return (
      <div className="flex min-h-screen bg-background">
        <Sidebar>
          <div className="flex-1 flex items-center justify-center h-full">
            <div className="text-muted-foreground">Loading...</div>
          </div>
        </Sidebar>
      </div>
    )
  }

  return (
    <div className="flex min-h-screen bg-background">
      <Sidebar>
        <div className="flex flex-col h-[calc(100vh-48px)] relative">
          {currentQuery?.isTraining ? (
            <SchemaProcessingAnimation
              isProcessing={true}
              projectName={currentQuery?.name || "Fine-tuning Model"}
              isTraining={true}
              trainingProgress={{
                epoch: trainingProgress.epoch,
                epochs: trainingProgress.epochs,
                accuracy: trainingProgress.accuracy,
                loss: trainingProgress.loss,
                eta: trainingProgress.eta,
              }}
            />
          ) : (
          <>
          <div className="absolute top-2 sm:top-4 right-2 sm:right-6 z-10 flex items-center gap-1 sm:gap-2">
            {!compareMode && (
              <Select value={selectedModel} onValueChange={setSelectedModel}>
                <SelectTrigger className="w-32 sm:w-40 text-xs sm:text-sm">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {availableModels.map(model => (
                    <SelectItem key={model.id} value={model.id}>
                      {model.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
            <Button
              variant={compareMode ? "default" : "outline"}
              size="sm"
              onClick={() => setCompareMode(!compareMode)}
              className="rounded-full text-xs sm:text-sm px-2 sm:px-4"
            >
              <Columns className="h-3 w-3 sm:h-4 sm:w-4 mr-1 sm:mr-2" />
              <span className="hidden sm:inline">Compare</span>
            </Button>
            {compareMode && panes.length < 4 && (
              <Select onValueChange={(modelId) => {
                setPanes(prev => [...prev, { id: String(prev.length + 1), model: modelId, messages: [], isLoading: false }])
              }}>
                <SelectTrigger className="w-32 sm:w-40 text-xs sm:text-sm">
                  <span className="flex items-center gap-1">
                    <Plus className="h-3 w-3" />
                    <span className="hidden sm:inline">Add Model</span>
                  </span>
                </SelectTrigger>
                <SelectContent>
                  {availableModels.map(model => (
                    <SelectItem key={model.id} value={model.id}>
                      {model.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
          </div>

          {!compareMode ? (
            <div className="flex-1 flex flex-col">
              <ScrollArea className="flex-1 p-3 sm:p-6" ref={scrollRef}>
                <div className="max-w-3xl mx-auto space-y-4 sm:space-y-6 pt-12 sm:pt-14 pb-4">
                  {messages.map((message, i) => (
                    <MessageBubble userName={user?.name} key={i} message={message} />
                  ))}
                  {currentQuery?.isTraining ? (
                    <TrainingIndicator progress={trainingProgress} />
                  ) : isLoading ? (
                    <TypingIndicator />
                  ) : null}
                </div>
              </ScrollArea>

              <div className="p-3 sm:p-6 pb-4 sm:pb-8">
                <AiChat
                  selectedModel={selectedModel}
                  onSend={handleSingleSend}
                  dataSourceName={primaryFile?.filename}
                  dataSources={fileNames}
                />
              </div>
            </div>
          ) : (
            <div className="flex-1 flex flex-col pt-12 sm:pt-14 overflow-hidden">
              <div className="flex-1 flex overflow-hidden">
                {panes.map((pane, index) => (
                  <div
                    key={pane.id}
                    className={"flex flex-col flex-1 min-w-0 " + (index < panes.length - 1 ? "border-r border-border" : "")}
                    style={{ width: `${100 / panes.length}%` }}
                  >
                    <div className="flex items-center justify-between px-2 sm:px-3 py-2 border-b border-border bg-muted/30 shrink-0">
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-green-500 shrink-0"></div>
                        <Select
                          value={pane.model}
                          onValueChange={(value) =>
                            setPanes(current => current.map(p => (p.id === pane.id ? { ...p, model: value } : p)))
                          }
                        >
                          <SelectTrigger className="w-24 sm:w-32 text-xs sm:text-sm border-0 bg-transparent p-0 h-auto font-medium">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {availableModels.map(model => (
                              <SelectItem key={model.id} value={model.id}>
                                {model.name}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => removePane(pane.id)}
                        className="h-6 w-6 text-muted-foreground hover:text-destructive shrink-0"
                      >
                        <X className="h-3 w-3" />
                      </Button>
                    </div>

                    <div className="flex-1 overflow-y-auto p-2 sm:p-3">
                      <div className="space-y-3">
                        {pane.messages.length === 0 && (
                          <div className="text-center py-8 text-muted-foreground text-xs sm:text-sm">
                            Send a message to compare
                          </div>
                        )}
                        {pane.messages.map((message, i) => (
                          <MessageBubble userName={user?.name} key={i} message={message} compact />
                        ))}
                        {pane.isLoading && <TypingIndicator />}
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              <div className="border-t border-border p-3 sm:p-4 bg-background shrink-0">
                <AiChat
                  onSend={handleCompareSend}
                  dataSourceName={primaryFile?.filename}
                  dataSources={fileNames}
                />
              </div>
            </div>
          )}
          </>
          )}
        </div>
      </Sidebar>
      
      {/* Compare Model Selection Modal */}
      <Dialog open={showCompareModal} onOpenChange={() => {}} modal={true}>
        <DialogContent className="sm:max-w-lg" onPointerDownOutside={(e) => e.preventDefault()} onEscapeKeyDown={(e) => e.preventDefault()}>
          <DialogHeader>
            <DialogTitle>Select a model to continue</DialogTitle>
            <DialogDescription>
              Choose which response you want to keep and continue the conversation with that model.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-3 mt-4">
            {compareResults.map((result, i) => {
              const firstLine = result.content.split('\n').find(line => line.trim().length > 0) || result.content.substring(0, 100)
              const cleanFirstLine = firstLine.replace(/\*\*/g, '').replace(/^#+\s*/, '').trim()
              return (
                <div
                  key={i}
                  onClick={() => handleSelectCompareModel(result, panes[0]?.messages.filter(m => m.role === 'user').pop()?.content || '')}
                  className="p-4 border border-border rounded-lg cursor-pointer hover:bg-muted/50 hover:border-primary/50 transition-all"
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-green-500"></div>
                      <span className="font-medium">{result.modelName}</span>
                    </div>
                    <div className="flex items-center gap-3 text-xs text-muted-foreground">
                      <span className="flex items-center gap-1">
                        <Zap className="h-3 w-3" />
                        {result.tokens}
                      </span>
                      <span className="flex items-center gap-1">
                        <Clock className="h-3 w-3" />
                        {result.time}
                      </span>
                    </div>
                  </div>
                  <p className="text-sm text-muted-foreground truncate">
                    {cleanFirstLine.length > 80 ? cleanFirstLine.substring(0, 80) + '...' : cleanFirstLine}
                  </p>
                </div>
              )
            })}
          </div>
          <div className="flex justify-end gap-3 mt-4">
            <Button variant="outline" onClick={() => setShowCompareModal(false)}>
              Cancel
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}

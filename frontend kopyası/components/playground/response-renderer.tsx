"use client"

import { useState, useMemo } from "react"
import { cn } from "@/lib/utils"

// ==================== CHART DATA TYPES ====================
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

// ==================== PARSE CHARTS ====================
export function parseCharts(text: string): { content: string; charts: ChartData[] } {
  const charts: ChartData[] = []
  const chartRegex = /\[?CH(?:ART)?:(\w+)\]([\s\S]*?)\[?\/CH(?:ART)?\]/g
  
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
        values: valuesMatch[1].split(/,\s*/).map(s => {
          const cleaned = s.trim().replace(/[^0-9.-]/g, '')
          const num = parseFloat(cleaned)
          return isNaN(num) ? 0 : num
        }),
        values2: values2Match ? values2Match[1].split(',').map(s => parseFloat(s.trim().replace(/,/g, "")) || 0) : undefined,
        values3: values3Match ? values3Match[1].split(',').map(s => parseFloat(s.trim().replace(/,/g, "")) || 0) : undefined,
        title: titleMatch ? titleMatch[1].trim() : '',
        xlabel: xlabelMatch ? xlabelMatch[1].trim() : undefined,
        ylabel: ylabelMatch ? ylabelMatch[1].trim() : undefined,
        series: seriesMatch ? seriesMatch[1].split(',').map(s => s.trim()) : undefined
      })
    }
  }
  
  const cleanContent = text.replace(chartRegex, '').trim()
  
  let result = cleanContent
    .replace(/\[CHART:[a-z_]*\][\s\S]*$/gi, '')
    .replace(/\[CHART:[a-z_]*$/gi, '')
    .replace(/\[CHAR[T]?:?[a-z_]*$/gi, '')
    .replace(/\[CHA?R?T?:?$/gi, '')
    .replace(/\[C?H?A?R?$/gi, '')
    .replace(/\[\/?[A-Z]{0,5}$/gi, '')
    .replace(/\[$/g, '')
    .replace(/\[CHART:[^\]]+\][^\n]*\n?\[\/CHART\]/gi, '')
    .replace(/^CH(?:ART)?:\w+\].*$/gm, '')
    .replace(/^\[\/CHART\]$/gm, '')
    .replace(/^(labels|values|values2|values3|title|xlabel|ylabel|series):[^\n]*$/gm, '')
    .replace(/(labels|values|values2|values3|title|xlabel|ylabel|series):[^\n]*$/g, '')
  
  const lines = result.split('\n')
  const cleanLines: string[] = []
  let tableBuffer: string[] = []
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const trimmed = line.trim()
    const isCompleteTableRow = trimmed.startsWith('|') && trimmed.endsWith('|') && trimmed.length > 2
    const isCompleteSeparator = /^\|[\-:\|\s]+\|$/.test(trimmed)
    const isPartialRow = trimmed.startsWith('|') && !trimmed.endsWith('|')
    const isPartialSeparator = /^\|[\-:\s]*$/.test(trimmed) || /^[\-:\|\s]+$/.test(trimmed)
    
    if (isCompleteTableRow || isCompleteSeparator) {
      tableBuffer.push(line)
    } else if (isPartialRow || (isPartialSeparator && !isCompleteSeparator)) {
      if (tableBuffer.length >= 2) {
        const hasHeader = tableBuffer[0].trim().startsWith('|') && tableBuffer[0].trim().endsWith('|')
        const hasSeparator = tableBuffer.some(r => /^\|[\-:\|\s]+\|$/.test(r.trim()))
        if (hasHeader && hasSeparator) {
          cleanLines.push(...tableBuffer)
        }
      }
      tableBuffer = []
      continue
    } else {
      if (tableBuffer.length >= 2) {
        const hasHeader = tableBuffer[0].trim().startsWith('|') && tableBuffer[0].trim().endsWith('|')
        const hasSeparator = tableBuffer.some(r => /^\|[\-:\|\s]+\|$/.test(r.trim()))
        if (hasHeader && hasSeparator) {
          cleanLines.push(...tableBuffer)
        }
      }
      tableBuffer = []
      if (trimmed !== '') {
        cleanLines.push(line)
      }
    }
  }
  
  if (tableBuffer.length >= 2) {
    const hasHeader = tableBuffer[0].trim().startsWith('|') && tableBuffer[0].trim().endsWith('|')
    const hasSeparator = tableBuffer.some(r => /^\|[\-:\|\s]+\|$/.test(r.trim()))
    if (hasHeader && hasSeparator) {
      cleanLines.push(...tableBuffer)
    }
  }
  
  const finalContent = cleanLines.join('\n').replace(/\n{3,}/g, '\n\n').trim()
  
  return { content: finalContent, charts }
}

// ==================== TOOLTIP ====================
function Tooltip({ x, y, content, visible }: { x: number; y: number; content: string; visible: boolean }) {
  if (!visible) return null
  return (
    <div className="absolute bg-gray-900 text-white text-xs px-2 py-1 rounded shadow-lg pointer-events-none z-50 whitespace-nowrap"
      style={{ left: x + 10, top: y - 25 }}>
      {content}
    </div>
  )
}

// ==================== ADVANCED CHART (20+ TYPES) ====================
export function AdvancedChart({ type, labels, values, values2, values3, title, xlabel, ylabel, series }: ChartData) {
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

  // 1. HORIZONTAL BAR CHART
  if (type === 'hbar' || type === 'bar') {
    return (
      <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-xl shadow-sm relative max-w-full overflow-x-auto">
        {title && <p className="text-sm font-semibold mb-3 text-gray-700 dark:text-gray-300">{title}</p>}
        <div className="space-y-2">
          {labels.map((label, i) => (
            <div key={i} className="flex items-center gap-2 group cursor-pointer"
              onMouseMove={(e) => showTooltip(e, `${label}: ${(values[i] ?? 0).toLocaleString()}`)}
              onMouseLeave={hideTooltip}>
              <span className="text-xs w-24 truncate text-gray-600 dark:text-gray-400 text-right">{label}</span>
              <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-6 overflow-hidden">
                <div className="h-full rounded-full transition-all duration-500 flex items-center justify-end pr-2 group-hover:opacity-80"
                  style={{ width: `${(values[i] / maxValue) * 100}%`, backgroundColor: colors[i % colors.length] }}>
                  <span className="text-[10px] text-white font-medium opacity-0 group-hover:opacity-100">{(values[i] ?? 0).toLocaleString()}</span>
                </div>
              </div>
              <span className="text-[10px] w-20 font-medium text-right tabular-nums">{(values[i] ?? 0).toLocaleString()}</span>
            </div>
          ))}
        </div>
        <Tooltip {...tooltip} />
      </div>
    )
  }

  // 2. GROUPED BAR CHART
  if (type === 'grouped') {
    const groupMax = Math.max(...values, ...(values2 || []))
    const seriesNames = series || ['Series 1', 'Series 2']
    const hasValues2 = values2 && values2.length > 0
    return (
      <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-xl shadow-sm relative max-w-full overflow-x-auto">
        {title && <p className="text-sm font-semibold mb-3 text-gray-700 dark:text-gray-300">{title}</p>}
        <div className="flex items-end gap-4 h-36 px-2">
          {labels.map((label, i) => (
            <div key={i} className="flex-1 flex flex-col items-center gap-1">
              <div className="flex items-end gap-1 h-28 w-full justify-center">
                <div className="w-5 bg-blue-500 rounded-t transition-all cursor-pointer hover:opacity-80"
                  style={{ height: `${(values[i] / groupMax) * 100}%` }}
                  onMouseMove={(e) => showTooltip(e, `${seriesNames[0]}: ${(values[i] ?? 0).toLocaleString()}`)}
                  onMouseLeave={hideTooltip} />
                {hasValues2 && <div className="w-5 bg-emerald-500 rounded-t transition-all cursor-pointer hover:opacity-80"
                  style={{ height: `${((values2?.[i] || 0) / groupMax) * 100}%` }}
                  onMouseMove={(e) => showTooltip(e, `${seriesNames[1]}: ${(values2?.[i] || 0).toLocaleString()}`)}
                  onMouseLeave={hideTooltip} />}
              </div>
              <span className="text-[10px] text-gray-500 truncate max-w-full text-center">{label}</span>
            </div>
          ))}
        </div>
        <div className="flex justify-center gap-4 mt-3 text-xs">
          <div className="flex items-center gap-1"><div className="w-3 h-3 bg-blue-500 rounded" />{seriesNames[0]}</div>
          {hasValues2 && <div className="flex items-center gap-1"><div className="w-3 h-3 bg-emerald-500 rounded" />{seriesNames[1]}</div>}
        </div>
        <Tooltip {...tooltip} />
      </div>
    )
  }

  // 3. STACKED BAR CHART
  if (type === 'stacked') {
    const seriesNames = series || ['Series 1', 'Series 2']
    const hasValues2 = values2 && values2.length > 0
    if (!hasValues2) {
      return (
        <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-xl shadow-sm relative max-w-full overflow-x-auto">
          {title && <p className="text-sm font-semibold mb-3 text-gray-700 dark:text-gray-300">{title}</p>}
          <div className="space-y-2">
            {labels.map((label, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="text-xs w-24 truncate text-gray-600 dark:text-gray-400 text-right">{label}</span>
                <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-6 overflow-hidden">
                  <div className="h-full rounded-full bg-blue-500" style={{ width: `${(values[i] / maxValue) * 100}%` }} />
                </div>
                <span className="text-xs w-16 font-medium text-right">{(values[i] ?? 0).toLocaleString()}</span>
              </div>
            ))}
          </div>
          <Tooltip {...tooltip} />
        </div>
      )
    }
    return (
      <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-xl shadow-sm relative max-w-full overflow-x-auto">
        {title && <p className="text-sm font-semibold mb-3 text-gray-700 dark:text-gray-300">{title}</p>}
        <div className="flex items-end gap-2 h-36 px-2">
          {labels.map((label, i) => {
            const stackTotal = values[i] + (values2?.[i] || 0)
            const maxStack = Math.max(...values.map((v, j) => v + (values2?.[j] || 0)))
            return (
              <div key={i} className="flex-1 flex flex-col items-center gap-1">
                <div className="w-full flex flex-col-reverse" style={{ height: `${(stackTotal / maxStack) * 100}%`, minHeight: '8px' }}>
                  <div className="w-full bg-blue-500 rounded-b cursor-pointer hover:opacity-80 transition-opacity"
                    style={{ height: `${(values[i] / stackTotal) * 100}%` }}
                    onMouseMove={(e) => showTooltip(e, `${seriesNames[0]}: ${(values[i] ?? 0).toLocaleString()}`)}
                    onMouseLeave={hideTooltip} />
                  <div className="w-full bg-emerald-500 rounded-t cursor-pointer hover:opacity-80 transition-opacity"
                    style={{ height: `${((values2?.[i] || 0) / stackTotal) * 100}%` }}
                    onMouseMove={(e) => showTooltip(e, `${seriesNames[1]}: ${(values2?.[i] || 0).toLocaleString()}`)}
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

  // 4. LINE CHART
  if (type === 'line') {
    const width = Math.min(450, typeof window !== "undefined" ? window.innerWidth - 80 : 450), height = 180, padding = 45
    const chartWidth = width - padding * 2, chartHeight = height - padding * 2
    const range = maxValue - minValue || 1
    const points = values.map((v, i) => ({
      x: padding + (i / Math.max(values.length - 1, 1)) * chartWidth,
      y: height - padding - ((v - minValue) / range) * chartHeight,
      value: v, label: labels[i]
    }))
    const pathD = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ')
    
    return (
      <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-xl shadow-sm relative max-w-full overflow-x-auto">
        {title && <p className="text-sm font-semibold mb-3 text-gray-700 dark:text-gray-300">{title}</p>}
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto">
          <defs>
            <linearGradient id="lineGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.3" />
              <stop offset="100%" stopColor="#3b82f6" stopOpacity="0" />
            </linearGradient>
          </defs>
          {[0, 0.25, 0.5, 0.75, 1].map((t, i) => (
            <line key={i} x1={padding} x2={width - padding} y1={height - padding - t * chartHeight} y2={height - padding - t * chartHeight} stroke="#e5e7eb" strokeDasharray="4" />
          ))}
          <path d={`${pathD} L ${points[points.length-1].x} ${height - padding} L ${padding} ${height - padding} Z`} fill="url(#lineGrad)" />
          <path d={pathD} fill="none" stroke="#3b82f6" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
          {points.map((p, i) => (
            <g key={i}>
              <circle cx={p.x} cy={p.y} r="5" fill="#3b82f6" className="cursor-pointer hover:r-7 transition-all"
                onMouseMove={(e) => showTooltip(e, `${p.label}: ${p.value.toLocaleString()}`)}
                onMouseLeave={hideTooltip} />
              <text x={p.x} y={height - padding + 15} textAnchor="middle" className="text-[9px] fill-gray-500">{p.label}</text>
            </g>
          ))}
          <text x={padding - 8} y={padding} textAnchor="end" className="text-[9px] fill-gray-400">{maxValue.toLocaleString()}</text>
          <text x={padding - 8} y={height - padding} textAnchor="end" className="text-[9px] fill-gray-400">{minValue.toLocaleString()}</text>
          {ylabel && <text x={12} y={height/2} textAnchor="middle" transform={`rotate(-90,12,${height/2})`} className="text-[10px] fill-gray-600">{ylabel}</text>}
          {xlabel && <text x={width/2} y={height - 5} textAnchor="middle" className="text-[10px] fill-gray-600">{xlabel}</text>}
        </svg>
        <Tooltip {...tooltip} />
      </div>
    )
  }

  // 5. MULTI-LINE CHART
  if (type === 'multiline') {
    const hasValues2 = values2 && values2.length > 0
    if (!hasValues2) {
      const lineMax = Math.max(...values)
      const points = values.map((v, i) => ({ x: (i / (values.length - 1)) * 280 + 20, y: 120 - (v / lineMax) * 100 }))
      return (
        <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-xl shadow-sm relative max-w-full overflow-x-auto">
          {title && <p className="text-sm font-semibold mb-3 text-gray-700 dark:text-gray-300">{title}</p>}
          <svg viewBox="0 0 320 140" className="w-full h-auto">
            <polyline fill="none" stroke="#3b82f6" strokeWidth="2" points={points.map(p => `${p.x},${p.y}`).join(' ')} />
            {points.map((p, i) => (
              <circle key={i} cx={p.x} cy={p.y} r="4" fill="#3b82f6" className="cursor-pointer"
                onMouseMove={(e) => showTooltip(e, `${labels[i]}: ${(values[i] ?? 0).toLocaleString()}`)}
                onMouseLeave={hideTooltip} />
            ))}
          </svg>
          <Tooltip {...tooltip} />
        </div>
      )
    }
    const width = Math.min(450, typeof window !== "undefined" ? window.innerWidth - 80 : 450), height = 180, padding = 45
    const chartWidth = width - padding * 2, chartHeight = height - padding * 2
    const allMax = Math.max(...values, ...values2)
    const allMin = Math.min(...values, ...values2)
    const range = allMax - allMin || 1
    const seriesNames = series || ['Series 1', 'Series 2']
    
    const points1 = values.map((v, i) => ({ x: padding + (i / Math.max(values.length - 1, 1)) * chartWidth, y: height - padding - ((v - allMin) / range) * chartHeight, value: v }))
    const points2 = values2.map((v, i) => ({ x: padding + (i / Math.max(values2.length - 1, 1)) * chartWidth, y: height - padding - ((v - allMin) / range) * chartHeight, value: v }))
    
    return (
      <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-xl shadow-sm relative max-w-full overflow-x-auto">
        {title && <p className="text-sm font-semibold mb-3 text-gray-700 dark:text-gray-300">{title}</p>}
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto">
          {[0, 0.5, 1].map((t, i) => <line key={i} x1={padding} x2={width-padding} y1={height-padding-t*chartHeight} y2={height-padding-t*chartHeight} stroke="#e5e7eb" strokeDasharray="4" />)}
          <path d={points1.map((p,i) => `${i===0?'M':'L'} ${p.x} ${p.y}`).join(' ')} fill="none" stroke="#3b82f6" strokeWidth="2" />
          <path d={points2.map((p,i) => `${i===0?'M':'L'} ${p.x} ${p.y}`).join(' ')} fill="none" stroke="#10b981" strokeWidth="2" />
          {points1.map((p,i) => <circle key={`a${i}`} cx={p.x} cy={p.y} r="4" fill="#3b82f6" className="cursor-pointer" onMouseMove={(e)=>showTooltip(e,`${seriesNames[0]}: ${p.value}`)} onMouseLeave={hideTooltip} />)}
          {points2.map((p,i) => <circle key={`b${i}`} cx={p.x} cy={p.y} r="4" fill="#10b981" className="cursor-pointer" onMouseMove={(e)=>showTooltip(e,`${seriesNames[1]}: ${p.value}`)} onMouseLeave={hideTooltip} />)}
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

  // 6. AREA CHART
  if (type === 'area') {
    const width = Math.min(450, typeof window !== "undefined" ? window.innerWidth - 80 : 450), height = 160, padding = 45
    const chartWidth = width - padding * 2, chartHeight = height - padding * 2
    const range = maxValue - minValue || 1
    const points = values.map((v, i) => ({
      x: padding + (i / Math.max(values.length - 1, 1)) * chartWidth,
      y: height - padding - ((v - minValue) / range) * chartHeight, value: v
    }))
    
    return (
      <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-xl shadow-sm relative max-w-full overflow-x-auto">
        {title && <p className="text-sm font-semibold mb-3 text-gray-700 dark:text-gray-300">{title}</p>}
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto">
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

  // 7. PIE CHART
  if (type === 'pie') {
    let currentAngle = 0
    const slices = values.map((v, i) => {
      const angle = (v / total) * 360
      const start = currentAngle
      currentAngle += angle
      return { value: v, label: labels[i], startAngle: start, angle, color: colors[i % colors.length], percent: ((v/total)*100).toFixed(1) }
    })
    
    return (
      <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-xl shadow-sm relative max-w-full overflow-x-auto">
        {title && <p className="text-sm font-semibold mb-3 text-gray-700 dark:text-gray-300">{title}</p>}
        <div className="flex items-center justify-center gap-6">
          <svg viewBox="0 0 150 150" className="w-full h-auto max-w-[150px] mx-auto">
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
                <span className="text-gray-600 dark:text-gray-400">{s.label}</span>
                <span className="text-gray-400">{s.percent}%</span>
              </div>
            ))}
          </div>
        </div>
        <Tooltip {...tooltip} />
      </div>
    )
  }

  // 8. DONUT CHART
  if (type === 'donut') {
    let currentAngle = 0
    const outerR = 65, innerR = 40
    
    return (
      <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-xl shadow-sm relative max-w-full overflow-x-auto">
        {title && <p className="text-sm font-semibold mb-3 text-gray-700 dark:text-gray-300">{title}</p>}
        <div className="flex items-center justify-center gap-6">
          <svg viewBox="0 0 150 150" className="w-full h-auto max-w-[150px] mx-auto">
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
            <text x="75" y="72" textAnchor="middle" className="text-xl font-bold fill-gray-700 dark:fill-gray-300">{total.toLocaleString()}</text>
            <text x="75" y="88" textAnchor="middle" className="text-[10px] fill-gray-400">Total</text>
          </svg>
          <div className="text-xs space-y-1.5">
            {labels.map((l, i) => (
              <div key={i} className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: colors[i % colors.length] }} />
                <span>{l}: {(values[i] ?? 0).toLocaleString()}</span>
              </div>
            ))}
          </div>
        </div>
        <Tooltip {...tooltip} />
      </div>
    )
  }

  // 9. GAUGE CHART
  if (type === 'gauge') {
    const pct = Math.min(100, Math.max(0, values[0]))
    const color = pct < 33 ? '#ef4444' : pct < 66 ? '#f59e0b' : '#10b981'
    const angle = (pct / 100) * 180 - 90
    
    return (
      <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-xl shadow-sm relative max-w-full overflow-x-auto">
        {title && <p className="text-sm font-semibold mb-3 text-gray-700 dark:text-gray-300">{title}</p>}
        <svg width="280" height="110" viewBox="0 0 180 110" className="mx-auto">
          <path d="M 20 90 A 70 70 0 0 1 160 90" fill="none" stroke="#e5e7eb" strokeWidth="14" strokeLinecap="round" />
          <path d={`M 20 90 A 70 70 0 ${pct > 50 ? 1 : 0} 1 ${90 + 70 * Math.cos(angle * Math.PI / 180)} ${90 + 70 * Math.sin(angle * Math.PI / 180)}`}
            fill="none" stroke={color} strokeWidth="14" strokeLinecap="round" />
          <text x="90" y="85" textAnchor="middle" className="text-3xl font-bold" fill={color}>{pct.toFixed(1)}%</text>
          <text x="90" y="105" textAnchor="middle" className="text-[11px] fill-gray-500">{labels[0] || 'Score'}</text>
        </svg>
      </div>
    )
  }

  // 10. SCATTER PLOT
  if (type === 'scatter') {
    const hasValues2 = values2 && values2.length > 0
    if (!hasValues2) {
      return (
        <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-xl shadow-sm relative max-w-full overflow-x-auto">
          {title && <p className="text-sm font-semibold mb-3 text-gray-700 dark:text-gray-300">{title}</p>}
          <div className="space-y-2">
            {labels.map((label, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="text-xs w-24 truncate text-gray-600 dark:text-gray-400 text-right">{label}</span>
                <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-6 overflow-hidden">
                  <div className="h-full rounded-full bg-blue-500" style={{ width: `${(values[i] / maxValue) * 100}%` }} />
                </div>
                <span className="text-xs w-16 font-medium text-right">{(values[i] ?? 0).toLocaleString()}</span>
              </div>
            ))}
          </div>
          <Tooltip {...tooltip} />
        </div>
      )
    }
    const width = Math.min(420, typeof window !== "undefined" ? window.innerWidth - 80 : 420), height = 200, padding = 40
    const xMax = Math.max(...values), xMin = Math.min(...values)
    const yMax = Math.max(...values2), yMin = Math.min(...values2)
    const xRange = xMax - xMin || 1, yRange = yMax - yMin || 1
    
    return (
      <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-xl shadow-sm relative max-w-full overflow-x-auto">
        {title && <p className="text-sm font-semibold mb-3 text-gray-700 dark:text-gray-300">{title}</p>}
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto">
          {[0, 0.5, 1].map((t, i) => (
            <g key={i}>
              <line x1={padding} x2={width-padding} y1={height-padding-t*(height-2*padding)} y2={height-padding-t*(height-2*padding)} stroke="#e5e7eb" strokeDasharray="4" />
              <line x1={padding+t*(width-2*padding)} y1={padding} x2={padding+t*(width-2*padding)} y2={height-padding} stroke="#e5e7eb" strokeDasharray="4" />
            </g>
          ))}
          <line x1={padding} y1={height-padding} x2={width-padding} y2={height-padding} stroke="#9ca3af" />
          <line x1={padding} y1={padding} x2={padding} y2={height-padding} stroke="#9ca3af" />
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

  // 11. RADAR CHART
  if (type === 'radar') {
    const cx = 120, cy = 100, r = 70
    const n = labels.length
    const angleStep = (2 * Math.PI) / n
    const points = values.map((v, i) => {
      const angle = angleStep * i - Math.PI / 2
      const dist = (v / maxValue) * r
      return { x: cx + dist * Math.cos(angle), y: cy + dist * Math.sin(angle), value: v, label: labels[i] }
    })
    
    return (
      <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-xl shadow-sm relative max-w-full overflow-x-auto">
        {title && <p className="text-sm font-semibold mb-3 text-gray-700 dark:text-gray-300">{title}</p>}
        <svg width="240" height="200">
          {[0.25, 0.5, 0.75, 1].map((t, i) => (
            <polygon key={i} points={Array.from({length: n}, (_, j) => {
              const angle = angleStep * j - Math.PI / 2
              return `${cx + t * r * Math.cos(angle)},${cy + t * r * Math.sin(angle)}`
            }).join(' ')} fill="none" stroke="#e5e7eb" />
          ))}
          {labels.map((_, i) => {
            const angle = angleStep * i - Math.PI / 2
            return <line key={i} x1={cx} y1={cy} x2={cx + r * Math.cos(angle)} y2={cy + r * Math.sin(angle)} stroke="#d1d5db" />
          })}
          <polygon points={points.map(p => `${p.x},${p.y}`).join(' ')} fill="#3b82f6" fillOpacity="0.3" stroke="#3b82f6" strokeWidth="2" />
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

  // 12. HEATMAP
  if (type === 'heatmap') {
    const rows = labels.length
    const cols = Math.ceil(values.length / rows)
    const cellW = 40, cellH = 30
    
    return (
      <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-xl shadow-sm relative overflow-x-auto">
        {title && <p className="text-sm font-semibold mb-3 text-gray-700 dark:text-gray-300">{title}</p>}
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

  // 13. WATERFALL CHART
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
      <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-xl shadow-sm relative max-w-full overflow-x-auto">
        {title && <p className="text-sm font-semibold mb-3 text-gray-700 dark:text-gray-300">{title}</p>}
        <div className="flex items-end gap-1 h-40 px-2">
          {bars.map((b, i) => (
            <div key={i} className="flex-1 flex flex-col items-center relative h-full">
              <div className="absolute w-full cursor-pointer rounded"
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

  // 14. FUNNEL CHART
  if (type === 'funnel') {
    const maxWidth = 200
    return (
      <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-xl shadow-sm relative max-w-full overflow-x-auto">
        {title && <p className="text-sm font-semibold mb-3 text-gray-700 dark:text-gray-300">{title}</p>}
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

  // 15. TREEMAP
  if (type === 'treemap') {
    const sorted = values.map((v, i) => ({ value: v, label: labels[i], color: colors[i % colors.length] })).sort((a, b) => b.value - a.value)
    return (
      <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-xl shadow-sm relative max-w-full overflow-x-auto">
        {title && <p className="text-sm font-semibold mb-3 text-gray-700 dark:text-gray-300">{title}</p>}
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

  // 16. BOX PLOT
  if (type === 'boxplot') {
    const sorted = [...values].sort((a, b) => a - b)
    const q1 = sorted[Math.floor(sorted.length * 0.25)]
    const median = sorted[Math.floor(sorted.length * 0.5)]
    const q3 = sorted[Math.floor(sorted.length * 0.75)]
    const min = sorted[0], max = sorted[sorted.length - 1]
    const range = max - min || 1
    const width = 200, height = 100, padding = 30
    
    return (
      <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-xl shadow-sm relative max-w-full overflow-x-auto">
        {title && <p className="text-sm font-semibold mb-3 text-gray-700 dark:text-gray-300">{title}</p>}
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto">
          <line x1={padding + ((min - min) / range) * (width - 2*padding)} y1={50} x2={padding + ((q1 - min) / range) * (width - 2*padding)} y2={50} stroke="#6b7280" strokeWidth="2" />
          <line x1={padding + ((q3 - min) / range) * (width - 2*padding)} y1={50} x2={padding + ((max - min) / range) * (width - 2*padding)} y2={50} stroke="#6b7280" strokeWidth="2" />
          <rect x={padding + ((q1 - min) / range) * (width - 2*padding)} y={30} width={((q3 - q1) / range) * (width - 2*padding)} height={40}
            fill="#3b82f6" fillOpacity="0.5" stroke="#3b82f6" strokeWidth="2" className="cursor-pointer"
            onMouseMove={(e) => showTooltip(e, `Q1: ${q1}, Median: ${median}, Q3: ${q3}`)}
            onMouseLeave={hideTooltip} />
          <line x1={padding + ((median - min) / range) * (width - 2*padding)} y1={30} x2={padding + ((median - min) / range) * (width - 2*padding)} y2={70} stroke="#1d4ed8" strokeWidth="3" />
          <text x={padding} y={80} className="text-[9px] fill-gray-500">{min}</text>
          <text x={width - padding} y={80} textAnchor="end" className="text-[9px] fill-gray-500">{max}</text>
        </svg>
        <div className="text-center text-[10px] text-gray-500">Min: {min} | Q1: {q1} | Median: {median} | Q3: {q3} | Max: {max}</div>
        <Tooltip {...tooltip} />
      </div>
    )
  }

  // 17. HISTOGRAM
  if (type === 'histogram') {
    return (
      <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-xl shadow-sm relative max-w-full overflow-x-auto">
        {title && <p className="text-sm font-semibold mb-3 text-gray-700 dark:text-gray-300">{title}</p>}
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

  // 18. BULLET CHART
  if (type === 'bullet') {
    const hasValues2 = values2 && values2.length > 0
    if (!hasValues2) {
      return (
        <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-xl shadow-sm relative max-w-full overflow-x-auto">
          {title && <p className="text-sm font-semibold mb-3 text-gray-700 dark:text-gray-300">{title}</p>}
          <div className="space-y-2">
            {labels.map((label, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="text-xs w-24 truncate text-gray-600 dark:text-gray-400 text-right">{label}</span>
                <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-6 overflow-hidden">
                  <div className="h-full rounded-full bg-blue-500" style={{ width: `${(values[i] / maxValue) * 100}%` }} />
                </div>
                <span className="text-xs w-16 font-medium text-right">{(values[i] ?? 0).toLocaleString()}</span>
              </div>
            ))}
          </div>
          <Tooltip {...tooltip} />
        </div>
      )
    }
    const target = values2[0]
    const actual = values[0]
    const max = Math.max(target, actual) * 1.2
    
    return (
      <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-xl shadow-sm relative max-w-full overflow-x-auto">
        {title && <p className="text-sm font-semibold mb-3 text-gray-700 dark:text-gray-300">{title}</p>}
        <div className="relative h-10 bg-gray-200 dark:bg-gray-700 rounded overflow-hidden">
          <div className="absolute inset-y-0 left-0 bg-gray-300 dark:bg-gray-600" style={{ width: '60%' }} />
          <div className="absolute inset-y-0 left-0 bg-gray-400 dark:bg-gray-500" style={{ width: '30%' }} />
          <div className="absolute top-2 bottom-2 left-0 bg-blue-600 rounded cursor-pointer"
            style={{ width: `${(actual / max) * 100}%` }}
            onMouseMove={(e) => showTooltip(e, `Actual: ${actual}`)}
            onMouseLeave={hideTooltip} />
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

  // 19. METRICS/PROGRESS CARDS
  if (type === 'metrics') {
    return (
      <div className="my-4 grid grid-cols-2 gap-3">
        {labels.map((label, i) => {
          const pct = values2 ? ((values[i] / values2[i]) * 100).toFixed(1) : null
          return (
            <div key={i} className="p-3 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-xl shadow-sm cursor-pointer hover:shadow-md transition-shadow"
              onMouseMove={(e) => showTooltip(e, `${label}: ${(values[i] ?? 0).toLocaleString()}${pct ? ` (${pct}% of target)` : ''}`)}
              onMouseLeave={hideTooltip}>
              <div className="text-[10px] text-gray-500 uppercase tracking-wide">{label}</div>
              <div className="text-xl font-bold text-gray-800 dark:text-gray-200">{(values[i] ?? 0).toLocaleString()}</div>
              {pct && (
                <div className="mt-1 h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
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

  // DEFAULT - horizontal bar fallback
  return (
    <div className="my-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-xl shadow-sm relative max-w-full overflow-x-auto">
      {title && <p className="text-sm font-semibold mb-3 text-gray-700 dark:text-gray-300">{title}</p>}
      <div className="space-y-2">
        {labels.map((label, i) => (
          <div key={i} className="flex items-center gap-2 group cursor-pointer"
            onMouseMove={(e) => showTooltip(e, `${label}: ${(values[i] ?? 0).toLocaleString()}`)}
            onMouseLeave={hideTooltip}>
            <span className="text-xs w-24 truncate text-gray-600 dark:text-gray-400 text-right">{label}</span>
            <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-6 overflow-hidden">
              <div className="h-full rounded-full transition-all duration-500 group-hover:opacity-80"
                style={{ width: `${(values[i] / maxValue) * 100}%`, backgroundColor: colors[i % colors.length] }} />
            </div>
            <span className="text-xs w-16 font-medium text-right">{(values[i] ?? 0).toLocaleString()}</span>
          </div>
        ))}
      </div>
      <Tooltip {...tooltip} />
    </div>
  )
}

// ==================== SORTABLE TABLE ====================
function SortableTable({ headers, rows }: { headers: string[]; rows: string[][] }) {
  const [sortCol, setSortCol] = useState<number | null>(null)
  const [sortAsc, setSortAsc] = useState(true)
  
  const sortedRows = useMemo(() => {
    if (sortCol === null) return rows
    return [...rows].sort((a, b) => {
      const aVal = a[sortCol] || ""
      const bVal = b[sortCol] || ""
      const aNum = parseFloat(aVal.replace(/[^0-9.-]/g, ""))
      const bNum = parseFloat(bVal.replace(/[^0-9.-]/g, ""))
      if (!isNaN(aNum) && !isNaN(bNum)) {
        return sortAsc ? aNum - bNum : bNum - aNum
      }
      return sortAsc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal)
    })
  }, [rows, sortCol, sortAsc])
  
  const handleSort = (colIdx: number) => {
    if (sortCol === colIdx) {
      setSortAsc(!sortAsc)
    } else {
      setSortCol(colIdx)
      setSortAsc(true)
    }
  }
  
  return (
    <div className="overflow-x-auto my-3">
      <table className="min-w-full border border-gray-200 dark:border-gray-700 rounded-lg text-xs">
        <thead className="bg-gray-100 dark:bg-gray-800">
          <tr>
            {headers.map((h, i) => (
              <th 
                key={i} 
                className="px-3 py-2 text-left font-semibold border-b border-gray-200 dark:border-gray-700 cursor-pointer hover:bg-gray-200 dark:hover:bg-gray-700 select-none"
                onClick={() => handleSort(i)}
              >
                <div className="flex items-center gap-1">
                  {h}
                  <span className="text-gray-400 text-[10px]">
                    {sortCol === i ? (sortAsc ? "▲" : "▼") : "⇅"}
                  </span>
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sortedRows.map((row, i) => (
            <tr key={i} className={i % 2 === 0 ? "bg-white dark:bg-gray-900 hover:bg-blue-50 dark:hover:bg-gray-800" : "bg-gray-50 dark:bg-gray-800/50 hover:bg-blue-50 dark:hover:bg-gray-800"}>
              {row.map((cell, j) => <td key={j} className="px-3 py-2 border-b border-gray-200 dark:border-gray-700">{cell}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ==================== RENDER FORMATTED TEXT ====================
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

// ==================== CONTENT RENDERER ====================
export function ContentRenderer({ content }: { content: string }) {
  const renderedContent = useMemo(() => {
    const { content: cleanContent, charts } = parseCharts(content)
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
      return <SortableTable key={"table-" + resultIndex++} headers={headers} rows={dataRows} />
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
        if (trimmedLine.startsWith("#") || (trimmedLine.startsWith("**") && (trimmedLine.endsWith("**") || trimmedLine.endsWith(":**")))) {
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
    
    charts.forEach((chart, i) => {
      result.push(<AdvancedChart key={"chart-" + i} {...chart} />)
    })

    return result
  }, [content])

  return <div className="text-sm">{renderedContent}</div>
}

// ==================== RESPONSE BLOCK TYPES ====================
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

export function ResponseRenderer({ blocks }: { blocks: ResponseBlock[] }) {
  return (
    <div className="space-y-4">
      {blocks.map((block, i) => {
        if (block.type === "text") {
          return <ContentRenderer key={i} content={block.content as string} />
        }
        return <ContentRenderer key={i} content={String(block.content)} />
      })}
    </div>
  )
}

// ─── Function Call Display ───
export function FunctionCallDisplay({ calls }: { calls: { function_name: string; arguments: any; result: any; error?: string; execution_ms: number }[] }) {
  if (!calls || calls.length === 0) return null
  
  return (
    <div className="my-2 space-y-2">
      {calls.map((call, i) => (
        <div key={i} className="border border-[#2684FF]/20 rounded-lg overflow-hidden bg-[#1a1a2e]/50">
          <div className="flex items-center gap-2 px-3 py-1.5 bg-[#2684FF]/10 border-b border-[#2684FF]/20">
            <span className="text-[#2684FF] text-xs">⚡</span>
            <span className="text-xs font-mono font-medium text-[#2684FF]">{call.function_name}</span>
            <span className="text-xs text-gray-500 ml-auto">{call.execution_ms}ms</span>
            {call.error && <span className="text-xs text-red-400">⚠ error</span>}
          </div>
          <details className="group">
            <summary className="px-3 py-1 text-xs text-gray-400 cursor-pointer hover:text-gray-300">
              Show details
            </summary>
            <div className="px-3 py-2 space-y-1">
              <div>
                <span className="text-xs text-gray-500">Arguments:</span>
                <pre className="text-xs text-gray-300 mt-0.5 overflow-x-auto max-h-32 overflow-y-auto bg-black/30 rounded p-1.5">
                  {JSON.stringify(call.arguments, null, 2)}
                </pre>
              </div>
              {call.error ? (
                <div>
                  <span className="text-xs text-red-400">Error:</span>
                  <pre className="text-xs text-red-300 mt-0.5 bg-black/30 rounded p-1.5">{call.error}</pre>
                </div>
              ) : (
                <div>
                  <span className="text-xs text-gray-500">Result:</span>
                  <pre className="text-xs text-gray-300 mt-0.5 overflow-x-auto max-h-48 overflow-y-auto bg-black/30 rounded p-1.5">
                    {JSON.stringify(call.result, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          </details>
        </div>
      ))}
    </div>
  )
}

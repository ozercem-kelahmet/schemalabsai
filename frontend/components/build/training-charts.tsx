"use client"

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer } from "recharts"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import type { TrainingMetrics } from "@/lib/types"
import { HelpCircle, TrendingDown, TrendingUp } from "lucide-react"

interface TrainingChartsProps {
  history: TrainingMetrics[]
}

export function TrainingCharts({ history }: TrainingChartsProps) {
  const chartData = history.map((h) => ({
    epoch: h.epoch,
    loss: Number(h.loss.toFixed(4)),
    accuracy: Number((h.accuracy * 100).toFixed(2)),
  }))

  // Calculate trends
  const lossTrend = chartData.length >= 2 
    ? chartData[chartData.length - 1].loss < chartData[0].loss 
      ? "decreasing" 
      : "increasing"
    : null
  
  const accuracyTrend = chartData.length >= 2 
    ? chartData[chartData.length - 1].accuracy > chartData[0].accuracy 
      ? "increasing" 
      : "decreasing"
    : null

  return (
    <TooltipProvider delayDuration={200}>
      <div className="grid gap-4 lg:grid-cols-2">
        {/* Loss Chart */}
        <div className="rounded-xl border border-border bg-card p-4">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <h4 className="text-sm font-medium text-foreground">Loss Curve</h4>
              {lossTrend && (
                <span className={`flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded ${
                  lossTrend === "decreasing" 
                    ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400" 
                    : "bg-amber-500/10 text-amber-600 dark:text-amber-400"
                }`}>
                  {lossTrend === "decreasing" ? <TrendingDown className="h-3 w-3" /> : <TrendingUp className="h-3 w-3" />}
                  {lossTrend === "decreasing" ? "Improving" : "Watch"}
                </span>
              )}
            </div>
            <Tooltip>
              <TooltipTrigger asChild>
                <button className="text-muted-foreground/50 hover:text-muted-foreground transition-colors">
                  <HelpCircle className="h-4 w-4" />
                </button>
              </TooltipTrigger>
              <TooltipContent side="left" className="max-w-[280px]">
                <p className="font-medium mb-1">Understanding the Loss Curve</p>
                <p className="text-xs text-muted-foreground mb-2">Loss measures prediction errors. A healthy training shows loss decreasing over epochs.</p>
                <div className="text-xs space-y-1">
                  <p><span className="text-emerald-500 font-medium">Good:</span> Steady decrease, flattening at low value</p>
                  <p><span className="text-amber-500 font-medium">Watch:</span> Fluctuating or increasing values</p>
                  <p><span className="text-red-500 font-medium">Issue:</span> Sudden spikes or no improvement</p>
                </div>
              </TooltipContent>
            </Tooltip>
          </div>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="epoch" stroke="hsl(var(--muted-foreground))" fontSize={10} tickFormatter={(v) => `E${v}`} />
                <YAxis stroke="hsl(var(--muted-foreground))" fontSize={10} />
                <RechartsTooltip
                  cursor={{ stroke: "hsl(var(--border))", strokeDasharray: "3 3" }}
                  contentStyle={{
                    backgroundColor: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "8px",
                    fontSize: "12px",
                  }}
                  labelStyle={{ color: "hsl(var(--foreground))" }}
                  labelFormatter={(label) => `Epoch ${label}`}
                  formatter={(value: number) => [value.toFixed(4), "Loss"]}
                />
                <Line type="monotone" dataKey="loss" stroke="#f87171" strokeWidth={2} dot={chartData.length < 20} activeDot={{ r: 4, stroke: "#f87171" }} name="Loss" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Accuracy Chart */}
        <div className="rounded-xl border border-border bg-card p-4">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <h4 className="text-sm font-medium text-foreground">Accuracy Curve</h4>
              {accuracyTrend && (
                <span className={`flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded ${
                  accuracyTrend === "increasing" 
                    ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400" 
                    : "bg-amber-500/10 text-amber-600 dark:text-amber-400"
                }`}>
                  {accuracyTrend === "increasing" ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
                  {accuracyTrend === "increasing" ? "Improving" : "Watch"}
                </span>
              )}
            </div>
            <Tooltip>
              <TooltipTrigger asChild>
                <button className="text-muted-foreground/50 hover:text-muted-foreground transition-colors">
                  <HelpCircle className="h-4 w-4" />
                </button>
              </TooltipTrigger>
              <TooltipContent side="left" className="max-w-[280px]">
                <p className="font-medium mb-1">Understanding the Accuracy Curve</p>
                <p className="text-xs text-muted-foreground mb-2">Accuracy shows percentage of correct predictions. Higher is better.</p>
                <div className="text-xs space-y-1">
                  <p><span className="text-emerald-500 font-medium">Excellent:</span> 85%+ accuracy, steady growth</p>
                  <p><span className="text-[#2684FF] font-medium">Good:</span> 70-85% with upward trend</p>
                  <p><span className="text-amber-500 font-medium">Fair:</span> 50-70%, may need more data</p>
                </div>
              </TooltipContent>
            </Tooltip>
          </div>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="epoch" stroke="hsl(var(--muted-foreground))" fontSize={10} tickFormatter={(v) => `E${v}`} />
                <YAxis stroke="hsl(var(--muted-foreground))" fontSize={10} unit="%" domain={[0, 100]} />
                <RechartsTooltip
                  cursor={{ stroke: "hsl(var(--border))", strokeDasharray: "3 3" }}
                  contentStyle={{
                    backgroundColor: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "8px",
                    fontSize: "12px",
                  }}
                  labelStyle={{ color: "hsl(var(--foreground))" }}
                  labelFormatter={(label) => `Epoch ${label}`}
                  formatter={(value: number) => [`${value.toFixed(1)}%`, "Accuracy"]}
                />
                <Line type="monotone" dataKey="accuracy" stroke="#2684FF" strokeWidth={2} dot={chartData.length < 20} activeDot={{ r: 4, stroke: "#2684FF" }} name="Accuracy %" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </TooltipProvider>
  )
}

"use client"

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"
import type { TrainingMetrics } from "@/lib/types"

interface TrainingChartsProps {
  history: TrainingMetrics[]
}

export function TrainingCharts({ history }: TrainingChartsProps) {
  const chartData = history.map((h) => ({
    epoch: h.epoch,
    loss: Number(h.loss.toFixed(4)),
    accuracy: Number((h.accuracy * 100).toFixed(2)),
  }))

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      {/* Loss Chart */}
      <div className="rounded-xl border border-border bg-card p-4">
        <h4 className="mb-4 text-sm font-medium text-foreground">Loss Curve</h4>
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis dataKey="epoch" stroke="hsl(var(--muted-foreground))" fontSize={10} />
              <YAxis stroke="hsl(var(--muted-foreground))" fontSize={10} />
              <Tooltip
                cursor={false}
                contentStyle={{
                  backgroundColor: "hsl(var(--card))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "8px",
                  fontSize: "12px",
                }}
                labelStyle={{ color: "hsl(var(--foreground))" }}
                itemStyle={{ color: "hsl(var(--foreground))" }}
              />
              <Line type="monotone" dataKey="loss" stroke="#f87171" strokeWidth={2} dot={false} name="Loss" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Accuracy Chart - Updated color from cyan to Schema blue */}
      <div className="rounded-xl border border-border bg-card p-4">
        <h4 className="mb-4 text-sm font-medium text-foreground">Accuracy Curve</h4>
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis dataKey="epoch" stroke="hsl(var(--muted-foreground))" fontSize={10} />
              <YAxis stroke="hsl(var(--muted-foreground))" fontSize={10} unit="%" />
              <Tooltip
                cursor={false}
                contentStyle={{
                  backgroundColor: "hsl(var(--card))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "8px",
                  fontSize: "12px",
                }}
                labelStyle={{ color: "hsl(var(--foreground))" }}
                itemStyle={{ color: "hsl(var(--foreground))" }}
              />
              <Line type="monotone" dataKey="accuracy" stroke="#2684FF" strokeWidth={2} dot={false} name="Accuracy %" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}

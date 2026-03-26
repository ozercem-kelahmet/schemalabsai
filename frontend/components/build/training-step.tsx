"use client"

import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { TrainingCharts } from "./training-charts"
import { TrainingLogs } from "./training-logs"
import { useRef } from "react"
import type { TrainingMetrics } from "@/lib/types"
import { Pause, Play, Square, Loader2 } from "lucide-react"

interface TrainingStepProps {
  currentMetrics: TrainingMetrics | null
  history: TrainingMetrics[]
  logs: string[]
  status: "idle" | "initializing" | "training" | "paused" | "completing" | "failed"
  elapsedTime: number
  error?: string
  onRetry?: () => void
}

export function TrainingStep({
  currentMetrics,
  history,
  logs,
  status,
  elapsedTime,
  error,
  onRetry,
}: TrainingStepProps) {
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${String(secs).padStart(2, "0")}`
  }

  // Progress bar - only increases, never goes backwards
  const epoch = currentMetrics?.epoch || 0
  const total = currentMetrics?.epoch ? (currentMetrics?.epoch + 1) : 0
  const maxProgressRef = useRef(0)
  const rawProgress = total > 0 ? Math.min(99, (epoch / total) * 100) : 0
  if (rawProgress > maxProgressRef.current) maxProgressRef.current = rawProgress
  const progress = maxProgressRef.current

  if (status === "failed") {
    return (
      <div className="space-y-6">
        <Card className="border-red-500/50 bg-red-500/5">
          <CardContent className="pt-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="h-10 w-10 rounded-full bg-red-500/10 flex items-center justify-center">
                <Square className="h-5 w-5 text-red-500" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-red-500">Training Failed</h3>
                <p className="text-sm text-muted-foreground">The training process encountered an error</p>
              </div>
            </div>
            <div className="rounded-lg bg-red-500/10 border border-red-500/20 p-4 mb-4">
              <p className="text-sm text-red-400 font-mono">{error || "Unknown error"}</p>
            </div>
            <Button variant="outline" onClick={onRetry} className="border-red-500/30 text-red-400 hover:bg-red-500/10">
              Back to Configuration
            </Button>
          </CardContent>
        </Card>
        <TrainingLogs logs={logs} />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Progress Header */}
      <Card className="border-border bg-card">
        <CardContent className="p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              {status === "training" && (
                <div className="relative">
                  <div className="h-10 w-10 animate-spin rounded-full border-2 border-[#0052CC]/20 border-t-[#0052CC]" />
                  <Loader2 className="absolute inset-0 m-auto h-5 w-5 animate-pulse text-[#2684FF]" />
                </div>
              )}
              {status === "paused" && (
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-yellow-500/20">
                  <Pause className="h-5 w-5 text-yellow-500" />
                </div>
              )}
              {status === "initializing" && (
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-[#0052CC]/20">
                  <Loader2 className="h-5 w-5 animate-spin text-[#2684FF]" />
                </div>
              )}
              {status === "completing" && (
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-emerald-500/20">
                  <Loader2 className="h-5 w-5 animate-spin text-emerald-500" />
                </div>
              )}
              <div>
                <p className="font-medium capitalize text-foreground">
                  {status === "training" ? "Training in Progress" : status}
                </p>
                <p className="text-sm text-muted-foreground">
                  Epoch {currentMetrics?.epoch || 0} of {currentMetrics?.epoch ? currentMetrics.epoch + 1 : 0}
                </p>
              </div>
            </div>

          </div>

          <div className="mt-6">
            <div className="mb-2 flex justify-between text-sm">
              <span className="text-muted-foreground">Progress</span>
              <span className="font-mono text-[#2684FF]">{progress.toFixed(0)}%</span>
            </div>
            <Progress value={progress} className="h-2 bg-muted [&>div]:bg-[#0052CC]" />
          </div>

          {/* Live Metrics */}
          <div className="mt-6 grid grid-cols-4 gap-4">
            <div className="rounded-lg bg-muted/50 p-3">
              <p className="text-xs text-muted-foreground">Current Loss</p>
              <p className="mt-1 font-mono text-lg text-red-500">{currentMetrics?.loss.toFixed(4) || "—"}</p>
            </div>
            <div className="rounded-lg bg-muted/50 p-3">
              <p className="text-xs text-muted-foreground">Current Accuracy</p>
              <p className="mt-1 font-mono text-lg text-[#2684FF]">
                {currentMetrics ? `${(currentMetrics.accuracy * 100).toFixed(1)}%` : "—"}
              </p>
            </div>
            <div className="rounded-lg bg-muted/50 p-3">
              <p className="text-xs text-muted-foreground">Learning Rate</p>
              <p className="mt-1 font-mono text-lg text-foreground">{currentMetrics?.learningRate.toFixed(6) || "—"}</p>
            </div>
            <div className="rounded-lg bg-muted/50 p-3">
              <p className="text-xs text-muted-foreground">Elapsed Time</p>
              <p className="mt-1 font-mono text-lg text-foreground">{formatTime(elapsedTime)}</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Charts */}
      <TrainingCharts history={history} />

      {/* Logs */}
      <TrainingLogs logs={logs} />
    </div>
  )
}

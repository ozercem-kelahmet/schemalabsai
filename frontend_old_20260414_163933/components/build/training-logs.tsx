"use client"

import { useEffect, useRef } from "react"

interface TrainingLogsProps {
  logs: string[]
}

export function TrainingLogs({ logs }: TrainingLogsProps) {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight
    }
  }, [logs])

  return (
    <div className="rounded-xl border border-white/10 bg-[#0d0d0e]">
      <div className="flex items-center justify-between border-b border-white/10 px-4 py-2">
        <span className="text-xs font-medium text-gray-400">Training Logs</span>
        <div className="flex gap-1.5">
          <div className="h-2.5 w-2.5 rounded-full bg-red-500/80" />
          <div className="h-2.5 w-2.5 rounded-full bg-yellow-500/80" />
          <div className="h-2.5 w-2.5 rounded-full bg-green-500/80" />
        </div>
      </div>
      <div ref={containerRef} className="h-48 overflow-y-auto p-4 font-mono text-xs">
        {logs.map((log, i) => (
          <div key={i} className="text-gray-400">
            <span className="text-gray-600">[{String(i + 1).padStart(3, "0")}]</span> {log}
          </div>
        ))}
        <span className="inline-block h-4 w-2 animate-pulse bg-cyan-400" />
      </div>
    </div>
  )
}

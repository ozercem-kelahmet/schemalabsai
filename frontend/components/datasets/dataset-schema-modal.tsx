"use client"

import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { SourceBadge } from "./source-badge"
import type { Dataset } from "@/lib/types"
import { Table, Columns, FileSpreadsheet, RefreshCw } from "lucide-react"

interface DatasetSchemaModalProps {
  dataset: Dataset | null
  open: boolean
  onOpenChange: (open: boolean) => void
}

const verticalLabels: Record<string, string> = {
  finance: "Finance",
  healthcare: "Healthcare",
  "e-commerce": "E-commerce",
  marketing: "Marketing",
  hr: "HR",
  operations: "Operations",
}

export function DatasetSchemaModal({ dataset, open, onOpenChange }: DatasetSchemaModalProps) {
  if (!dataset) return null

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl border-border bg-card text-foreground sm:max-w-2xl">
        <DialogHeader>
          <div className="flex items-start justify-between gap-4">
            <div>
              <DialogTitle className="text-xl font-semibold text-foreground">{dataset.name}</DialogTitle>
              <DialogDescription className="mt-1 text-muted-foreground">{dataset.description}</DialogDescription>
            </div>
            <SourceBadge source={dataset.source} />
          </div>
        </DialogHeader>

        {/* Dataset Info */}
        <div className="flex flex-wrap items-center gap-3 border-b border-border pb-4">
          {verticalLabels[dataset.vertical] && (
            <span className="rounded-md bg-muted px-2 py-1 text-xs text-muted-foreground">
              {verticalLabels[dataset.vertical]}
            </span>
          )}
          {(dataset as any).sizeMB > 0 && (
            <span className="rounded-md bg-red-500/10 px-2 py-1 text-xs font-medium text-red-500 whitespace-nowrap">{(dataset as any).sizeMB < 1 ? ((dataset as any).sizeMB).toFixed(2) : ((dataset as any).sizeMB).toFixed(1)} MB</span>
          )}
          <div className="flex items-center gap-1.5 text-sm text-muted-foreground">
            <Table className="h-4 w-4" />
            <span className="font-mono text-foreground">{dataset.rows.toLocaleString()}</span>
            <span className="text-muted-foreground">rows</span>
          </div>
          <div className="flex items-center gap-1.5 text-sm text-muted-foreground">
            <Columns className="h-4 w-4" />
            <span className="font-mono text-foreground">{dataset.columns}</span>
            <span className="text-muted-foreground">cols</span>
          </div>
          {dataset.syncStatus && (
            <span
              className={`flex items-center gap-1.5 rounded-md px-2 py-1 text-xs ${
                dataset.syncStatus === "synced"
                  ? "bg-emerald-500/10 text-emerald-500"
                  : dataset.syncStatus === "pending"
                    ? "bg-yellow-500/10 text-yellow-500"
                    : "bg-orange-500/10 text-orange-500"
              }`}
            >
              <RefreshCw className="h-3 w-3" />
              {dataset.syncStatus === "synced"
                ? "Synced"
                : dataset.syncStatus === "pending"
                  ? "Pending Sync"
                  : "Outdated"}
            </span>
          )}
        </div>

        {/* Last Updated */}
        {dataset.lastUpdated && (
          <div className="text-xs text-muted-foreground">
            Last updated:{" "}
            {new Date(dataset.lastUpdated).toLocaleDateString("en-US", {
              month: "short",
              day: "numeric",
              year: "numeric",
              hour: "2-digit",
              minute: "2-digit",
            })}
          </div>
        )}

        {/* Full Schema */}
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <FileSpreadsheet className="h-4 w-4 text-[#2684FF]" />
            <h4 className="text-sm font-medium text-foreground">Full Schema</h4>
          </div>
          <div className="max-h-64 overflow-y-auto rounded-lg bg-muted/50 p-4">
            <div className="space-y-2">
              {dataset.schema.map((col) => (
                <div
                  key={col.name}
                  className="flex items-center justify-between gap-2 rounded-md bg-background px-3 py-2 border border-border"
                >
                  <div className="flex items-center gap-2 min-w-0 flex-1">
                    <span className="font-mono text-sm text-foreground truncate">{col.name}</span>
                    {col.description && <span className="text-xs text-muted-foreground whitespace-nowrap">{col.description.replace(/,\s*[\d.]+\s*MB$/, '')}</span>}
                  </div>
                  <div className="flex items-center gap-2">
                    {col.description?.includes('MB') && (
                      <span className="rounded-full bg-emerald-500/20 px-2 py-0.5 text-xs font-medium text-emerald-500 whitespace-nowrap shrink-0">
                        {col.description.match(/([\d.]+\s*MB)/)?.[1] || ''}
                      </span>
                    )}
                    <span className="rounded bg-[#0052CC]/20 px-2 py-0.5 text-xs font-medium text-[#2684FF]">
                      {col.type}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Sample Data Preview */}
        {dataset.sampleData && dataset.sampleData.length > 0 && (
          <div className="space-y-3">
            <h4 className="text-sm font-medium text-foreground">Sample Data</h4>
            <div className="overflow-x-auto rounded-lg bg-muted/50 border border-border">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-border">
                    {Object.keys(dataset.sampleData[0]).map((key) => (
                      <th key={key} className="px-3 py-2 text-left font-mono font-medium text-muted-foreground">
                        {key}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {dataset.sampleData.slice(0, 3).map((row, idx) => (
                    <tr key={idx} className="border-b border-border/50">
                      {Object.values(row).map((value, vidx) => (
                        <td key={vidx} className="px-3 py-2 font-mono text-foreground">
                          {String(value)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Action - just close, no select */}
        <div className="flex gap-3 pt-2">
          <Button
            onClick={() => onOpenChange(false)}
            className="w-full bg-[#0052CC]/10 text-[#2684FF] hover:bg-[#0052CC]/20 border border-[#0052CC]/20"
          >
            Close
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}

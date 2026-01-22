"use client"

import { Card, CardContent } from "@/components/ui/card"
import { SourceBadge } from "./source-badge"
import type { Dataset } from "@/lib/types"
import { Table, Columns } from "lucide-react"

interface DatasetCardProps {
  dataset: Dataset
  onViewSchema: (dataset: Dataset) => void
}

const verticalLabels: Record<string, string> = {
  finance: "Finance",
  healthcare: "Healthcare",
  "e-commerce": "E-commerce",
  marketing: "Marketing",
  hr: "HR",
  operations: "Operations",
}

export function DatasetCard({ dataset, onViewSchema }: DatasetCardProps) {
  return (
    <Card
      className="group border-border bg-card transition-all hover:border-[#0052CC]/30 hover:bg-accent/50 cursor-pointer"
      onClick={() => onViewSchema(dataset)}
    >
      <CardContent className="p-5">
        {/* Header */}
        <div className="flex items-start justify-between gap-3">
          <div className="flex-1">
            <h3 className="font-medium text-foreground group-hover:text-[#2684FF] transition-colors">{dataset.name}</h3>
            <p className="mt-1 line-clamp-2 text-sm text-muted-foreground">{dataset.description}</p>
          </div>
          <SourceBadge source={dataset.source} />
        </div>

        {/* Tags */}
        <div className="mt-4 flex flex-wrap gap-2">
          <span className="rounded-md bg-muted px-2 py-1 text-xs text-muted-foreground">
            {verticalLabels[dataset.vertical]}
          </span>
          <span className="rounded-md bg-muted px-2 py-1 text-xs capitalize text-muted-foreground">
            {dataset.complexity}
          </span>
          {dataset.syncStatus && (
            <span
              className={`rounded-md px-2 py-1 text-xs ${
                dataset.syncStatus === "synced"
                  ? "bg-emerald-500/10 text-emerald-500"
                  : dataset.syncStatus === "pending"
                    ? "bg-yellow-500/10 text-yellow-500"
                    : "bg-orange-500/10 text-orange-500"
              }`}
            >
              {dataset.syncStatus === "synced" ? "Synced" : dataset.syncStatus === "pending" ? "Pending" : "Outdated"}
            </span>
          )}
        </div>

        {/* Stats */}
        <div className="mt-4 flex items-center gap-4 border-t border-border pt-4">
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
        </div>

        {/* Schema Preview */}
        <div className="mt-4 rounded-lg bg-muted/50 p-3">
          <p className="mb-2 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">Schema Preview</p>
          <div className="flex flex-wrap gap-1.5">
            {dataset.schema.slice(0, 4).map((col) => (
              <span
                key={col.name}
                className="rounded bg-background px-1.5 py-0.5 font-mono text-xs text-muted-foreground border border-border"
              >
                {col.name}
              </span>
            ))}
            {dataset.schema.length > 4 && (
              <span className="rounded bg-background px-1.5 py-0.5 text-xs text-muted-foreground border border-border">
                +{dataset.columns - 4} more
              </span>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

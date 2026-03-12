"use client"

import type React from "react"

import { Checkbox } from "@/components/ui/checkbox"
import { Label } from "@/components/ui/label"
import type { DataSource, Vertical, Complexity, RowCount } from "@/lib/types"

interface DatasetFiltersProps {
  selectedSources: DataSource[]
  selectedVerticals: Vertical[]
  selectedComplexity: Complexity[]
  selectedRowCount: RowCount[]
  onSourceChange: (sources: DataSource[]) => void
  onVerticalChange: (verticals: Vertical[]) => void
  onComplexityChange: (complexity: Complexity[]) => void
  onRowCountChange: (rowCount: RowCount[]) => void
  availableSources?: DataSource[]
  availableVerticals?: string[]
  availableComplexity?: Complexity[]
  availableRowCount?: RowCount[]
}

const sources: { id: DataSource; label: string; icon: React.ReactNode }[] = [
  {
    id: "databricks",
    label: "Databricks",
    icon: (
      <svg viewBox="0 0 24 24" className="h-4 w-4 text-[#FF3621]" fill="currentColor">
        <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
      </svg>
    ),
  },
  {
    id: "supabase",
    label: "Supabase",
    icon: (
      <svg viewBox="0 0 24 24" className="h-4 w-4 text-[#3ECF8E]" fill="currentColor">
        <path d="M21.362 9.354H12V.396a.396.396 0 00-.716-.233L2.203 12.424l-.401.562a1.04 1.04 0 00.836 1.659H12v8.959a.396.396 0 00.716.233l9.081-12.261.401-.562a1.04 1.04 0 00-.836-1.66z" />
      </svg>
    ),
  },
  {
    id: "api",
    label: "API",
    icon: (
      <svg viewBox="0 0 24 24" className="h-4 w-4 text-[#2684FF]" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M4 6h16M4 12h16M4 18h16" />
      </svg>
    ),
  },
  {
    id: "google-drive",
    label: "Google Drive",
    icon: (
      <svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor">
        <path d="M12 2L4 14h4l4-7 4 7h4L12 2z" fill="#4285F4" />
        <path d="M4 14l4 8h8l4-8H4z" fill="#FBBC04" />
      </svg>
    ),
  },
  {
    id: "excel",
    label: "Excel",
    icon: <svg viewBox="0 0 24 24" className="h-4 w-4 text-[#217346]" fill="currentColor"><path d="M21.17 3H7.83A1.83 1.83 0 006 4.83v14.34A1.83 1.83 0 007.83 21h13.34A1.83 1.83 0 0023 19.17V4.83A1.83 1.83 0 0021.17 3zM15 17h-2l-2-3-2 3H7l3-5-3-5h2l2 3 2-3h2l-3 5 3 5z"/></svg>,
  },
  {
    id: "upload",
    label: "Uploaded Files",
    icon: (
      <svg viewBox="0 0 24 24" className="h-4 w-4 text-[#10B981]" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12" />
      </svg>
    ),
  },
  {
    id: "generated",
    label: "Generated Data",
    icon: (
      <svg viewBox="0 0 24 24" className="h-4 w-4 text-[#8B5CF6]" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
      </svg>
    ),
  },
  {
    id: "postgresql",
    label: "PostgreSQL",
    icon: (
      <svg viewBox="0 0 24 24" className="h-4 w-4 text-[#336791]" fill="currentColor">
        <path d="M17.128 0a10.134 10.134 0 00-2.755.403l-.063.02a10.922 10.922 0 00-1.612.556c-.108.043-.185.076-.242.1l-.066.029a9.923 9.923 0 00-.463.221 9.5 9.5 0 00-.612.334c-.108.064-.216.131-.324.2a9.064 9.064 0 00-1.852 1.585A8.63 8.63 0 007.458 5.4l-.028.064a8.487 8.487 0 00-.46 1.493l-.017.074a8.349 8.349 0 00-.144 2.32l.007.065.009.098a8.63 8.63 0 00.274 1.4l-.002-.012c.074.276.161.548.262.813l.015.041a8.724 8.724 0 001.79 2.835l.009.01c.067.07.135.14.205.207l.01.01a8.964 8.964 0 002.332 1.681l.048.024a9.088 9.088 0 001.79.683l.056.014a9.104 9.104 0 002.032.283h.111a9.184 9.184 0 002.032-.283l.056-.014a9.088 9.088 0 001.79-.683l.048-.024a8.964 8.964 0 002.332-1.681l.01-.01c.07-.067.138-.137.205-.207l.009-.01a8.724 8.724 0 001.79-2.835l.015-.041c.101-.265.188-.537.262-.813l-.002.012a8.63 8.63 0 00.274-1.4l.009-.098.007-.065a8.349 8.349 0 00-.144-2.32l-.017-.074a8.487 8.487 0 00-.46-1.493l-.028-.064a8.63 8.63 0 00-1.681-1.953 9.064 9.064 0 00-1.852-1.585c-.108-.069-.216-.136-.324-.2a9.5 9.5 0 00-.612-.334 9.923 9.923 0 00-.463-.221l-.066-.029c-.057-.024-.134-.057-.242-.1a10.922 10.922 0 00-1.612-.556l-.063-.02A10.134 10.134 0 0017.128 0z"/>
      </svg>
    ),
  },
  {
    id: "mysql",
    label: "MySQL",
    icon: <svg viewBox="0 0 24 24" className="h-4 w-4 text-[#4479A1]" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/></svg>,
  },
  {
    id: "mongodb",
    label: "MongoDB",
    icon: <svg viewBox="0 0 24 24" className="h-4 w-4 text-[#47A248]" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/></svg>,
  },
  {
    id: "snowflake",
    label: "Snowflake",
    icon: <svg viewBox="0 0 24 24" className="h-4 w-4 text-[#29B5E8]" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/></svg>,
  },
  {
    id: "pinecone",
    label: "Pinecone",
    icon: <svg viewBox="0 0 24 24" className="h-4 w-4 text-[#7B61FF]" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/></svg>,
  },
  {
    id: "weaviate",
    label: "Weaviate",
    icon: <svg viewBox="0 0 24 24" className="h-4 w-4 text-[#00C8A8]" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/></svg>,
  },
  {
    id: "chroma",
    label: "Chroma",
    icon: <svg viewBox="0 0 24 24" className="h-4 w-4 text-[#FFD700]" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/></svg>,
  },
  {
    id: "lancedb",
    label: "LanceDB",
    icon: <svg viewBox="0 0 24 24" className="h-4 w-4 text-[#3B82F6]" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/></svg>,
  },
  {
    id: "graphql",
    label: "GraphQL",
    icon: <svg viewBox="0 0 24 24" className="h-4 w-4 text-[#E10098]" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/></svg>,
  },
  {
    id: "rest",
    label: "REST API",
    icon: <svg viewBox="0 0 24 24" className="h-4 w-4 text-[#6366F1]" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/></svg>,
  },
  {
    id: "gcs",
    label: "Google Cloud Storage",
    icon: <svg viewBox="0 0 24 24" className="h-4 w-4 text-[#4285F4]" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/></svg>,
  },
  {
    id: "aws-s3",
    label: "AWS S3",
    icon: <svg viewBox="0 0 24 24" className="h-4 w-4 text-[#FF9900]" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/></svg>,
  },
]

const verticals: { id: Vertical; label: string }[] = [
  { id: "finance", label: "Finance" },
  { id: "healthcare", label: "Healthcare" },
  { id: "e-commerce", label: "E-commerce" },
  { id: "marketing", label: "Marketing" },
  { id: "hr", label: "HR" },
  { id: "operations", label: "Operations" },
]

const complexityOptions: { id: Complexity; label: string; description: string }[] = [
  { id: "simple", label: "Simple", description: "5-10 columns" },
  { id: "medium", label: "Medium", description: "10-25 columns" },
  { id: "advanced", label: "Advanced", description: "25+ columns" },
]

const rowCountOptions: { id: RowCount; label: string; description: string }[] = [
  { id: "small", label: "Small", description: "< 1K rows" },
  { id: "medium", label: "Medium", description: "1K-10K rows" },
  { id: "large", label: "Large", description: "10K+ rows" },
]

export function DatasetFilters({
  selectedSources,
  selectedVerticals,
  selectedComplexity,
  selectedRowCount,
  onSourceChange,
  onVerticalChange,
  onComplexityChange,
  onRowCountChange,
  availableSources,
  availableVerticals,
  availableComplexity,
  availableRowCount,
}: DatasetFiltersProps) {
  const toggleFilter = <T extends string>(current: T[], value: T, onChange: (values: T[]) => void) => {
    if (current.includes(value)) {
      onChange(current.filter((v) => v !== value))
    } else {
      onChange([...current, value])
    }
  }

  return (
    <div className="w-72 shrink-0 space-y-6 rounded-xl border border-border bg-card p-5">
      <div>
        <h3 className="mb-3 text-sm font-semibold text-foreground">Filters</h3>
        <p className="text-xs text-muted-foreground">Narrow down datasets</p>
      </div>

      {/* Source Filter */}
      <div>
        <h4 className="mb-3 text-xs font-medium uppercase tracking-wider text-muted-foreground">Source</h4>
        <div className="space-y-2">
          {(availableSources ? sources.filter(s => availableSources.includes(s.id)) : sources).map((source) => (
            <div key={source.id} className="flex items-center gap-3">
              <Checkbox
                id={`source-${source.id}`}
                checked={selectedSources.includes(source.id)}
                onCheckedChange={() => toggleFilter(selectedSources, source.id, onSourceChange)}
                className="border-border data-[state=checked]:bg-[#0052CC] data-[state=checked]:border-[#0052CC]"
              />
              <Label
                htmlFor={`source-${source.id}`}
                className="flex cursor-pointer items-center gap-2 text-sm text-foreground"
              >
                {source.icon}
                {source.label}
              </Label>
            </div>
          ))}
        </div>
      </div>



      {/* Vertical Filter */}
      {availableVerticals && availableVerticals.length > 0 && (
        <div>
          <h4 className="mb-3 text-xs font-medium uppercase tracking-wider text-muted-foreground">Vertical</h4>
          <div className="space-y-2">
            {availableVerticals.map((vertical) => (
              <div key={vertical} className="flex items-center gap-3">
                <Checkbox
                  id={`vertical-${vertical}`}
                  checked={selectedVerticals.includes(vertical as Vertical)}
                  onCheckedChange={() => {
                    const newVerticals = selectedVerticals.includes(vertical as Vertical)
                      ? selectedVerticals.filter(v => v !== vertical)
                      : [...selectedVerticals, vertical as Vertical]
                    onVerticalChange(newVerticals)
                  }}
                  className="border-border data-[state=checked]:bg-[#0052CC] data-[state=checked]:border-[#0052CC]"
                />
                <Label htmlFor={`vertical-${vertical}`} className="cursor-pointer text-sm text-foreground capitalize">
                  {vertical.replace(/-/g, " ")}
                </Label>
              </div>
            ))}
          </div>
        </div>
      )}
      {/* Complexity Filter */}
      <div>
        <h4 className="mb-3 text-xs font-medium uppercase tracking-wider text-muted-foreground">Complexity</h4>
        <div className="space-y-2">
          {(availableComplexity ? complexityOptions.filter(o => availableComplexity.includes(o.id)) : complexityOptions).map((option) => (
            <div key={option.id} className="flex items-center gap-3">
              <Checkbox
                id={`complexity-${option.id}`}
                checked={selectedComplexity.includes(option.id)}
                onCheckedChange={() => toggleFilter(selectedComplexity, option.id, onComplexityChange)}
                className="border-border data-[state=checked]:bg-[#0052CC] data-[state=checked]:border-[#0052CC]"
              />
              <Label htmlFor={`complexity-${option.id}`} className="cursor-pointer text-sm text-foreground">
                <span>{option.label}</span>
                <span className="ml-1 text-xs text-muted-foreground">({option.description})</span>
              </Label>
            </div>
          ))}
        </div>
      </div>

      {/* Row Count Filter */}
      <div>
        <h4 className="mb-3 text-xs font-medium uppercase tracking-wider text-muted-foreground">Row Count</h4>
        <div className="space-y-2">
          {(availableRowCount ? rowCountOptions.filter(o => availableRowCount.includes(o.id)) : rowCountOptions).map((option) => (
            <div key={option.id} className="flex items-center gap-3">
              <Checkbox
                id={`rows-${option.id}`}
                checked={selectedRowCount.includes(option.id)}
                onCheckedChange={() => toggleFilter(selectedRowCount, option.id, onRowCountChange)}
                className="border-border data-[state=checked]:bg-[#0052CC] data-[state=checked]:border-[#0052CC]"
              />
              <Label htmlFor={`rows-${option.id}`} className="cursor-pointer text-sm text-foreground">
                <span>{option.label}</span>
                <span className="ml-1 text-xs text-muted-foreground">({option.description})</span>
              </Label>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

import { DatasetGrid } from "@/components/datasets/dataset-grid"
import { Database } from "lucide-react"

export default function DatabasePage() {
  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[#0052CC]/10 dark:bg-[#0052CC]/20">
            <Database className="h-5 w-5 text-[#0052CC] dark:text-[#2684FF]" />
          </div>
          <div>
            <h1 className="text-xl font-semibold text-foreground">Database</h1>
            <p className="text-sm text-muted-foreground">Explore connected data sources across your infrastructure</p>
          </div>
        </div>
      </div>

      {/* Info Banner */}
      <div className="rounded-xl border border-[#0052CC]/20 bg-[#0052CC]/5 p-4">
        <p className="text-sm text-foreground">
          <span className="font-medium text-[#0052CC] dark:text-[#2684FF]">Multi-Source Connectivity:</span>{" "}
          <span className="text-muted-foreground">
            Connect and manage data from databases, APIs, cloud storage, and file uploads. Click any dataset to view its schema. Use the{" "}
            <span className="font-medium text-foreground">+ Connect</span> button to add new data sources, or{" "}
            <span className="font-medium text-foreground">+ Generate</span> new ones. <span className="italic text-amber-600 dark:text-amber-400">Don't upload sensitive personal data, confidential info, or anything you don't have rights to use.</span>
          </span>
        </p>
      </div>

      {/* Dataset Grid with Filters */}
      <DatasetGrid />
    </div>
  )
}

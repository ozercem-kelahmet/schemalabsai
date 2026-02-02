import { Card, CardContent } from "@/components/ui/card"
import type { Model } from "@/lib/types"
import { Calendar, TrendingUp, Zap, Box } from "lucide-react"
import { Button } from "@/components/ui/button"

interface ModelInfoCardProps {
  model: Model
}

export function ModelInfoCard({ model }: ModelInfoCardProps) {
  return (
    <Card className="border-border bg-card">
      <CardContent className="p-4">
        <div>
          <h3 className="font-medium text-foreground">{model.name}</h3>
          <p className="mt-1 text-sm text-muted-foreground line-clamp-2">{model.description}</p>
        </div>

        <div className="mt-4 grid grid-cols-2 gap-3">
          <div className="rounded-lg bg-muted/50 p-2.5">
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <TrendingUp className="h-3 w-3" />
              Accuracy
            </div>
            <p className="mt-1 font-mono text-sm font-semibold text-emerald-400">
              {(model.accuracy * 100).toFixed(1)}%
            </p>
          </div>
          <div className="rounded-lg bg-muted/50 p-2.5">
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <Zap className="h-3 w-3" />
              Sync
            </div>
            <p className="mt-1 text-sm font-medium capitalize text-foreground">{model.syncMode}</p>
          </div>
        </div>

        {/* Real-time update notification */}
        {model.pendingUpdates && model.pendingUpdates.length > 0 && (
          <div className="mt-3 rounded-lg border border-amber-500/30 bg-amber-500/10 p-2.5">
            <p className="text-xs font-medium text-amber-400">Data Update Available</p>
            <p className="mt-1 text-xs text-muted-foreground">{model.pendingUpdates[0].message}</p>
            <Button
              size="sm"
              className="mt-2 h-6 gap-1 bg-amber-500/20 text-amber-400 hover:bg-amber-500/30 text-xs w-full"
            >
              Update Model
            </Button>
          </div>
        )}

        <div className="mt-3 rounded-lg bg-muted/50 p-2.5">
          <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <Calendar className="h-3 w-3" />
            Created
          </div>
          <p className="mt-1 text-sm text-foreground">
            {model.createdAt.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}
          </p>
        </div>

        {model.baseModel && (
          <div className="mt-3 rounded-lg bg-muted/50 p-2.5">
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <Box className="h-3 w-3" />
              Base Model
            </div>
            <p className="mt-1 text-sm font-medium text-foreground">{model.baseModel}</p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

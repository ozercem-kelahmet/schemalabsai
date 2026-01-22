import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Database, Upload, MessageSquare, Key, Check, Clock } from "lucide-react"

const activities = [
  {
    id: 1,
    type: "upload",
    title: "sales_data_2024.csv",
    description: "12,450 rows processed",
    time: "2 minutes ago",
    status: "success",
    icon: Upload,
  },
  {
    id: 2,
    type: "playground",
    title: "Product Analysis Session",
    description: "Compared GPT-4o vs Claude 3.5",
    time: "15 minutes ago",
    status: "success",
    icon: MessageSquare,
  },
  {
    id: 3,
    type: "connection",
    title: "PostgreSQL Connected",
    description: "schema-prod-db.aws.com",
    time: "1 hour ago",
    status: "success",
    icon: Database,
  },
  {
    id: 4,
    type: "api",
    title: "API Key Generated",
    description: "Production key for mobile app",
    time: "3 hours ago",
    status: "success",
    icon: Key,
  },
]

export function RecentActivity() {
  return (
    <Card className="bg-card border-border h-full">
      <CardHeader className="pb-2 pt-4 px-4">
        <CardTitle className="text-sm font-semibold text-foreground">Recent Activity</CardTitle>
        <CardDescription className="text-xs">Your latest actions and events</CardDescription>
      </CardHeader>
      <CardContent className="px-4 pb-4">
        <div className="space-y-2">
          {activities.map((activity) => (
            <div
              key={activity.id}
              className="flex items-start gap-3 rounded-lg border border-border bg-secondary/30 p-2"
            >
              <div className="rounded-md bg-secondary p-1.5">
                <activity.icon className="h-3.5 w-3.5 text-muted-foreground" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-1.5">
                  <p className="text-xs font-medium text-foreground truncate">{activity.title}</p>
                  {activity.status === "success" && <Check className="h-3 w-3 text-success shrink-0" />}
                </div>
                <p className="text-xs text-muted-foreground truncate">{activity.description}</p>
              </div>
              <div className="flex items-center gap-1 text-xs text-muted-foreground whitespace-nowrap">
                <Clock className="h-3 w-3" />
                {activity.time}
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

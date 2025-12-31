import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Database, Zap, Key, MessageSquare } from "lucide-react"

const stats = [
  {
    name: "Data Sources",
    value: "12",
    change: "+2 this week",
    icon: Database,
  },
  {
    name: "API Calls",
    value: "45.2K",
    change: "+12.5% from last month",
    icon: Zap,
  },
  {
    name: "Active Keys",
    value: "8",
    change: "3 endpoints",
    icon: Key,
  },
  {
    name: "Playground Sessions",
    value: "234",
    change: "Last 7 days",
    icon: MessageSquare,
  },
]

export function StatsCards() {
  return (
    <div className="grid gap-3 grid-cols-2 lg:grid-cols-4">
      {stats.map((stat) => (
        <Card key={stat.name} className="bg-card border-border">
          <CardHeader className="flex flex-row items-center justify-between pb-1 pt-3 px-4">
            <CardTitle className="text-xs font-medium text-muted-foreground">{stat.name}</CardTitle>
            <stat.icon className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent className="px-4 pb-3">
            <div className="text-xl font-bold text-foreground">{stat.value}</div>
            <p className="text-xs text-muted-foreground">{stat.change}</p>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}

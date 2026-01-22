"use client"
import { Database, Play, Terminal, Webhook } from "lucide-react"
import Link from "next/link"

const actions = [
  {
    title: "Connect Data",
    description: "Upload CSVs, PDFs, or connect to Postgres, Pinecone, Databricks, and more.",
    icon: Database,
    href: "/data-sources",
  },
  {
    title: "Open Playground",
    description: "Compare LLMs with your data.",
    icon: Play,
    href: "/playground",
  },
  {
    title: "Create API Keys",
    description: "Generate secure keys to access Schema model via SDK or CLI.",
    icon: Terminal,
    href: "/api-keys",
  },
  {
    title: "Create Endpoints",
    description: "Generate endpoints for your applications.",
    icon: Webhook,
    href: "/endpoints",
  },
]

export function QuickActions() {
  return (
    <div className="grid gap-4 sm:gap-6 grid-cols-1 sm:grid-cols-2 lg:grid-cols-4">
      {actions.map((action) => (
        <Link key={action.title} href={action.href}>
          <div className="group flex flex-col gap-3 sm:gap-4 rounded-xl border border-border bg-card p-4 sm:p-6 transition-all hover:border-accent/50 hover:shadow-lg h-full">
            <div className="rounded-lg bg-secondary/50 p-2.5 sm:p-3 w-fit">
              <action.icon className="h-5 w-5 sm:h-6 sm:w-6 text-foreground" />
            </div>
            <div className="space-y-1.5 sm:space-y-2">
              <h3 className="text-base sm:text-lg font-semibold text-foreground group-hover:text-accent transition-colors">
                {action.title}
              </h3>
              <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">{action.description}</p>
            </div>
          </div>
        </Link>
      ))}
    </div>
  )
}

export function TerminalQuickStart() {
  return (
    <div className="rounded-xl overflow-hidden border border-border shadow-lg w-full">
      {/* Terminal window chrome */}
      <div className="bg-[#1e1e1e] px-3 sm:px-4 py-2 sm:py-3 flex items-center gap-2 border-b border-border/50">
        <div className="flex gap-1.5 sm:gap-2">
          <div className="w-2.5 h-2.5 sm:w-3 sm:h-3 rounded-full bg-[#ff5f56]" />
          <div className="w-2.5 h-2.5 sm:w-3 sm:h-3 rounded-full bg-[#ffbd2e]" />
          <div className="w-2.5 h-2.5 sm:w-3 sm:h-3 rounded-full bg-[#27ca40]" />
        </div>
        <div className="flex-1 text-center">
          <span className="text-muted-foreground text-xs sm:text-sm font-mono">bash — 80x24</span>
        </div>
      </div>

      {/* Terminal content */}
      <div className="bg-[#0a0a0a] p-4 sm:p-6 lg:p-8 font-mono text-xs sm:text-sm leading-loose overflow-x-auto">
        <div className="space-y-2 min-w-0">
          <p className="text-[#6a9955]"># Install the SDK</p>
          <p className="whitespace-nowrap">
            <span className="text-[#4ec9b0]">$</span>
            <span className="text-[#d4d4d4]"> pip install schema-ai</span>
          </p>
        </div>

        <div className="space-y-2 mt-4 sm:mt-6 min-w-0">
          <p className="text-[#6a9955]"># Initialize client</p>
          <p className="whitespace-nowrap">
            <span className="text-[#c586c0]">import</span>
            <span className="text-[#d4d4d4]"> schema </span>
            <span className="text-[#c586c0]">as</span>
            <span className="text-[#d4d4d4]"> sc</span>
          </p>
          <p className="whitespace-nowrap">
            <span className="text-[#9cdcfe]">client</span>
            <span className="text-[#d4d4d4]"> = </span>
            <span className="text-[#9cdcfe]">sc</span>
            <span className="text-[#d4d4d4]">.</span>
            <span className="text-[#dcdcaa]">Client</span>
            <span className="text-[#d4d4d4]">(</span>
            <span className="text-[#9cdcfe]">api_key</span>
            <span className="text-[#d4d4d4]">=</span>
            <span className="text-[#ce9178]">"YOUR_KEY"</span>
            <span className="text-[#d4d4d4]">)</span>
          </p>
        </div>
      </div>
    </div>
  )
}

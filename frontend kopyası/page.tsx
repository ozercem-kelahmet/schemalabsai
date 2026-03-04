"use client"

import { Sidebar } from "@/components/sidebar"
import { QuickActions, TerminalQuickStart } from "@/components/dashboard/quick-actions"

export default function DashboardPage() {
  return (
    <div className="flex min-h-screen bg-background">
      <Sidebar>
        <div className="p-4 sm:p-6 lg:p-8 pt-8 sm:pt-10 lg:pt-12 space-y-6 sm:space-y-8 w-full max-w-7xl mx-auto">
          <div className="space-y-2">
            <h1 className="text-xl sm:text-2xl font-semibold text-foreground">Welcome</h1>
            <p className="text-sm sm:text-base text-muted-foreground max-w-2xl">
              Schema Labs console to instantly develop AI agents and models using tabular data, and databases with
              state-of-the-art LLMs.
            </p>
          </div>

          <QuickActions />

          <div className="space-y-4">
            <h2 className="text-base sm:text-lg font-semibold text-foreground">Quick Start</h2>
            <TerminalQuickStart />
          </div>
        </div>
      </Sidebar>
    </div>
  )
}

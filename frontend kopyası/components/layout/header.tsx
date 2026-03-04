"use client"

import { usePathname } from "next/navigation"
import { Bell, Search } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"

const pageTitles: Record<string, { title: string; description: string }> = {
  "/": {
    title: "Dashboard",
    description: "Overview of your models and activity",
  },
  "/datasets": {
    title: "Datasets",
    description: "Select a dataset to train your model",
  },
  "/build": {
    title: "Model Builder",
    description: "Configure, train, and evaluate your model",
  },
  "/playground": {
    title: "Playground",
    description: "Interact with your trained models",
  },
}

export function Header() {
  const pathname = usePathname()
  const pageInfo = pageTitles[pathname] || pageTitles["/"]

  return (
    <header className="sticky top-0 z-30 flex h-16 items-center justify-between border-b border-white/10 bg-[#0A0A0B]/80 px-6 backdrop-blur-sm">
      <div>
        <h1 className="text-lg font-semibold text-white">{pageInfo.title}</h1>
        <p className="text-sm text-gray-500">{pageInfo.description}</p>
      </div>

      <div className="flex items-center gap-4">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-500" />
          <Input
            placeholder="Search..."
            className="w-64 border-white/10 bg-white/5 pl-9 text-sm text-white placeholder:text-gray-500 focus:border-cyan-500/50 focus:ring-cyan-500/20"
          />
        </div>
        <Button variant="ghost" size="icon" className="relative text-gray-400 hover:bg-white/5 hover:text-white">
          <Bell className="h-5 w-5" />
          <span className="absolute right-2 top-2 h-2 w-2 rounded-full bg-cyan-500" />
        </Button>
      </div>
    </header>
  )
}

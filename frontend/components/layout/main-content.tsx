"use client"

import type React from "react"
import { useSidebar } from "./sidebar"
import { cn } from "@/lib/utils"

export function MainContent({ children }: { children: React.ReactNode }) {
  const { collapsed } = useSidebar()

  return (
    <main 
      className={cn(
        "px-4 pt-16 pb-6 transition-all duration-300 min-h-screen",
        "sm:px-6",
        "md:pt-6",
        collapsed ? "md:pl-24" : "md:pl-72",
        "md:pr-6"
      )}
    >
      {children}
    </main>
  )
}

"use client"

import type React from "react"
import { Sidebar, SidebarProvider } from "@/components/layout/sidebar"
import { MainContent } from "@/components/layout/main-content"

export default function MainLayout({ children }: { children: React.ReactNode }) {
  return (
    <SidebarProvider>
      <Sidebar />
      <MainContent>{children}</MainContent>
    </SidebarProvider>
  )
}

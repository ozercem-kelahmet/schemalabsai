"use client"

import { Toaster } from "sonner"
import { VersionChecker } from "@/components/version-checker"
import type React from "react"
import { Sidebar, SidebarProvider } from "@/components/layout/sidebar"
import { MainContent } from "@/components/layout/main-content"

export default function MainLayout({ children }: { children: React.ReactNode }) {
  return (
    <SidebarProvider>
      <Sidebar />
      <MainContent><><VersionChecker />{children}
          <Toaster position="top-right" richColors /></></MainContent>
    </SidebarProvider>
  )
}

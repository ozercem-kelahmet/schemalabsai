"use client"

import type React from "react"
import { Sidebar, SidebarProvider } from "@/components/layout/sidebar"
import { MainContent } from "@/components/layout/main-content"
import { Toaster } from "sonner"

export default function MainLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <SidebarProvider>
      <Sidebar />
      <MainContent>{children}</MainContent>
      <Toaster position="top-right" />
    </SidebarProvider>
  )
}

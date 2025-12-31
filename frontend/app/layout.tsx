import type React from "react"
import type { Metadata, Viewport } from "next"
import { Geist, Geist_Mono } from "next/font/google"
import { Analytics } from "@vercel/analytics/next"
import { ThemeProvider } from "@/components/theme-provider"
import { QueryStoreProvider } from "@/lib/query-store"
import { Providers } from "@/components/providers"
import { AuthProvider } from "@/lib/auth"
import { ToastProvider } from "@/components/ui/toast"
import "./globals.css"

const _geist = Geist({ subsets: ["latin"] })
const _geistMono = Geist_Mono({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: "SchemaLabs - Data Language Model Platform",
  description:
    "End-to-end transformer based neural network that brings table-native understanding to AI development. Built by Schema Labs, NYC.",
  generator: "Schema Labs",
  icons: {
    icon: "/icon.svg",
    apple: "/icon.svg",
  },
}

export const viewport: Viewport = {
  themeColor: "#1a1a1a",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`font-sans antialiased`}>
        <ThemeProvider attribute="class" defaultTheme="light" enableSystem disableTransitionOnChange>
          <Providers><AuthProvider><QueryStoreProvider><ToastProvider>{children}</ToastProvider></QueryStoreProvider></AuthProvider></Providers>
        </ThemeProvider>
        <Analytics />
      </body>
    </html>
  )
}

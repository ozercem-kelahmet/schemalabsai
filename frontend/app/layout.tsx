import type React from "react"
import type { Metadata, Viewport } from "next"
import { Geist } from "next/font/google"
import { ThemeProvider } from "@/components/theme-provider"
import { QueryStoreProvider } from "@/lib/query-store"
import { Providers } from "@/components/providers"
import { AuthProvider } from "@/lib/auth"
import "./globals.css"

const geist = Geist({ subsets: ["latin"], display: "swap", preload: false })

export const metadata: Metadata = {
  title: "Schema Console",
  description: "End-to-end transformer based neural network that brings table-native understanding to AI development.",
  generator: "Schema Labs",
  icons: {
    icon: "/icon.svg",
    apple: "/apple-icon.png",
  },
}

export const viewport: Viewport = {
  themeColor: "#0A0A0B",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`font-sans antialiased bg-background text-foreground ${geist.className}`} suppressHydrationWarning>
        <ThemeProvider attribute="class" defaultTheme="dark" enableSystem disableTransitionOnChange>
          <Providers>
            <AuthProvider>
              <QueryStoreProvider>
                {children}
              </QueryStoreProvider>
            </AuthProvider>
          </Providers>
        </ThemeProvider>
      </body>
    </html>
  )
}

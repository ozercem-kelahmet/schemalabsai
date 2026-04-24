import type { Metadata } from "next"
import type React from "react"

export const metadata: Metadata = {
  title: "SchemaLabs - Authentication",
  description: "Sign in or create your SchemaLabs account",
}

export default function AuthLayout({
  children,
}: {
  children: React.ReactNode
}) {
  // Auth pages render without sidebar - just return children directly
  return <>{children}</>
}

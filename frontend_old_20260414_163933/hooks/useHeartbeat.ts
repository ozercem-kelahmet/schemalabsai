"use client"

import { useEffect } from "react"

export function useHeartbeat() {
  useEffect(() => {
    const sendHeartbeat = () => {
      fetch("/api/heartbeat", { method: "GET", credentials: "include" }).catch(() => {})
    }

    sendHeartbeat()
    const interval = setInterval(sendHeartbeat, 30000)

    const handleUnload = () => {
      navigator.sendBeacon("/api/heartbeat?offline=true")
    }
    window.addEventListener("beforeunload", handleUnload)

    return () => {
      clearInterval(interval)
      window.removeEventListener("beforeunload", handleUnload)
    }
  }, [])
}
